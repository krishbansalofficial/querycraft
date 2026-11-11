"""Schema serialization (CREATE TABLE + example values) and question aware pruning.

Schemas are read straight from the .sqlite file so train, dev and the held back set use one code path.
Pruning only kicks in when the serialized schema exceeds the token budget: tables, then columns, are
dropped in order of lexical relevance to question + evidence. Key columns are never dropped.
"""
import json
import re
import sqlite3
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=None)
def _q(name):
    """Identifier as shown to the model: backticks when it is not a plain identifier or is a reserved word."""
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        try:  # let SQLite decide: reserved words (transaction, order, ...) fail as bare table + column names
            sqlite3.connect(":memory:").execute(f"SELECT {name} FROM (SELECT 1 AS {_iq(name)}) AS {name}")
            return name
        except sqlite3.Error:
            pass
    return f"`{name}`"


def _iq(name):
    """Identifier for our own queries: always double quoted."""
    return '"' + name.replace('"', '""') + '"'


@lru_cache(maxsize=None)
def read_schema(db_path: str, n_values: int = 3):
    """Return list of tables: {name, columns:[{name,type,pk,values}], fks:[(col, ref_table, ref_col)]}.
    Cached on disk next to the db because sampling values on multi GB dbs is slow."""
    cache = Path(db_path).with_suffix(f".schema{n_values}.json")
    if cache.exists():
        return json.loads(cache.read_text())
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.text_factory = lambda b: b.decode(errors="replace")
    tables = []
    names = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    for t in names:
        cols = []
        for _, cname, ctype, _, _, pk in conn.execute(f"PRAGMA table_info({_iq(t)})"):
            try:
                vals = [r[0] for r in conn.execute(
                    f"SELECT DISTINCT {_iq(cname)} FROM {_iq(t)} WHERE {_iq(cname)} IS NOT NULL LIMIT {n_values}")]
            except sqlite3.Error:
                vals = []
            vals = [v if not isinstance(v, str) else v[:40] for v in vals if not isinstance(v, bytes)]
            cols.append({"name": cname, "type": ctype or "", "pk": bool(pk), "values": vals})
        fks = [(r[3], r[2], r[4]) for r in conn.execute(f"PRAGMA foreign_key_list({_iq(t)})")]
        tables.append({"name": t, "columns": cols, "fks": fks})
    conn.close()
    try:
        cache.write_text(json.dumps(tables))
    except OSError:
        pass
    return tables


def serialize(tables, keep=None, bare=frozenset()) -> str:
    """keep: optional {table: set(columns)}; tables absent from keep are dropped.
    bare: tables rendered compactly (column names + keys only, no types or example values)."""
    return "\n".join(_blocks(tables, keep, bare))


def _cost(tables, keep, n_tokens, bare=frozenset()):
    # per table blocks repeat across questions on the same db, so a cached n_tokens makes this cheap
    return sum(n_tokens(b) + 1 for b in _blocks(tables, keep, bare))


def _blocks(tables, keep, bare=frozenset()):
    out = []
    kept_tables = {t["name"] for t in tables} if keep is None else set(keep)
    for t in tables:
        if t["name"] not in kept_tables:
            continue
        detail = t["name"] not in bare
        cols = [c for c in t["columns"] if keep is None or c["name"] in keep[t["name"]]]
        pks = [c["name"] for c in t["columns"] if c["pk"]]
        decls = []  # (declaration, trailing comment)
        for c in cols:
            inline_pk = " PRIMARY KEY" if c["pk"] and len(pks) == 1 else ""
            ex = f" -- e.g. {', '.join(repr(v) for v in c['values'])}" if c["values"] and detail else ""
            decls.append((f"{_q(c['name'])}{' ' + c['type'] if detail else ''}{inline_pk}".rstrip(), ex))
        if len(pks) > 1 and all(keep is None or p in keep[t["name"]] for p in pks):
            decls.append((f"PRIMARY KEY ({', '.join(map(_q, pks))})", ""))
        for col, rt, rc in t["fks"]:
            if rt in kept_tables and (keep is None or col in keep[t["name"]]):
                decls.append((f"FOREIGN KEY ({_q(col)}) REFERENCES {_q(rt)}({_q(rc) if rc else ''})", ""))
        body = "\n".join(f"  {d}{',' if i < len(decls) - 1 else ''}{ex}" for i, (d, ex) in enumerate(decls))
        out.append(f"CREATE TABLE {_q(t['name'])} (\n{body}\n);")
    return out


_TOK = re.compile(r"[a-z0-9]+")


def _words(s: str) -> set:
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(s))
    return {w for w in _TOK.findall(s.lower().replace("_", " ")) if len(w) > 1}


def _score(name, text_words, text_lower):
    w = _words(name)
    if not w:
        return 0.0
    exact = 2.0 if name.lower() in text_lower else 0.0
    return exact + len(w & text_words) / len(w)


def prune(tables, question, evidence, n_tokens, budget):
    """Fit the schema into `budget` tokens, degrading gracefully:
      1. full schema if it fits;
      2. else every table in compact form, upgrading the most relevant tables back to full detail;
      3. else (huge databases) the most relevant tables only, compact, then partial tables by column relevance.
    Relevance = lexical overlap of table/column names and example values with question + hint, plus a boost
    for tables one foreign key away from a relevant table (bridge tables needed for joins).
    Returns (keep, bare) for serialize()."""
    full = {t["name"]: {c["name"] for c in t["columns"]} for t in tables}
    if _cost(tables, None, n_tokens) <= budget:
        return full, frozenset()
    text = f"{question} {evidence}"
    tw, tl = _words(text), text.lower()
    col_score = {(t["name"], c["name"]): _score(c["name"], tw, tl) + 0.3 * any(_score(str(v), tw, tl) >= 1 for v in c["values"])
                 for t in tables for c in t["columns"]}
    base = {t["name"]: _score(t["name"], tw, tl) + max([col_score[(t["name"], c["name"])] for c in t["columns"]] or [0])
            for t in tables}
    nbrs = {t["name"]: set() for t in tables}
    for t in tables:
        for _, rt, _ in t["fks"]:
            if rt in nbrs:
                nbrs[t["name"]].add(rt)
                nbrs[rt].add(t["name"])
    t_score = {n: s + 0.5 * max([base[m] for m in nbrs[n]] or [0]) for n, s in base.items()}
    order = sorted(tables, key=lambda t: -t_score[t["name"]])
    all_bare = frozenset(full)

    if _cost(tables, full, n_tokens, all_bare) <= budget:  # stage 2
        bare = set(all_bare)
        for t in order:
            trial = bare - {t["name"]}
            if _cost(tables, full, n_tokens, trial) <= budget:
                bare = trial
        return full, frozenset(bare)

    keep = {}  # stage 3: whole compact tables by relevance, then partial ones
    for t in order:
        trial = {**keep, t["name"]: full[t["name"]]}
        if _cost(tables, trial, n_tokens, all_bare) <= budget:
            keep = trial
    for t in order:
        if t["name"] in keep or t_score[t["name"]] <= 0:
            continue
        keys = {c["name"] for c in t["columns"] if c["pk"]} | {fk[0] for fk in t["fks"]}
        cur = keys & full[t["name"]]
        if _cost(tables, {**keep, t["name"]: cur}, n_tokens, all_bare) > budget:
            continue
        for c in sorted(t["columns"], key=lambda c: -col_score[(t["name"], c["name"])]):
            if _cost(tables, {**keep, t["name"]: cur | {c["name"]}}, n_tokens, all_bare) <= budget:
                cur = cur | {c["name"]}
        keep[t["name"]] = cur
    if not keep:  # pathological: a single table larger than the budget
        t = order[0]
        cur = set()
        for c in sorted(t["columns"], key=lambda c: -col_score[(t["name"], c["name"])]):
            if _cost(tables, {t["name"]: cur | {c["name"]}}, n_tokens, all_bare) <= budget:
                cur |= {c["name"]}
        keep = {t["name"]: cur}
    return keep, all_bare
