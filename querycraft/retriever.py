"""Trained table retriever (v2 part 1): bi encoder + cross encoder reranker over table text.

Fixes the bottleneck in docs/plans/schema_linking.md: the trained selector drops a needed table on 18% of dev
questions. A retriever ranks every table of the database, so the number of tables shown (k) becomes a knob
chosen on val instead of whatever the selector happened to list.

    python -m querycraft.retriever train configs/retriever/table_retriever.yaml   # bi + cross encoder on train
    python -m querycraft.retriever rank  configs/retriever/table_retriever.yaml   # rank tables for val/dev/held back
    python -m querycraft.retriever sweep configs/retriever/table_retriever.yaml   # pick source, k, hop on val (vLLM)
    python -m querycraft.retriever link  configs/retriever/table_retriever.yaml   # write {split}_tables.jsonl
    python -m querycraft.run_eval configs/retriever/70_qwen1.5b_lora_r16_retrieved.yaml   # second pass

Each stage is its own process so the small encoders never share GPU memory with vLLM.
Table text = name, columns with types, and the 3 example values per column that the prompt already shows.
"""
import json
import random
import sys
from pathlib import Path

from .config import load_config, logged_stage, run_dir
from .data import load_split, selection
from .linker import set_metrics
from .prompts import fk_hop, gold_tables
from .schema import read_schema

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge query instruction
SPLITS = ("val", "dev", "heldback", "iowa")  # iowa: the warehouse demo set (scripts/iowa_questions.py)


def table_text(t, max_cols=60):
    cols = []
    for c in t["columns"][:max_cols]:
        vals = f" e.g. {', '.join(map(str, c['values']))}" if c["values"] else ""
        cols.append(f"{c['name']} {c['type']}".strip() + vals)
    return f"table {t['name']}: " + "; ".join(cols)


def query_text(ex):
    return f"{ex['question'].strip()} {(ex.get('evidence') or '').strip()}".strip()


def _tables(ex):
    return read_schema(ex["db_path"])


# ---------------------------------------------------------------- index (FAISS default, pgvector behind a flag)

class FaissIndex:
    """One exact inner product index per database (BIRD databases have at most ~70 tables)."""

    def __init__(self):
        self.idx, self.names = {}, {}

    def add(self, db, names, vecs):
        import faiss
        i = faiss.IndexFlatIP(vecs.shape[1])
        i.add(vecs.astype("float32"))
        self.idx[db], self.names[db] = i, list(names)

    def has(self, db):
        return db in self.idx

    def search(self, db, qvec, k):
        k = min(k, len(self.names[db]))
        s, ix = self.idx[db].search(qvec.reshape(1, -1).astype("float32"), k)
        return [(self.names[db][j], float(v)) for j, v in zip(ix[0], s[0])]

    def save(self, d):
        import faiss
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        for db, i in self.idx.items():
            faiss.write_index(i, str(d / f"{db}.faiss"))
        (d / "names.json").write_text(json.dumps(self.names))

    @classmethod
    def load(cls, d):
        import faiss
        self = cls()
        self.names = json.loads((Path(d) / "names.json").read_text())
        self.idx = {db: faiss.read_index(str(Path(d) / f"{db}.faiss")) for db in self.names}
        return self


class PgVectorIndex:
    """Same interface on Postgres + pgvector (config `index: pgvector`, DSN in QC_PG_DSN). Cosine distance <=>."""

    def __init__(self, dsn, dim=384):
        import psycopg
        self.conn = psycopg.connect(dsn, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.conn.execute(f"CREATE TABLE IF NOT EXISTS qc_tables (db text, name text, emb vector({dim}), "
                          "PRIMARY KEY (db, name))")

    @staticmethod
    def _v(x):
        return "[" + ",".join(f"{float(a):.6f}" for a in x) + "]"

    def add(self, db, names, vecs):
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM qc_tables WHERE db = %s", (db,))
            cur.executemany("INSERT INTO qc_tables VALUES (%s, %s, %s::vector)",
                            [(db, n, self._v(v)) for n, v in zip(names, vecs)])

    def has(self, db):
        return self.conn.execute("SELECT 1 FROM qc_tables WHERE db = %s LIMIT 1", (db,)).fetchone() is not None

    def search(self, db, qvec, k):
        v = self._v(qvec)
        rows = self.conn.execute("SELECT name, 1 - (emb <=> %s::vector) FROM qc_tables WHERE db = %s "
                                 "ORDER BY emb <=> %s::vector LIMIT %s", (v, db, v, k))
        return [(n, float(s)) for n, s in rows]


def make_index(cfg):
    import os
    if cfg.get("index", "faiss") == "pgvector":
        return PgVectorIndex(os.environ["QC_PG_DSN"])
    return FaissIndex()


# ---------------------------------------------------------------- inference

class Retriever:
    """Loads the trained encoders once. rank_many(rows) -> [{'bi': [[table, score]], 'rerank': [[table, score]]}]."""

    def __init__(self, rdir, cfg, index=None, device=None):
        from sentence_transformers import CrossEncoder, SentenceTransformer
        self.cfg = cfg
        self.bi = SentenceTransformer(str(Path(rdir) / "bi"), device=device)
        self.cross = CrossEncoder(str(Path(rdir) / "cross"), device=device)
        self.index = index or make_index(cfg)
        self.texts = {}

    def _ensure(self, ex):
        db = ex["db_id"]
        if db not in self.texts:
            ts = _tables(ex)
            self.texts[db] = {t["name"].lower(): table_text(t) for t in ts}
            if not self.index.has(db):
                vec = self.bi.encode(list(self.texts[db].values()), normalize_embeddings=True, batch_size=64)
                self.index.add(db, list(self.texts[db]), vec)

    def rank_many(self, rows, batch=64):
        for ex in rows:
            self._ensure(ex)
        q = self.bi.encode([QUERY_PREFIX + query_text(ex) for ex in rows], normalize_embeddings=True, batch_size=batch)
        n = self.cfg["cross"]["candidates"]
        bis = [self.index.search(ex["db_id"], v, n) for ex, v in zip(rows, q)]
        pairs = [(query_text(ex), self.texts[ex["db_id"]][t]) for ex, b in zip(rows, bis) for t, _ in b]
        scores = iter(self.cross.predict(pairs, batch_size=batch, show_progress_bar=False).tolist()) if pairs else iter(())
        out = []
        for b in bis:
            rr = sorted(((t, next(scores)) for t, _ in b), key=lambda x: -x[1])
            out.append({"bi": [[t, round(s, 5)] for t, s in b], "rerank": [[t, round(s, 5)] for t, s in rr]})
        return out


def pick(ranked, tables, k, hop):
    """Top k tables of a ranked list, plus every table one foreign key away if hop."""
    names = {t for t, _ in ranked[:k]}
    return fk_hop(tables, names) if hop else names


# ---------------------------------------------------------------- training

def _by_db(rows):
    out = {}
    for ex in rows:
        out.setdefault(ex["db_id"], []).append(ex)
    return sorted(out.items())


def _bi_triplets(rows, base, rng):
    """(query, gold table, hardest non gold table of the same database under the base model)."""
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(base)
    groups = _by_db(rows)
    pool = [table_text(t) for _, exs in groups for t in _tables(exs[0])]
    trip = []
    for db, exs in groups:
        ts = _tables(exs[0])
        names = [t["name"].lower() for t in ts]
        texts = [table_text(t) for t in ts]
        tv = m.encode(texts, normalize_embeddings=True, batch_size=64)
        qv = m.encode([QUERY_PREFIX + query_text(ex) for ex in exs], normalize_embeddings=True, batch_size=64)
        for ex, q in zip(exs, qv):
            gold = gold_tables(ex["sql"]) & set(names)
            neg = [i for i in (tv @ q).argsort()[::-1] if names[i] not in gold]
            for g in sorted(gold):
                n = texts[neg[0]] if neg else rng.choice(pool)  # single table databases: a random other table
                trip.append({"anchor": QUERY_PREFIX + query_text(ex), "positive": texts[names.index(g)], "negative": n})
    del m
    return trip


def train_bi(cfg, out, rows):
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.sentence_transformer.losses import CachedMultipleNegativesRankingLoss
    from sentence_transformers.sentence_transformer.training_args import BatchSamplers
    b = cfg["bi"]
    trip = _bi_triplets(rows, b["base"], random.Random(cfg["seed"]))
    model = SentenceTransformer(b["base"])
    model.max_seq_length = b["max_seq"]
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out / "bi_ckpt"), num_train_epochs=b["epochs"], per_device_train_batch_size=b["batch"],
        learning_rate=float(b["lr"]), warmup_steps=0.1, bf16=True, seed=cfg["seed"], save_strategy="no",
        batch_sampler=BatchSamplers.NO_DUPLICATES, logging_steps=50, report_to="none")
    SentenceTransformerTrainer(model=model, args=args, train_dataset=Dataset.from_list(trip),
                               loss=CachedMultipleNegativesRankingLoss(model, mini_batch_size=b["mini_batch"])).train()
    model.save(str(out / "bi"))
    return {"bi_triplets": len(trip)}


def train_cross(cfg, out, rows):
    """Pairs: every gold table (label 1) plus the trained bi encoder's top candidates that are not gold (label 0)."""
    from datasets import Dataset
    from sentence_transformers import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments, SentenceTransformer
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
    c = cfg["cross"]
    bi = SentenceTransformer(str(out / "bi"))
    idx, pairs = FaissIndex(), []
    for db, exs in _by_db(rows):
        texts = {t["name"].lower(): table_text(t) for t in _tables(exs[0])}
        idx.add(db, list(texts), bi.encode(list(texts.values()), normalize_embeddings=True, batch_size=64))
        qv = bi.encode([QUERY_PREFIX + query_text(ex) for ex in exs], normalize_embeddings=True, batch_size=64)
        for ex, q in zip(exs, qv):
            gold = gold_tables(ex["sql"]) & set(texts)
            if not gold:
                continue
            cand = {t for t, _ in idx.search(db, q, c["candidates"])} | gold
            pairs += [{"query": query_text(ex), "passage": texts[t], "label": float(t in gold)} for t in sorted(cand)]
    del bi
    model = CrossEncoder(c["base"], num_labels=1, max_length=c["max_seq"])
    args = CrossEncoderTrainingArguments(
        output_dir=str(out / "cross_ckpt"), num_train_epochs=c["epochs"], per_device_train_batch_size=c["batch"],
        learning_rate=float(c["lr"]), warmup_steps=0.1, bf16=True, seed=cfg["seed"], save_strategy="no",
        logging_steps=100, report_to="none")
    CrossEncoderTrainer(model=model, args=args, train_dataset=Dataset.from_list(pairs),
                        loss=BinaryCrossEntropyLoss(model)).train()
    model.save(str(out / "cross"))
    return {"cross_pairs": len(pairs), "cross_positive_share": round(sum(p["label"] for p in pairs) / len(pairs), 4)}


# ---------------------------------------------------------------- stages

def stage_train(cfg):
    import torch
    out = run_dir(cfg)
    rows = load_split("train")
    with logged_stage(cfg, "train") as meta:
        torch.cuda.reset_peak_memory_stats()
        meta |= train_bi(cfg, out, rows)
        meta |= train_cross(cfg, out, rows)
        meta["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)


def stage_rank(cfg):
    import time
    out = run_dir(cfg) / "ranked"
    out.mkdir(exist_ok=True)
    with logged_stage(cfg, "rank") as meta:
        r = Retriever(run_dir(cfg), cfg)
        for split in SPLITS:
            rows = load_split(split)
            t0 = time.time()
            ranked = r.rank_many(rows)
            meta[f"{split}_ms_per_question"] = round(1000 * (time.time() - t0) / len(rows), 2)
            (out / f"{split}.jsonl").write_text("".join(json.dumps({"id": ex["id"], "db_id": ex["db_id"], **rk}) + "\n"
                                                        for ex, rk in zip(rows, ranked)))
        if isinstance(r.index, FaissIndex):
            r.index.save(run_dir(cfg) / "index")


def _ranked(cfg, split):
    return {(d := json.loads(l))["id"]: d for l in (run_dir(cfg) / "ranked" / f"{split}.jsonl").read_text().splitlines()}


def settings(cfg):
    s = cfg["sweep"]
    return [{"source": src, "k": k, "hop": hop} for src in s["sources"] for k in s["k"] for hop in s["hop"]]


def tag(st):
    return f"{st['source']}_k{st['k']}{'_hop' if st['hop'] else ''}"


def link_tables(rows, ranked, st):
    return [{"id": ex["id"], "tables": sorted(pick(ranked[ex["id"]][st["source"]], _tables(ex), st["k"], st["hop"])),
             "scores": ranked[ex["id"]][st["source"]]} for ex in rows]


def _metrics(rows, lt):
    return set_metrics([(set(l["tables"]), gold_tables(ex["sql"])) for l, ex in zip(lt, rows)])


@selection
def stage_sweep(cfg):
    """Second pass of the SFT model on val for every (source, k, hop); best val EX wins (ties: fewer tables)."""
    from .evaluate import score_file
    from .infer import Generator, predict
    sql_cfg = load_config(cfg["sql_config"])
    sw = run_dir(cfg) / "sweep"
    sw.mkdir(exist_ok=True)
    rows, ranked = load_split("val"), _ranked(cfg, "val")
    with logged_stage(cfg, "sweep") as meta:
        gen, res = None, {}
        for st in settings(cfg):
            t = tag(st)
            lt = link_tables(rows, ranked, st)
            (sw / f"{t}_val_tables.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lt))
            pf = sw / f"{t}_val_preds.jsonl"
            if not pf.exists():
                gen = gen or Generator(sql_cfg, sql_cfg["model"])
                c = {**sql_cfg, "prompt": {**sql_cfg["prompt"], "link_from": str(sw / (t + "_{split}_tables.jsonl"))}}
                predict(gen, c, "val", out_path=pf)
            res[t] = {**st, "val_ex": score_file(pf)["ex"], **_metrics(rows, lt)}
            print(t, res[t], flush=True)
        best = max(res.values(), key=lambda r: (r["val_ex"], -r["mean_predicted"]))
        (run_dir(cfg) / "sweep.json").write_text(json.dumps({"by_setting": res, "chosen": best}, indent=2))
        meta["chosen"] = best


def stage_link(cfg):
    chosen = json.loads((run_dir(cfg) / "sweep.json").read_text())["chosen"]
    st = {k: chosen[k] for k in ("source", "k", "hop")}
    ev = run_dir(cfg) / "eval"
    ev.mkdir(exist_ok=True)
    res = {"name": cfg["name"], "chosen": st}
    for split in SPLITS:
        rows, ranked = load_split(split), _ranked(cfg, split)
        lt = link_tables(rows, ranked, st)
        (ev / f"{split}_tables.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lt))
        res[split] = _metrics(rows, lt)
        res[f"{split}_by_setting"] = {tag(s): _metrics(rows, link_tables(rows, ranked, s)) for s in settings(cfg)}
    (run_dir(cfg) / "results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("chosen", *SPLITS)}, indent=1))


if __name__ == "__main__":
    stage, path = sys.argv[1], sys.argv[2]
    {"train": stage_train, "rank": stage_rank, "sweep": stage_sweep, "link": stage_link}[stage](load_config(path))
