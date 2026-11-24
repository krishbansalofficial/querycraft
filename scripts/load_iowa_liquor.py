"""Load the public Iowa Liquor Sales data into a small star schema in SQLite (v2 part 6, live demo warehouse).

    python scripts/load_iowa_liquor.py --max-rows 2000000        # writes data/warehouse/iowa_liquor.sqlite

Source: the State of Iowa's Iowa Liquor Sales dataset (Alcoholic Beverages Division, one row per invoice line).
The official Socrata endpoint (data.iowa.gov/resource/m3tr-qhgy) returned 404 on 2026-11-23 after the portal moved
to the Iowa Data Hub, so the script falls back to a public Kaggle mirror of the 2022 calendar year
(leonczarlinski/iowa-liquor-sales, needs the Kaggle CLI token). The data is not redistributed with this repo.

--max-rows caps the invoice lines loaded so it fits on a laptop (rows are read in file order).
Tables: sales (fact), stores, items, vendors, categories, counties. Read only at serving time.
"""
import argparse
import re
import sqlite3
import subprocess
import sys
import urllib.request
from pathlib import Path

RAW = Path("data/warehouse/raw")
OUT = Path("data/warehouse/iowa_liquor.sqlite")
OFFICIAL = "https://data.iowa.gov/resource/m3tr-qhgy.csv?$limit={n}"
MIRROR = "leonczarlinski/iowa-liquor-sales"


def norm(c):
    return re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_")


def fetch(max_rows):
    RAW.mkdir(parents=True, exist_ok=True)
    have = sorted(RAW.glob("*.csv"))
    if have:
        return have[0]
    try:  # official source first
        dst = RAW / "official.csv"
        with urllib.request.urlopen(OFFICIAL.format(n=max_rows), timeout=60) as r:
            head = r.read(2048)
            if b"<html" in head.lower() or b"<!doctype" in head.lower():
                raise RuntimeError("official endpoint returned a web page, not CSV")
            dst.write_bytes(head + r.read())
        return dst
    except Exception as e:
        print(f"official source unavailable ({e}); using the Kaggle mirror {MIRROR}", flush=True)
    kaggle = Path(sys.executable).with_name("kaggle")
    subprocess.run([str(kaggle), "datasets", "download", MIRROR, "-p", str(RAW), "--unzip"], check=True)
    return sorted(RAW.glob("*.csv"))[0]


SCHEMA = """
CREATE TABLE counties (county_number INTEGER PRIMARY KEY, county TEXT);
CREATE TABLE stores (store_number INTEGER PRIMARY KEY, store_name TEXT, address TEXT, city TEXT, zip_code TEXT,
  county_number INTEGER REFERENCES counties(county_number));
CREATE TABLE vendors (vendor_number INTEGER PRIMARY KEY, vendor_name TEXT);
CREATE TABLE categories (category_number INTEGER PRIMARY KEY, category_name TEXT);
CREATE TABLE items (item_number TEXT PRIMARY KEY, item_description TEXT,
  category_number INTEGER REFERENCES categories(category_number), vendor_number INTEGER REFERENCES vendors(vendor_number),
  pack INTEGER, bottle_volume_ml INTEGER);
CREATE TABLE sales (invoice_line TEXT PRIMARY KEY, date TEXT, store_number INTEGER REFERENCES stores(store_number),
  item_number TEXT REFERENCES items(item_number), state_bottle_cost REAL, state_bottle_retail REAL,
  bottles_sold INTEGER, sale_dollars REAL, volume_sold_liters REAL);
"""

COLS = {  # our column -> normalized source column candidates
    "invoice_line": ["invoice_item_number"], "date": ["date"], "store_number": ["store_number"],
    "store_name": ["store_name"], "address": ["address"], "city": ["city"], "zip_code": ["zip_code"],
    "county_number": ["county_number"], "county": ["county"], "category_number": ["category"],
    "category_name": ["category_name"], "vendor_number": ["vendor_number"], "vendor_name": ["vendor_name"],
    "item_number": ["item_number"], "item_description": ["item_description"], "pack": ["pack"],
    "bottle_volume_ml": ["bottle_volume_ml"], "state_bottle_cost": ["state_bottle_cost"],
    "state_bottle_retail": ["state_bottle_retail"], "bottles_sold": ["bottles_sold"],
    "sale_dollars": ["sale_dollars"], "volume_sold_liters": ["volume_sold_liters"],
}


def _date(s):
    """MM/DD/YYYY (mirror) or ISO (official) -> YYYY-MM-DD."""
    s = str(s)
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", s)
    return f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}" if m else s[:10]


def _num(x, cast):
    try:
        return cast(str(x).replace("$", "").replace(",", "")) if str(x).strip() not in ("", "nan", "None") else None
    except ValueError:
        return None


def load(csv_path, max_rows, out=OUT):
    import pandas as pd
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    con = sqlite3.connect(out)
    con.executescript(SCHEMA)
    seen = {k: set() for k in ("counties", "stores", "vendors", "categories", "items")}
    n = 0
    for chunk in pd.read_csv(csv_path, chunksize=200_000, dtype=str, nrows=max_rows):
        chunk.columns = [norm(c) for c in chunk.columns]
        src = {k: next((c for c in v if c in chunk.columns), None) for k, v in COLS.items()}
        def g(r, k):  # missing cells arrive as float NaN even with dtype=str
            v = r.get(src[k]) if src[k] else None
            return None if v is None or (isinstance(v, float) and v != v) else v
        sales, dims = [], {k: [] for k in seen}
        for r in chunk.to_dict("records"):
            store, item = _num(g(r, "store_number"), int), str(g(r, "item_number") or "").strip()
            cty, ven, cat = _num(g(r, "county_number"), int), _num(g(r, "vendor_number"), int), _num(g(r, "category_number"), int)
            if cty is not None and cty not in seen["counties"]:
                seen["counties"].add(cty); dims["counties"].append((cty, (g(r, "county") or "").title()))
            if store is not None and store not in seen["stores"]:
                seen["stores"].add(store)
                dims["stores"].append((store, g(r, "store_name"), g(r, "address"), (g(r, "city") or "").title(),
                                       str(g(r, "zip_code") or "")[:5], cty))
            if ven is not None and ven not in seen["vendors"]:
                seen["vendors"].add(ven); dims["vendors"].append((ven, g(r, "vendor_name")))
            if cat is not None and cat not in seen["categories"]:
                seen["categories"].add(cat); dims["categories"].append((cat, g(r, "category_name")))
            if item and item not in seen["items"]:
                seen["items"].add(item)
                dims["items"].append((item, g(r, "item_description"), cat, ven, _num(g(r, "pack"), int),
                                      _num(g(r, "bottle_volume_ml"), int)))
            sales.append((g(r, "invoice_line"), _date(g(r, "date")), store, item or None,
                          _num(g(r, "state_bottle_cost"), float), _num(g(r, "state_bottle_retail"), float),
                          _num(g(r, "bottles_sold"), int), _num(g(r, "sale_dollars"), float),
                          _num(g(r, "volume_sold_liters"), float)))
        for t, rows in dims.items():
            if rows:
                con.executemany(f"INSERT OR IGNORE INTO {t} VALUES ({','.join('?' * len(rows[0]))})", rows)
        con.executemany("INSERT OR IGNORE INTO sales VALUES (?,?,?,?,?,?,?,?,?)", sales)
        con.commit()
        n += len(sales)
        print(f"loaded {n:,} invoice lines", flush=True)
    for ddl in ("CREATE INDEX sales_date ON sales(date)", "CREATE INDEX sales_store ON sales(store_number)",
                "CREATE INDEX sales_item ON sales(item_number)", "CREATE INDEX items_cat ON items(category_number)"):
        con.execute(ddl)
    con.commit()
    con.execute("VACUUM")
    con.close()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=2_000_000, help="cap on invoice lines (laptop friendly)")
    a = ap.parse_args()
    n = load(fetch(a.max_rows), a.max_rows)
    print(f"{OUT}: {n:,} invoice lines, {OUT.stat().st_size / 2**20:.0f} MB")


if __name__ == "__main__":
    main()
