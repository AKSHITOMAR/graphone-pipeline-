"""JSONL checkpoints -> flat CSV tabs + one XLSX (import into Google Sheets) + optional direct gspread push."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

TABS = {"Startups": "startups.jsonl", "Products": "products.jsonl", "Research Papers": "papers.jsonl",
        "Jobs": "jobs.jsonl", "News": "news.jsonl"}


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, list):
            out[key] = "; ".join(map(str, v))
        else:
            out[key] = v
    return out


def read_jsonl(p: Path) -> list[dict]:
    return [flatten(json.loads(l)) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def mapping_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def export_all(out: Path) -> dict[str, list[dict]]:
    tabs = {name: read_jsonl(out / fn) for name, fn in TABS.items()}
    tabs["Entity Mapping Log"] = mapping_rows(out / "entity_mapping.jsonl")
    from openpyxl import Workbook
    wb = Workbook()
    wb.remove(wb.active)
    for name, rows in tabs.items():
        cols = list(dict.fromkeys(k for r in rows for k in r))
        with open(out / f"{name.replace(' ', '_')}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        ws = wb.create_sheet(name[:31])
        ws.append(cols)
        for r in rows:
            ws.append([("" if r.get(c) is None else str(r.get(c))[:32000]) for c in cols])
    wb.save(out / "graphone_output.xlsx")
    return tabs


def push_to_gsheet(tabs: dict[str, list[dict]], title: str = "GraphOne AI Engineer Output") -> str:
    """Needs `pip install gspread` and GOOGLE_APPLICATION_CREDENTIALS (service account). Returns sheet URL."""
    import gspread
    gc = gspread.service_account(filename=os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
    sh = gc.create(title)
    sh.share(None, perm_type="anyone", role="reader")
    for i, (name, rows) in enumerate(tabs.items()):
        cols = list(dict.fromkeys(k for r in rows for k in r)) or ["empty"]
        ws = sh.sheet1 if i == 0 else sh.add_worksheet(name, rows=max(len(rows) + 1, 10), cols=len(cols))
        ws.update_title(name)
        ws.update([cols] + [[("" if r.get(c) is None else str(r.get(c))[:45000]) for c in cols] for r in rows])
    return sh.url
