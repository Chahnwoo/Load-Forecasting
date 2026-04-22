#!/usr/bin/env python3
"""
audit_caiso_workbooks.py

Simpler CAISO workbook audit:
- read the CAISO library page
- extract direct /documents/...xlsx links from the page itself
- download relevant files
- audit workbook/sheet coverage
"""

from __future__ import annotations

import os
import re
import sys
from datetime import date, datetime, timedelta
from urllib.parse import urljoin

import pandas as pd
import requests

CAISO_LOAD_LIBRARY_URL = "https://www.caiso.com/library/historical-ems-hourly-load"
BASE_URL = "https://www.caiso.com"

TZ_MARKET = "America/Los_Angeles"
TZ_UTC = "UTC"

CANONICAL_LOAD_COLS = ["caiso", "pge", "sce", "sdge", "vea", "mwd"]

LOAD_COLUMN_ALIASES = {
    "caiso": "caiso",
    "caiso total": "caiso",
    "total": "caiso",
    "iso": "caiso",
    "system": "caiso",
    "system total": "caiso",
    "pge": "pge",
    "pge tac": "pge",
    "pge-tac": "pge",
    "pg&e": "pge",
    "pg&e tac": "pge",
    "pg&e-tac": "pge",
    "sce": "sce",
    "sce tac": "sce",
    "sce-tac": "sce",
    "southern california edison": "sce",
    "sdge": "sdge",
    "sdg&e": "sdge",
    "sdge tac": "sdge",
    "sdge-tac": "sdge",
    "sdg&e tac": "sdge",
    "sdg&e-tac": "sdge",
    "vea": "vea",
    "vea tac": "vea",
    "vea-tac": "vea",
    "mwd": "mwd",
    "mwd tac": "mwd",
    "mwd-tac": "mwd",
}

DATE_COLUMN_ALIASES = {
    "date", "opr_dt", "opr date", "trading date", "trade date", "market date"
}
HR_COLUMN_ALIASES = {
    "hr", "hour", "opr_hr", "he", "hour ending", "he hour"
}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "caiso-load-audit/3.0"})


def normalize_colname(c: str) -> str:
    s = str(c).strip().lower()
    s = re.sub(r"[_\\-/]+", " ", s)
    s = re.sub(r"\\s+", " ", s)
    return s.strip()


def coerce_date_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_datetime(s, unit="D", origin="1899-12-30", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def market_naive_to_utc_naive(ts: pd.Series) -> pd.Series:
    ts = pd.to_datetime(ts)
    localized = ts.dt.tz_localize(
        TZ_MARKET,
        ambiguous="infer",
        nonexistent="shift_forward",
    )
    return localized.dt.tz_convert(TZ_UTC).dt.tz_localize(None)


def http_get_text(url: str, timeout: int = 60) -> str:
    r = SESSION.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def download_file(url: str, dest_path: str, timeout: int = 120) -> None:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        return
    r = SESSION.get(url, timeout=timeout)
    r.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(r.content)


def extract_xlsx_links_from_library_page(html: str) -> list[str]:
    """
    Extract direct CAISO document links like /documents/...xlsx from the library page.
    """
    patterns = [
        r'href="([^"]*/documents/[^"]+\\.xlsx)"',
        r"href='([^']*/documents/[^']+\\.xlsx)'",
        r'href="([^"]+\\.xlsx)"',
        r"href='([^']+\\.xlsx)'",
    ]

    links: list[str] = []
    for pattern in patterns:
        matches = re.findall(pattern, html, flags=re.IGNORECASE)
        links.extend(urljoin(BASE_URL, m) for m in matches)

    seen = set()
    out = []
    for u in links:
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


def xlsx_maybe_relevant(url: str, start_date: date, end_date: date) -> bool:
    years = re.findall(r"(?:19|20)\\d{2}", url)
    if not years:
        return True
    years_i = {int(y) for y in years}
    return any(start_date.year <= y <= end_date.year for y in years_i)


def find_header_row(raw: pd.DataFrame) -> int | None:
    scan = min(len(raw), 80)
    for i in range(scan):
        row = raw.iloc[i].astype(str).map(normalize_colname).tolist()
        has_date = any(x in DATE_COLUMN_ALIASES for x in row)
        has_hr = any(x in HR_COLUMN_ALIASES for x in row)
        if has_date and has_hr:
            return i
    return None


def standardize_load_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for c in df.columns:
        norm = normalize_colname(c)
        if norm in LOAD_COLUMN_ALIASES:
            rename_map[c] = LOAD_COLUMN_ALIASES[norm]
        elif norm in DATE_COLUMN_ALIASES:
            rename_map[c] = "date"
        elif norm in HR_COLUMN_ALIASES:
            rename_map[c] = "hr"

    df = df.rename(columns=rename_map).copy()
    df.columns = [normalize_colname(c) for c in df.columns]
    return df


def parse_candidate_sheet(xls: pd.ExcelFile, sheet: str) -> dict:
    result = {
        "sheet": sheet,
        "status": "unknown",
        "reason": "",
        "rows": 0,
        "load_cols": [],
        "min_time_utc": None,
        "max_time_utc": None,
    }

    try:
        raw = pd.read_excel(xls, sheet_name=sheet, header=None)
    except Exception as e:
        result["status"] = "failed"
        result["reason"] = f"read_raw_failed: {type(e).__name__}: {e}"
        return result

    header_row = find_header_row(raw)
    if header_row is None:
        result["status"] = "skipped"
        result["reason"] = "no_header_row_found"
        return result

    try:
        df = pd.read_excel(xls, sheet_name=sheet, header=header_row)
    except Exception as e:
        result["status"] = "failed"
        result["reason"] = f"read_with_header_failed: {type(e).__name__}: {e}"
        return result

    df = standardize_load_columns(df)

    if "date" not in df.columns or "hr" not in df.columns:
        result["status"] = "skipped"
        result["reason"] = f"missing_date_or_hr columns={list(df.columns)}"
        return result

    df["date"] = coerce_date_series(df["date"])
    df["hr"] = pd.to_numeric(df["hr"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["date", "hr"]).copy()

    if df.empty:
        result["status"] = "skipped"
        result["reason"] = "empty_after_date_hr_cleaning"
        return result

    df = df.sort_values(["date", "hr"]).copy()
    df["time_market"] = df["date"] + pd.to_timedelta(df["hr"].astype(int) - 1, unit="h")

    try:
        df["time_utc"] = market_naive_to_utc_naive(df["time_market"])
    except Exception as e:
        result["status"] = "failed"
        result["reason"] = f"tz_convert_failed: {type(e).__name__}: {e}"
        return result

    load_cols = [c for c in CANONICAL_LOAD_COLS if c in df.columns]
    if not load_cols:
        result["status"] = "skipped"
        result["reason"] = f"no_recognized_load_columns columns={list(df.columns)}"
        return result

    result["status"] = "parsed"
    result["reason"] = "ok"
    result["rows"] = len(df)
    result["load_cols"] = load_cols
    result["min_time_utc"] = df["time_utc"].min()
    result["max_time_utc"] = df["time_utc"].max()
    return result


def audit_workbook(path: str) -> tuple[dict, list[dict]]:
    xls = pd.ExcelFile(path)
    sheet_results = [parse_candidate_sheet(xls, sheet) for sheet in xls.sheet_names]
    parsed = [r for r in sheet_results if r["status"] == "parsed"]

    workbook_summary = {
        "file": os.path.basename(path),
        "num_sheets": len(xls.sheet_names),
        "num_parsed_sheets": len(parsed),
        "best_sheet": None,
        "best_load_cols": [],
        "best_rows": 0,
        "best_min_time_utc": None,
        "best_max_time_utc": None,
    }

    if parsed:
        parsed.sort(key=lambda r: (len(r["load_cols"]), r["rows"]), reverse=True)
        best = parsed[0]
        workbook_summary["best_sheet"] = best["sheet"]
        workbook_summary["best_load_cols"] = best["load_cols"]
        workbook_summary["best_rows"] = best["rows"]
        workbook_summary["best_min_time_utc"] = best["min_time_utc"]
        workbook_summary["best_max_time_utc"] = best["max_time_utc"]

    return workbook_summary, sheet_results


def month_starts(start_date: date, end_date: date) -> list[date]:
    months = []
    cur = date(start_date.year, start_date.month, 1)
    while cur <= end_date:
        months.append(cur)
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)
    return months


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python audit_caiso_workbooks.py START_DATE END_DATE")
        print("Example: python audit_caiso_workbooks.py 2026-01-01 2026-04-21")
        sys.exit(1)

    start_date = datetime.strptime(sys.argv[1], "%Y-%m-%d").date()
    end_date = datetime.strptime(sys.argv[2], "%Y-%m-%d").date()

    cache_dir = "./data/cache/caiso_load_xlsx"

    print("=" * 100)
    print("CAISO LIBRARY DISCOVERY")
    print("=" * 100)

    listing_html = http_get_text(CAISO_LOAD_LIBRARY_URL)
    xlsx_links = extract_xlsx_links_from_library_page(listing_html)
    xlsx_links = [u for u in xlsx_links if xlsx_maybe_relevant(u, start_date - timedelta(days=31), end_date + timedelta(days=31))]

    print(f"Final downloadable XLSX links found: {len(xlsx_links)}")
    for u in xlsx_links[:20]:
        print(u)
    if len(xlsx_links) > 20:
        print(f"... and {len(xlsx_links) - 20} more")

    if not xlsx_links:
        print("\nNo XLSX links were discovered from the library page HTML.")
        print("Save the page source and inspect whether CAISO changed the markup again.")
        return

    workbook_summaries = []
    all_sheet_rows = []

    print("\n" + "=" * 100)
    print("WORKBOOK PARSE AUDIT")
    print("=" * 100)

    for url in xlsx_links:
        fn = url.split("/")[-1]
        path = os.path.join(cache_dir, fn)
        print(f"\nAuditing workbook: {fn}")

        try:
            download_file(url, path)
            wb_summary, sheet_results = audit_workbook(path)
            workbook_summaries.append(wb_summary)

            for r in sheet_results:
                all_sheet_rows.append({"file": fn, **r})

            print(f"  Parsed sheets: {wb_summary['num_parsed_sheets']} / {wb_summary['num_sheets']}")
            print(f"  Best sheet: {wb_summary['best_sheet']}")
            print(f"  Best load columns: {wb_summary['best_load_cols']}")
            print(f"  Best rows: {wb_summary['best_rows']}")
            print(f"  Best UTC coverage: {wb_summary['best_min_time_utc']} -> {wb_summary['best_max_time_utc']}")

        except Exception as e:
            print(f"  FAILED workbook audit: {type(e).__name__}: {e}")
            workbook_summaries.append({
                "file": fn,
                "num_sheets": None,
                "num_parsed_sheets": 0,
                "best_sheet": None,
                "best_load_cols": [],
                "best_rows": 0,
                "best_min_time_utc": None,
                "best_max_time_utc": None,
            })

    wb_df = pd.DataFrame(workbook_summaries)
    sheets_df = pd.DataFrame(all_sheet_rows)

    print("\n" + "=" * 100)
    print("WORKBOOK SUMMARY TABLE")
    print("=" * 100)
    print(wb_df.to_string(index=False))

    print("\n" + "=" * 100)
    print("MONTH-BY-MONTH COVERAGE CHECK")
    print("=" * 100)

    for mstart in month_starts(start_date, end_date):
        if mstart.month == 12:
            mend = date(mstart.year + 1, 1, 1) - timedelta(days=1)
        else:
            mend = date(mstart.year, mstart.month + 1, 1) - timedelta(days=1)

        month_start_ts = pd.Timestamp(f"{mstart} 00:00:00")
        month_end_ts = pd.Timestamp(f"{mend} 23:00:00")

        mask = (
            wb_df["best_min_time_utc"].notna()
            & wb_df["best_max_time_utc"].notna()
            & (pd.to_datetime(wb_df["best_min_time_utc"]) <= month_end_ts)
            & (pd.to_datetime(wb_df["best_max_time_utc"]) >= month_start_ts)
        )
        covering = wb_df.loc[mask]

        print(f"\nMonth {mstart.strftime('%Y-%m')}:")
        print(f"  Expected UTC range: {month_start_ts} -> {month_end_ts}")
        if covering.empty:
            print("  No audited workbook appears to cover this month.")
        else:
            for _, row in covering.iterrows():
                print(
                    f"  {row['file']} | best_sheet={row['best_sheet']} | "
                    f"cols={row['best_load_cols']} | "
                    f"coverage={row['best_min_time_utc']} -> {row['best_max_time_utc']}"
                )

    wb_df.to_csv("caiso_workbook_audit_summary.csv", index=False)
    sheets_df.to_csv("caiso_sheet_audit_details.csv", index=False)

    print("\nSaved workbook summary to: caiso_workbook_audit_summary.csv")
    print("Saved sheet details to: caiso_sheet_audit_details.csv")


if __name__ == "__main__":
    main()