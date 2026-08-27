#!/usr/bin/env python3
"""Acquire provenance-bearing CAISO OASIS system actual and DAM forecasts."""

from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from src.backtesting.acquisition import caiso_request, sha256_file, write_manifest
from src.backtesting.strict_dataset import ACTUAL_ITEM, DAM_ITEM, ENTITY, operating_intervals


def download(session: requests.Session, url: str, destination: Path, retries: int = 4) -> str:
    sidecar = destination.with_suffix(destination.suffix + ".provenance.json")
    if destination.exists() or sidecar.exists():
        if not (destination.exists() and sidecar.exists()):
            raise RuntimeError(f"incomplete prior download state for {destination}")
        metadata = json.loads(sidecar.read_text())
        if metadata.get("request_url") != url or metadata.get("checksum") != sha256_file(destination):
            raise RuntimeError(f"existing raw response provenance mismatch: {destination}")
        return metadata["retrieved_at_utc"]
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=(20, 180))
            response.raise_for_status()
            if response.content.startswith(b"<?xml") and b"ERR_" in response.content:
                raise RuntimeError("CAISO OASIS returned an error document")
            destination.write_bytes(response.content)  # exact response bytes
            retrieved = datetime.now(timezone.utc).isoformat()
            sidecar.write_text(json.dumps({"request_url": url, "retrieved_at_utc": retrieved,
                                           "checksum": sha256_file(destination)}, indent=2) + "\n")
            return retrieved
        except (requests.RequestException, RuntimeError):
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def read_oasis_zip(path: Path) -> pd.DataFrame:
    frames = []
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            raw = archive.read(name)
            if name.lower().endswith(".csv"):
                frames.append(pd.read_csv(io.BytesIO(raw)))
            elif name.lower().endswith(".xml") and b"ERR_" in raw:
                raise RuntimeError(f"OASIS error member in {path}: {name}")
    if not frames:
        raise RuntimeError(f"no CSV result in OASIS response {path}")
    return pd.concat(frames, ignore_index=True)


def normalize(rows: pd.DataFrame, item: str) -> pd.DataFrame:
    columns = {str(c).upper(): c for c in rows.columns}
    report_item = columns.get("XML_DATA_ITEM", columns.get("DATA_ITEM"))
    required = {"INTERVALSTARTTIME_GMT", "TAC_AREA_NAME", "MARKET_RUN_ID", "MW"}
    if required - set(columns):
        raise RuntimeError(f"OASIS response missing columns: {sorted(required - set(columns))}")
    if report_item is None:
        raise RuntimeError("OASIS response missing columns: ['XML_DATA_ITEM']")
    expected_market = {DAM_ITEM: "DAM", ACTUAL_ITEM: "ACTUAL"}.get(item)
    if expected_market is None:
        raise ValueError(f"unsupported CAISO report item: {item}")
    out = pd.DataFrame({"interval_start_utc": rows[columns["INTERVALSTARTTIME_GMT"]],
                        "tac_area_name": rows[columns["TAC_AREA_NAME"]],
                        "data_item": rows[report_item],
                        "market_run_id": rows[columns["MARKET_RUN_ID"]],
                        "mw": rows[columns["MW"]]})
    out = out[(out.tac_area_name == ENTITY) & (out.data_item == item)
              & (out.market_run_id == expected_market)].copy()
    out["interval_start_utc"] = pd.to_datetime(out.interval_start_utc, utc=True)
    return out


def raw_identifiers(rows: pd.DataFrame) -> list[dict[str, str]]:
    """Return canonical raw identifiers from a current or historical response."""
    columns = {str(c).upper(): c for c in rows.columns}
    report_item = columns.get("XML_DATA_ITEM", columns.get("DATA_ITEM"))
    required = {"TAC_AREA_NAME", "MARKET_RUN_ID"}
    if required - set(columns) or report_item is None:
        missing = sorted(required - set(columns))
        if report_item is None:
            missing.append("XML_DATA_ITEM")
        raise RuntimeError(f"OASIS response missing raw identifier columns: {missing}")
    identifiers = rows[[columns["TAC_AREA_NAME"], report_item,
                        columns["MARKET_RUN_ID"]]].copy()
    identifiers.columns = ["TAC_AREA_NAME", "XML_DATA_ITEM", "MARKET_RUN_ID"]
    return identifiers.drop_duplicates().astype(str).to_dict("records")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", default="2025-12")
    parser.add_argument("--output-root", default="data/backtesting")
    parser.add_argument("--source-document", action="append", required=True,
                        help="Immutable local copy or URL for applicable CAISO documentation (repeatable)")
    args = parser.parse_args()
    expected = operating_intervals(args.month)
    history_times = expected.time_utc - pd.Timedelta(days=7)
    acquisition_times = pd.concat([expected.time_utc, history_times]).drop_duplicates().sort_values()
    month_dir = Path(args.output_root) / args.month
    raw_dir, processed = month_dir / "raw/caiso", month_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True); processed.mkdir(parents=True, exist_ok=True)
    records, frames = [], []
    with requests.Session() as session:
        for utc_day, group in acquisition_times.groupby(acquisition_times.dt.strftime("%Y-%m-%d")):
            start, end = group.min(), group.max() + pd.Timedelta(hours=1)
            for market_run_id in ("DAM", "ACTUAL"):
                url, params = caiso_request(start, end, market_run_id=market_run_id)
                target = raw_dir / f"oasis_sld_fcst_{market_run_id.lower()}_{utc_day}.zip"
                retrieved = download(session, url, target)
                response_rows = read_oasis_zip(target)
                frames.append(response_rows)
                records.append({"source": "caiso-oasis", "endpoint": url.split("?")[0],
                                "request_url": url, "request_parameters": params,
                                "retrieved_at_utc": retrieved, "source_filename": target.name,
                                "local_path": str(target), "checksum": sha256_file(target),
                                "raw_identifiers": raw_identifiers(response_rows),
                                "interval_start_utc": str(start),
                                "interval_end_utc_exclusive": str(end),
                                "applicable_procedure": "CAISO 1210 v19.8 (effective 2025-10-01)"})
    raw = pd.concat(frames, ignore_index=True)
    wanted = set(expected.time_utc)
    for item, filename in ((ACTUAL_ITEM, "actuals.csv"), (DAM_ITEM, "caiso_dam.csv")):
        out = normalize(raw, item)
        duplicates = out.interval_start_utc[out.interval_start_utc.duplicated()].astype(str).tolist()
        missing = sorted(str(x) for x in wanted - set(out.interval_start_utc))
        if duplicates or missing:
            raise RuntimeError(f"{item} incomplete: missing={missing}, duplicates={duplicates}")
        out[out.interval_start_utc.isin(wanted)].sort_values("interval_start_utc").to_csv(processed / filename, index=False)
        if item == ACTUAL_ITEM:
            history = out[out.interval_start_utc.isin(set(history_times))].copy()
            missing_history = set(history_times) - set(history.interval_start_utc)
            if missing_history:
                raise RuntimeError(f"previous-week actual history incomplete: {sorted(map(str, missing_history))}")
            pd.DataFrame({"observation_time_utc": history.interval_start_utc,
                          # A deliberately loose bound; every D-1 origin is six
                          # days later, and no retrieval timestamp is backdated.
                          "available_time_utc": history.interval_start_utc + pd.Timedelta(hours=24),
                          "load_mw": history.mw,
                          "availability_policy": "caiso_actual_plus_24h_v1"}).sort_values(
                              "observation_time_utc").to_csv(processed / "load_history.csv", index=False)
    write_manifest(month_dir / "acquisition_manifest.caiso.json", records, month=args.month,
                   source_documents=args.source_document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
