"""NSF award abstracts from the public Awards API (no key required).

Awards are fetched in monthly windows of the initial award date and cached one file
per month, so an interrupted download resumes where it stopped. Only project-level
fields are kept. Names, e-mail addresses and phone numbers of investigators and
programme officers, which the API also returns, are discarded at parse time.
"""

from __future__ import annotations

import calendar
import json
import time
from pathlib import Path

import pandas as pd

from rim.data.ids import make_doc_id
from rim.provenance import DatasetManifest, sha256_file, utc_now

API_URL = "https://api.nsf.gov/services/v1/awards.json"
PAGE_SIZE = 100
KEPT_FIELDS = (
    "id",
    "title",
    "abstractText",
    "date",
    "startDate",
    "dirAbbr",
    "divAbbr",
    "fundProgramName",
    "estimatedTotalAmt",
    "awardeeName",
    "transType",
)


def parse_award(raw: dict) -> dict | None:
    """One API record to one row. Returns None for records without an ID or any text."""
    award_id = str(raw.get("id") or "").strip()
    title = " ".join(str(raw.get("title") or "").split())
    abstract = " ".join(str(raw.get("abstractText") or "").split())
    if not award_id or not (title or abstract):
        return None
    date = pd.to_datetime(raw.get("date"), format="%m/%d/%Y", errors="coerce")
    amount = pd.to_numeric(raw.get("estimatedTotalAmt"), errors="coerce")
    return {
        "doc_id": make_doc_id("nsf", award_id),
        "title": title,
        "text": abstract,
        "year": int(date.year) if pd.notna(date) else None,
        "award_date": date.strftime("%Y-%m-%d") if pd.notna(date) else None,
        "directorate": raw.get("dirAbbr") or None,
        "division": raw.get("divAbbr") or None,
        "program": raw.get("fundProgramName") or None,
        "amount_usd": float(amount) if pd.notna(amount) else None,
        "awardee": raw.get("awardeeName") or None,
        "award_type": raw.get("transType") or None,
    }


def fetch_month(year: int, month: int, client, pause: float = 0.2, retries: int = 4) -> list[dict]:
    last_day = calendar.monthrange(year, month)[1]
    params = {
        "dateStart": f"{month:02d}/01/{year}",
        "dateEnd": f"{month:02d}/{last_day:02d}/{year}",
        "rpp": PAGE_SIZE,
    }
    rows: list[dict] = []
    offset = 0
    while True:
        for attempt in range(retries):
            try:
                response = client.get(API_URL, params={**params, "offset": offset}, timeout=120)
                response.raise_for_status()
                payload = response.json()["response"]
                break
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2.0 * (attempt + 1))
        awards = payload.get("award", [])
        rows.extend(r for r in (parse_award(a) for a in awards) if r is not None)
        total = int(payload.get("metadata", {}).get("totalCount", 0))
        offset += PAGE_SIZE
        if not awards or offset >= total:
            return rows
        time.sleep(pause)


def prepare(
    root: str | Path, first_year: int, last_year: int, log=print
) -> tuple[pd.DataFrame, DatasetManifest]:
    import httpx

    root = Path(root)
    raw_dir = root / "raw" / "nsf_awards"
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    with httpx.Client() as client:
        for year in range(first_year, last_year + 1):
            for month in range(1, 13):
                path = raw_dir / f"{year}-{month:02d}.jsonl"
                if not path.exists():
                    rows = fetch_month(year, month, client)
                    tmp = path.with_suffix(".tmp")
                    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
                    tmp.rename(path)
                    log(f"nsf {year}-{month:02d}: {len(rows)} awards")
                if path.stat().st_size:
                    frames.append(pd.read_json(path, lines=True, dtype={"doc_id": str}))
    awards = pd.concat(frames, ignore_index=True).drop_duplicates("doc_id", ignore_index=True)
    awards["year"] = awards["year"].astype("Int64")
    out = root / "processed" / "nsf_awards"
    out.mkdir(parents=True, exist_ok=True)
    awards.to_parquet(out / "corpus.parquet", index=False, compression="zstd")
    manifest = DatasetManifest(
        name="nsf_awards",
        source="NSF Awards API",
        source_url=API_URL,
        revision=f"award dates {first_year}-01-01 to {last_year}-12-31, retrieved {utc_now()}",
        license="Public domain (U.S. Government work)",
        license_url="https://www.nsf.gov/policies/copyright",
        redistributable=True,
        retrieved_at=utc_now(),
        n_documents=len(awards),
        files={"corpus.parquet": sha256_file(out / "corpus.parquet")},
        notes=(
            "The API serves the current state of each award, so abstracts and amounts "
            "can change between retrievals. Investigator and officer contact fields are "
            "not stored."
        ),
    )
    manifest.write(out / "manifest.json")
    return awards, manifest
