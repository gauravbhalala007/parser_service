# app.py
# Robust FastAPI Parser for Amazon DSP KPI PDFs with ranking & status bucket
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import io, re
import pdfplumber
import pandas as pd

app = FastAPI(title="Amazon DSP KPI Parser")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:4000",
        "http://127.0.0.1:4000",
        "http://localhost:5000",
        "http://127.0.0.1:5000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        # you can also temporarily allow all for local:
        # "*"
    ],
    allow_credentials=False,
    allow_methods=["*"],     # POST included
    allow_headers=["*"],     # Content-Type, etc.
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or your exact Hosting origin(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# MODELS
# ===============================
class DriverRow(BaseModel):
    transporter_id: str = Field(alias="Transporter ID")
    Delivered: Optional[float] = None
    DCR: Optional[float] = None
    POD: Optional[float] = None
    CC: Optional[float] = None
    CE: Optional[float] = None
    LoR_DPMO: Optional[float] = Field(None, alias="LoR DPMO")
    DNR_DPMO: Optional[float] = Field(None, alias="DNR DPMO")
    CDF_DPMO: Optional[float] = Field(None, alias="CDF DPMO")

    # Computed per Albert
    POD_Score: Optional[float] = None
    CC_Score: Optional[float] = None
    DCR_Score: Optional[float] = None
    CE_Score: Optional[float] = None
    LoR_Score: Optional[float] = None
    DNR_Score: Optional[float] = None
    CDF_Score: Optional[float] = None
    FinalScore: Optional[float] = None  # "TOTAL"

    # Added for UI
    rank: Optional[int] = None
    statusBucket: Optional[str] = None  # Fantastic / Great / Fair / Poor

class ParserSummary(BaseModel):
    overallScore: Optional[float] = None
    reliabilityScore: Optional[float] = None
    rankAtStation: Optional[int] = None
    stationCount: Optional[int] = None
    rankDeltaWoW: Optional[int] = None
    weekText: Optional[str] = None
    weekNumber: Optional[int] = None
    year: Optional[int] = None
    stationCode: Optional[str] = None  # e.g., DBY5 if present

    reliabilityNextDay: Optional[float] = None
    reliabilitySameDay: Optional[float] = None

class ParserResponse(BaseModel):
    count: int
    drivers: List[DriverRow]
    summary: Optional[ParserSummary] = None

# ===============================
# HELPERS
# ===============================
NBSP = "\u00A0"

def clean_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    # normalize common invisible separators to a space
    s = (
        s.replace("\u00A0", " ")  # NBSP
         .replace("\u2009", " ")  # THIN SPACE
         .replace("\u202F", " ")  # NARROW NBSP
         .replace("\u00AD", "")   # SOFT HYPHEN
         .replace("\n", " ")
         .replace("\r", " ")
    )
    return re.sub(r"\s+", " ", s).strip()

def keyize(s: str) -> str:
    """letters+digits only, lowercase; good for robust header matching"""
    return re.sub(r"[^a-z0-9]", "", s.lower())

def to_num(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = clean_str(x)
    if not s or s == "-":
        return None
    s = s.replace("%", "").replace(" ", "")
    # Handle German/US number formats
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    else:
        s = s.replace(",", ".")
    s = re.sub(r"[^0-9.\-]", "", s)
    try:
        return float(s)
    except:
        return None

def to_percent(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = clean_str(x)
    if not s or s == "-":
        return None
    if s.endswith("%"):
        return to_num(s[:-1])
    n = to_num(s)
    if n is None:
        return None
    return n * 100.0 if n < 1.0 else n

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def status_bucket(final_score: Optional[float]) -> str:
    if final_score is None:
        return "Unknown"
    s = float(final_score)
    if s >= 93:
        return "FANTASTIC_PLUS"
    if s >= 85:
        return "FANTASTIC"
    if s >= 70:
        return "GREAT"
    if s >= 50:
        return "FAIR"
    if s >= 0:
        return "POOR"
    # if somehow negative, keep it simple:
    return "POOR"
# ===============================
# KPI FORMULAS (Albert’s rules)
# ===============================
def compute_scores(row: Dict[str, Any]) -> Dict[str, Optional[float]]:
    # --- parse raw values (unchanged) ---
    pod = to_percent(row.get("POD"))
    cc  = to_percent(row.get("CC"))
    dcr = to_percent(row.get("DCR"))
    ce  = to_num(row.get("CE"))
    lor = to_num(row.get("LoR DPMO"))
    dnr = to_num(row.get("DNR DPMO"))
    cdf = to_num(row.get("CDF DPMO"))

    # --- per-metric scores (unchanged) ---
    POD_Score = None if pod is None else clamp(pod, 0, 100)
    CC_Score  = None if cc  is None else clamp(cc,  0, 100)
    DCR_Score = None if dcr is None else clamp(dcr, 0, 100)
    CE_Score  = None if ce  is None else clamp(100 - 50.0 * ce, 50, 100)
    LoR_Score = None if lor is None else clamp(max(70.0, 100.0 - (lor / 1200.0) * 30.0), 0, 100)
    DNR_Score = None if dnr is None else clamp(max(70.0, 100.0 - (dnr / 1200.0) * 30.0), 0, 100)
    CDF_Score = None if cdf is None else clamp(134.33333333333334 - 0.013333333333333334 * cdf, 0.0, 100.0)

    # --- Excel-style hard denominator (16.1) ---
    def z(x): return 0.0 if x is None else x
    numerator = (
        z(DCR_Score) +
        z(POD_Score) +
        3.0 * z(CC_Score) +
        5.0 * z(DNR_Score) +
        5.0 * z(LoR_Score) +
        (z(CE_Score) / 50.0) +
        z(CDF_Score)
    )
    FinalScore = numerator / 16.1

    return {
        "POD_Score": POD_Score, "CC_Score": CC_Score, "DCR_Score": DCR_Score,
        "CE_Score": CE_Score, "LoR_Score": LoR_Score, "DNR_Score": DNR_Score,
        "CDF_Score": CDF_Score, "FinalScore": FinalScore
    }

# ===============================
# HEADER NORMALIZATION
# ===============================
HEADER_MAP = {
    "transporter id": "Transporter ID",
    "zustellende-id": "Transporter ID",
    "associate id": "Transporter ID",
    "driver id": "Transporter ID",
    "dcr": "DCR",
    "pod": "POD",
    "cc": "CC",
    "ce": "CE",
    "lor dpmo": "LoR DPMO",
    "lor (dpmo)": "LoR DPMO",
    "dnr dpmo": "DNR DPMO",
    "dnr (dpmo)": "DNR DPMO",
    "cdf dpmo": "CDF DPMO",
    "cdf (dpmo)": "CDF DPMO",
    "cdfdpmo": "CDF DPMO",
    "cdfdpm0": "CDF DPMO",
    "CDF": "CDF DPMO",
    "cdf": "CDF DPMO",
    "CDF  DPMO": "CDF DPMO",
    "delivered": "Delivered",
    "zugestellte pakete": "Delivered",
}

def norm_col(c: Any) -> str:
    raw = clean_str(c)
    k = keyize(raw)
    # s = clean_str(c).lower()
    # s = re.sub(r"\s+", " ", s)
    # return HEADER_MAP.get(s, clean_str(c))
    return HEADER_MAP.get(k, raw)
    

# ===============================
# PDF TABLE EXTRACTION
# ===============================

def get_col(df: pd.DataFrame, canonical: str) -> Optional[str]:
    """
    Return the actual column name in df that matches the canonical header
    by keyized comparison, or None if not found.
    """
    want = keyize(canonical)
    for col in df.columns:
        if keyize(col) == want:
            return col
    return None


def extract_driver_rows(pdf: pdfplumber.PDF) -> List[Dict[str, Any]]:
    def is_header_like(row: List[Any]) -> bool:
        if not row:
            return False
        texty = sum(1 for c in row if re.search(r"[A-Za-z]", clean_str(c)))
        return texty >= 3

    def looks_like_transporter_id(x: Any) -> bool:
        s = clean_str(x)
        # Amazon IDs are typically uppercase letters+digits, ~10–16 chars
        return bool(re.match(r"^[A-Z0-9]{8,20}$", s))

    def coalesce_numbers(a, b):
        va = to_num(a)
        if va is not None:
            return va
        vb = to_num(b)
        return vb

    def merge_adjacent_cols_if_needed(df: pd.DataFrame, target_canonical: str) -> pd.DataFrame:
        want = keyize(target_canonical)
        cols = list(df.columns)
        for i in range(len(cols) - 1):
            k = keyize(str(cols[i]) + str(cols[i + 1]))
            if k == want:
                merged = []
                for _, row in df.iterrows():
                    merged.append(coalesce_numbers(row[cols[i]], row[cols[i + 1]]))
                df[target_canonical] = merged
                df = df.drop(columns=[cols[i], cols[i + 1]])
                return df
        return df

    def split_combo_headers(header: List[str]) -> List[str]:
        h = header[:]
        i = 0
        while i < len(h) - 1:
            cur = clean_str(h[i]); nxt = clean_str(h[i + 1])
            kcur = keyize(cur);    knxt = keyize(nxt)

            # 'CE CDF DPMO' + ''  →  'CE', 'CDF DPMO'
            if ("ce" in kcur and ("cdfdpmo" in kcur or "cdf" in kcur)) and (knxt == "" or nxt == ""):
                h[i] = "CE"
                h[i + 1] = "CDF DPMO"
                i += 2
                continue
            i += 1
        return h

    def build_df_with_header(header: List[str], raw_rows: List[List[Any]]) -> pd.DataFrame:
        # normalize header & pad/truncate rows
        header = [norm_col(h) for h in header]
        header = split_combo_headers(header)
        hlen = len(header)
        fixed = []
        for r in raw_rows:
            rr = list(r or [])
            if len(rr) < hlen:
                rr = rr + [""] * (hlen - len(rr))
            elif len(rr) > hlen:
                rr = rr[:hlen]
            fixed.append(rr)
        df = pd.DataFrame(fixed, columns=header)

        # rename using HEADER_MAP keys
        rename_map = {}
        for col in df.columns:
            kcol = keyize(col)
            for ksrc, vdst in HEADER_MAP.items():
                if keyize(ksrc) == kcol:
                    rename_map[col] = vdst
                    break
        if rename_map:
            df = df.rename(columns=rename_map)

        # if DPMO columns still missing, try adjacent merges
        if get_col(df, "CDF DPMO") is None:
            df = merge_adjacent_cols_if_needed(df, "CDF DPMO")
        if get_col(df, "LoR DPMO") is None:
            df = merge_adjacent_cols_if_needed(df, "LoR DPMO")
        if get_col(df, "DNR DPMO") is None:
            df = merge_adjacent_cols_if_needed(df, "DNR DPMO")

        print("PDF table columns:", list(df.columns))
        return df

    def split_ce_cdf_cell(val):
        if val is None:
            return (None, None)
        s = clean_str(val)
        nums = re.findall(r"[0-9.,\-]+", s)
        if len(nums) >= 2:
            return (to_num(nums[0]), to_num(nums[1]))
        return (to_num(s), None)

    rows: List[Dict[str, Any]] = []
    last_header: List[str] | None = None

    for page in pdf.pages:
        tables = page.extract_tables() or []
        for table in tables:
            if not table or len(table) < 1:
                continue

            raw0 = [clean_str(h) for h in table[0]]
            raw1 = [clean_str(h) for h in table[1]] if len(table) > 1 else None

            header: List[str] | None = None
            data_start = 1

            # Case A: this table has a header row (or two)
            if "Transporter ID" in raw0 or is_header_like(raw0):
                if raw1 and is_header_like(raw1):
                    # two-row header
                    combined = []
                    for a, b in zip(raw0, raw1):
                        if a and b: combined.append(f"{a} {b}".strip())
                        elif b:     combined.append(b)
                        else:       combined.append(a)
                    header = combined
                    data_start = 2
                else:
                    header = raw0
                    data_start = 1
                last_header = header[:]  # remember for next pages

            # Case B: continuation table without header → reuse last_header
            elif last_header is not None and table and looks_like_transporter_id(table[0][0]):
                header = last_header[:]
                data_start = 0  # first row is data

            # Otherwise, skip (definitions, legends, other tables)
            if header is None:
                continue

            df = build_df_with_header(header, table[data_start:])

            # Need Transporter ID
            tid_col = get_col(df, "Transporter ID")
            if tid_col is None:
                continue

            delivered_c = get_col(df, "Delivered")
            dcr_c       = get_col(df, "DCR")
            pod_c       = get_col(df, "POD")
            cc_c        = get_col(df, "CC")
            ce_c        = get_col(df, "CE")
            lor_c       = get_col(df, "LoR DPMO")
            dnr_c       = get_col(df, "DNR DPMO")
            cdf_c       = get_col(df, "CDF DPMO")

            for _, r in df.iterrows():
                tid = clean_str(r.get(tid_col))
                if not tid or tid.lower() == "none":
                    continue

                ce_val  = to_num(r.get(ce_c))  if ce_c  else None
                cdf_val = to_num(r.get(cdf_c)) if cdf_c else None

                # Row-level fallback if CE & CDF got packed into one cell
                if ce_c and cdf_c is None and ce_val is None:
                    ce_try, cdf_try = split_ce_cdf_cell(r.get(ce_c))
                    if ce_val is None:  ce_val  = ce_try
                    if cdf_val is None: cdf_val = cdf_try

                rec = {
                    "Transporter ID": tid,
                    "Delivered": to_num(r.get(delivered_c)) if delivered_c else None,
                    "DCR":       to_percent(r.get(dcr_c))    if dcr_c else None,
                    "POD":       to_percent(r.get(pod_c))    if pod_c else None,
                    "CC":        to_percent(r.get(cc_c))     if cc_c else None,
                    "CE":        ce_val,
                    "LoR DPMO":  to_num(r.get(lor_c))        if lor_c else None,
                    "DNR DPMO":  to_num(r.get(dnr_c))        if dnr_c else None,
                    "CDF DPMO":  cdf_val,
                }
                rec.update(compute_scores(rec))
                rows.append(rec)

    return rows

# ===============================
# SUMMARY EXTRACTION
# ===============================
def extract_summary(pdf: pdfplumber.PDF) -> Dict[str, Any]:
    text = "\n".join(filter(None, (p.extract_text() for p in pdf.pages))) or ""
    res: Dict[str, Any] = {}

    def grab(rex: str, flags=re.IGNORECASE):
        m = re.search(rex, text, flags)
        return m

    # Week number & year  e.g. "Week 42 - 2025"
    if m := grab(r"\bWeek\s+(\d{1,2})\s*-\s*(\d{4})\b"):
        res["weekNumber"] = int(m.group(1))
        res["year"] = int(m.group(2))
        res["weekText"] = f"Week {m.group(1)} - {m.group(2)}"

    # Overall score  e.g. "Overall Score: 84.98 | Fantastic"
    if m := grab(r"Overall\s+Score:\s*([\d.,]+)"):
        # to_num converts "84,98" or "84.98" -> float
        res["overallScore"] = to_num(m.group(1))

    # Rank at station + WoW delta  e.g. "Rank at DBY5: 1 ( 0 WoW)"
    if m := grab(r"Rank\s+at\s+([A-Z0-9\-]+)\s*:\s*(\d+)\s*\(\s*([+-]?\s*\d+)\s*WoW", re.IGNORECASE):
        res["stationCode"] = clean_str(m.group(1))
        res["rankAtStation"] = int(m.group(2))
        # may include a leading plus/minus with space: "+ 3" / "0" / "- 2"
        res["rankDeltaWoW"] = int(to_num(m.group(3)) or 0)

    # Reliability (two flavors)
    # Next Day Capacity Reliability 102.08%
    if m := grab(r"Next\s+Day\s+Capacity\s+Reliability\s+([\d.,]+)\s*%"):
        res["reliabilityNextDay"] = to_num(m.group(1))

    # Same Day / Sub-Same Day Capacity Reliability 107%
    if m := grab(r"(?:Same\s+Day|Sub-?Same\s+Day)[^%]*?Capacity\s+Reliability\s+([\d.,]+)\s*%"):
        res["reliabilitySameDay"] = to_num(m.group(1))

    # For backward compatibility, also populate 'reliabilityScore' with Next-Day if available
    if res.get("reliabilityNextDay") is not None:
        res["reliabilityScore"] = res["reliabilityNextDay"]

    # Rank “in station X of Y” pattern (if such a variant exists in other PDFs)
    if m := grab(r"(?:Rank\s+(?:in\s+Station|at\s+[A-Z0-9\-]+))\D*(\d+)\D*(?:of|/|von)\D*(\d+)", re.IGNORECASE):
        res["rankAtStation"] = int(m.group(1))
        res["stationCount"] = int(m.group(2))

    return res
# ===============================
# RANKING
# ===============================
def add_ranking_and_status(drivers: List[Dict[str, Any]]) -> None:
    """
    Dense rank by FinalScore desc. Adds 'rank' (1..N) and 'statusBucket'.
    """
    # Sort copy to compute rank
    sortable = [d for d in drivers if isinstance(d.get("FinalScore"), (int, float))]
    sortable.sort(key=lambda x: x["FinalScore"], reverse=True)

    # Dense rank
    rank = 0
    last_score = None
    score_to_rank: Dict[float, int] = {}
    for d in sortable:
        fs = d["FinalScore"]
        if fs != last_score:
            rank += 1
            last_score = fs
        score_to_rank[fs] = rank

    # Apply rank & bucket
    for d in drivers:
        fs = d.get("FinalScore")
        d["rank"] = score_to_rank.get(fs) if isinstance(fs, (int, float)) else None
        d["statusBucket"] = status_bucket(fs)

# ===============================
# ROUTES
# ===============================
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/parse", response_model=ParserResponse)
async def parse_pdf(file: UploadFile = File(...)):
    content = await file.read()
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        drivers = extract_driver_rows(pdf)
        add_ranking_and_status(drivers)  # rank + bucket
        summary = extract_summary(pdf)
    return {
        "count": len(drivers),
        "drivers": drivers,
        "summary": summary or None
    }
