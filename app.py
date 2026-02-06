# parser_service/app.py
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
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    statusBucket: Optional[str] = None

    # POD Quality (driver-level counts)
    POD_Q_Opportunities: Optional[float] = None
    POD_Q_Success: Optional[float] = None
    POD_Q_Bypass: Optional[float] = None
    POD_Q_Rejects: Optional[float] = None
    POD_Q_BlurryPhoto: Optional[float] = None
    POD_Q_PhotoTooDark: Optional[float] = None
    POD_Q_NoPackageDetected: Optional[float] = None
    POD_Q_PackageInCar: Optional[float] = None
    POD_Q_PackageTooClose: Optional[float] = None

class ParserSummary(BaseModel):
    overallScore: Optional[float] = None
    overallStatus: Optional[str] = None
    reliabilityScore: Optional[float] = None
    rankAtStation: Optional[int] = None
    stationCount: Optional[int] = None
    rankDeltaWoW: Optional[int] = None
    weekText: Optional[str] = None
    weekNumber: Optional[int] = None
    year: Optional[int] = None
    stationCode: Optional[str] = None

    reliabilityNextDay: Optional[float] = None
    reliabilitySameDay: Optional[float] = None

    podQualitySummary: Optional[Dict[str, Any]] = None
    podQualityRejects: Optional[Dict[str, Any]] = None

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
    s = (
        s.replace("\u00A0", " ")
         .replace("\u2009", " ")
         .replace("\u202F", " ")
         .replace("\u00AD", "")
         .replace("\n", " ")
         .replace("\r", " ")
    )
    return re.sub(r"\s+", " ", s).strip()

def keyize(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())

def to_num(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = clean_str(x)
    if not s or s == "-":
        return None
    s = s.replace("%", "").replace(" ", "")
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
    return "POOR"

# ===============================
# KPI FORMULAS (Albert’s rules)
# ===============================
def compute_scores(
    row: Dict[str, Any],
    week_number: Optional[int] = None,
    year: Optional[int] = None,   # <-- ADDED (only to decide FinalScore regime)
) -> Dict[str, Optional[float]]:
    # --- parse raw values (unchanged) ---
    pod = to_percent(row.get("POD"))
    cc  = to_percent(row.get("CC"))
    dcr = to_percent(row.get("DCR"))
    ce  = to_num(row.get("CE"))
    lor = to_num(row.get("LoR DPMO"))
    dnr = to_num(row.get("DNR DPMO"))
    cdf = to_num(row.get("CDF DPMO"))
    delivered = to_num(row.get("Delivered"))

    ce_val  = 0.0 if ce  is None else ce
    lor_val = 0.0 if lor is None else lor
    dnr_val = 0.0 if dnr is None else dnr
    cdf_val = 0.0 if cdf is None else cdf

    # --- per-metric scores (unchanged) ---
    POD_Score = None if pod is None else clamp(pod, 0, 100)
    CC_Score  = None if cc  is None else clamp(cc,  0, 100)
    DCR_Score = None if dcr is None else clamp(dcr, 0, 100)
    CE_Score  = None if ce  is None else clamp(100 - 50.0 * ce, 50, 100)

    CE_Score  = clamp(100.0 - ce_val * 50.0, 50.0, 100.0)
    LoR_Score = clamp(100.0 - (lor_val / 1200.0) * 30.0, 0.0, 100.0)
    DNR_Score = clamp(100.0 - (dnr_val / 1200.0) * 30.0, 0.0, 100.0)

    # UPDATED: CDF score mode is decided strictly by header presence (not by value).
    cdf_mode = row.get("_cdf_mode")  # "PCT" | "DPMO" | None
    if cdf_mode == "PCT":
        cdf_pct = to_percent(row.get("CDF"))
        CDF_Score = clamp(cdf_pct, 0.0, 100.0)
    elif cdf_mode == "DPMO":
        cdf_dpmo = to_num(row.get("CDF DPMO"))
        cdf_dpmo_val = 0.0 if cdf_dpmo is None else cdf_dpmo
        CDF_Score = clamp(100.33 - 0.01333 * float(cdf_dpmo_val), 0.0, 100.0)
    else:
        CDF_Score = None

    CE_Score  = round(CE_Score, 2)
    LoR_Score = round(LoR_Score, 2)
    DNR_Score = round(DNR_Score, 2)
    CDF_Score = None if CDF_Score is None else round(CDF_Score, 2)

    def z(x: Optional[float]) -> float:
        return 0.0 if x is None else float(x)

    FinalScore: Optional[float] = None

    # ==========================
    # FINAL SCORE REGIME SWITCH (ONLY CHANGE):
    # Old formula valid through (Year=2025, Week=40)
    # ==========================
    def use_old_formula(y: Optional[int], w: Optional[int]) -> bool:
        if w is None:
            return False
        if y is None:
            # If year is missing, keep legacy behavior (week-only)
            return w <= 40
        if y < 2025:
            return True
        if y == 2025 and w <= 40:
            return True
        return False

    if use_old_formula(year, week_number):
        # ==========================
        #  OLD FORMULA
        # ==========================
        dcr_v = DCR_Score
        pod_v = POD_Score
        cc_v  = CC_Score
        cdf_v = CDF_Score

        vals = [v for v in [dcr_v, pod_v, cc_v, cdf_v] if v is not None]
        if vals:
            avg_base = sum(vals) / len(vals)
            ce_raw   = z(ce)
            delivered_v = z(delivered)
            dnr_v = z(dnr)

            value_term = 0.0
            if delivered_v > 0:
                value_term = (dnr_v / delivered_v - 10.0 * ce_raw) * 11.0 + 10.0 * ce_raw

            FinalScore = avg_base * 1.0 - value_term - 10.0 * ce_raw
        else:
            FinalScore = None

    else:
        # ==========================
        #  NEW FORMULAS
        # ==========================
        denom = 14.1
        w_dnr = 4.0
        w_lor = 4.0

        # Week-41/42 transition applies only for 2025 (otherwise week numbers repeat each year)
        if year == 2025 and week_number == 41:
            denom = 16.1
            w_dnr = 5.0
            w_lor = 5.0
        elif year == 2025 and week_number == 42:
            denom = 16.5
            w_dnr = 5.0
            w_lor = 5.0

        numerator = (
            z(DCR_Score) +
            z(POD_Score) +
            3.0 * z(CC_Score) +
            w_dnr * z(DNR_Score) +
            w_lor * z(LoR_Score) +
            (z(CE_Score) / 50.0) +
            z(CDF_Score)
        )

        FinalScore = numerator / denom if denom != 0 else None

    return {
        "POD_Score": POD_Score, "CC_Score": CC_Score, "DCR_Score": DCR_Score,
        "CE_Score": CE_Score, "LoR_Score": LoR_Score, "DNR_Score": DNR_Score,
        "CDF_Score": CDF_Score, "FinalScore": round(FinalScore, 2)
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
    "dsc dpmo": "DNR DPMO",
    "DSC DPMO": "DNR DPMO",
    "dsc (dpmo)": "DNR DPMO",

    "cdf dpmo": "CDF DPMO",
    "cdf (dpmo)": "CDF DPMO",
    "cdfdpmo": "CDF DPMO",
    "cdfdpm0": "CDF DPMO",
    "cdf": "CDF",

    "delivered": "Delivered",
    "zugestellte pakete": "Delivered",
}

def norm_col(c: Any) -> str:
    raw = clean_str(c)
    k = keyize(raw)
    return HEADER_MAP.get(k, raw)

# ===============================
# PDF TABLE EXTRACTION
# ===============================
def get_col(df: pd.DataFrame, canonical: str) -> Optional[str]:
    want = keyize(canonical)
    for col in df.columns:
        if keyize(col) == want:
            return col
    return None

def extract_driver_rows(
    pdf: pdfplumber.PDF,
    week_number: Optional[int] = None,
    year: Optional[int] = None,   # <-- ADDED (only to pass into compute_scores)
) -> List[Dict[str, Any]]:
    def is_header_like(row: List[Any]) -> bool:
        if not row:
            return False
        texty = sum(1 for c in row if re.search(r"[A-Za-z]", clean_str(c)))
        return texty >= 3

    def looks_like_transporter_id(x: Any) -> bool:
        s = clean_str(x)
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
            if ("ce" in kcur and ("cdfdpmo" in kcur or "cdf" in kcur)) and (knxt == "" or nxt == ""):
                h[i] = "CE"
                h[i + 1] = "CDF DPMO"
                i += 2
                continue
            i += 1
        return h

    def build_df_with_header(header: List[str], raw_rows: List[List[Any]]) -> pd.DataFrame:
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

        rename_map = {}
        for col in df.columns:
            kcol = keyize(col)
            for ksrc, vdst in HEADER_MAP.items():
                if keyize(ksrc) == kcol:
                    rename_map[col] = vdst
                    break
        if rename_map:
            df = df.rename(columns=rename_map)

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

            if "Transporter ID" in raw0 or is_header_like(raw0):
                if raw1 and is_header_like(raw1):
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
                last_header = header[:]
            elif last_header is not None and table and looks_like_transporter_id(table[0][0]):
                header = last_header[:]
                data_start = 0

            if header is None:
                continue

            df = build_df_with_header(header, table[data_start:])

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

            cdf_pct_c   = get_col(df, "CDF")
            cdf_dpmo_c  = get_col(df, "CDF DPMO")
            cdf_mode = None
            if cdf_pct_c is not None:
                cdf_mode = "PCT"
            elif cdf_dpmo_c is not None:
                cdf_mode = "DPMO"

            for _, r in df.iterrows():
                tid = clean_str(r.get(tid_col))
                if not tid or tid.lower() == "none":
                    continue

                ce_val = to_num(r.get(ce_c)) if ce_c else None
                cdf_dpmo_val = to_num(r.get(cdf_dpmo_c)) if cdf_dpmo_c else None
                cdf_pct_val  = to_percent(r.get(cdf_pct_c)) if cdf_pct_c else None

                if ce_c and cdf_dpmo_c is None and ce_val is None:
                    ce_try, cdf_try = split_ce_cdf_cell(r.get(ce_c))
                    if ce_val is None:
                        ce_val = ce_try
                    if cdf_dpmo_val is None:
                        cdf_dpmo_val = cdf_try

                rec = {
                    "Transporter ID": tid,
                    "Delivered": to_num(r.get(delivered_c)) if delivered_c else None,
                    "DCR":       to_percent(r.get(dcr_c))    if dcr_c else None,
                    "POD":       to_percent(r.get(pod_c))    if pod_c else None,
                    "CC":        to_percent(r.get(cc_c))     if cc_c else None,
                    "CE":        ce_val,
                    "LoR DPMO":  to_num(r.get(lor_c))        if lor_c else None,
                    "DNR DPMO":  to_num(r.get(dnr_c))        if dnr_c else None,
                    "CDF DPMO":  cdf_dpmo_val,
                    "CDF":       cdf_pct_val,
                    "_cdf_mode": cdf_mode,
                }

                # ONLY CHANGE: pass year into compute_scores so FinalScore regime uses (year, week)
                rec.update(compute_scores(rec, week_number=week_number, year=year))
                rows.append(rec)

    return rows

# ===============================
# SUMMARY EXTRACTION
# ===============================
def extract_summary(pdf: pdfplumber.PDF) -> Dict[str, Any]:
    text = "\n".join(filter(None, (p.extract_text() for p in pdf.pages))) or ""
    res: Dict[str, Any] = {}

    def grab(rex: str, flags=re.IGNORECASE):
        return re.search(rex, text, flags)

    if m := grab(r"\bWeek\s+(\d{1,2})\s*-\s*(\d{4})\b"):
        res["weekNumber"] = int(m.group(1))
        res["year"] = int(m.group(2))
        res["weekText"] = f"Week {m.group(1)} - {m.group(2)}"

    if m := grab(r"Overall\s+Score:\s*([\d.,]+)(?:\s*[|/]\s*([A-Za-z+\- ]+))?"):
        res["overallScore"] = to_num(m.group(1))
        if m.group(2):
            res["overallStatus"] = clean_str(m.group(2)).upper().replace(" ", "_")

    if res.get("overallStatus") is None:
        if m := grab(r"\bOverall\s+(?:Standing|Performance)\s+(?:as|is|:)?\s*([A-Za-z+\- ]+)\b"):
            res["overallStatus"] = clean_str(m.group(1)).upper().replace(" ", "_")

    if m := grab(r"Rank\s+at\s+([A-Z0-9\-]+)\s*:\s*(\d+)\s*\(\s*([+-]?\s*\d+)\s*WoW", re.IGNORECASE):
        res["stationCode"] = clean_str(m.group(1))
        res["rankAtStation"] = int(m.group(2))
        res["rankDeltaWoW"] = int(to_num(m.group(3)) or 0)

    if m := grab(r"Next\s+Day\s+Capacity\s+Reliability\s+([\d.,]+)\s*%"):
        res["reliabilityNextDay"] = to_num(m.group(1))

    if m := grab(r"(?:Same\s+Day|Sub-?Same\s+Day)[^%]*?Capacity\s+Reliability\s+([\d.,]+)\s*%"):
        res["reliabilitySameDay"] = to_num(m.group(1))

    if res.get("reliabilityNextDay") is not None:
        res["reliabilityScore"] = res["reliabilityNextDay"]

    if m := grab(r"(?:Rank\s+(?:in\s+Station|at\s+[A-Z0-9\-]+))\D*(\d+)\D*(?:of|/|von)\D*(\d+)", re.IGNORECASE):
        res["rankAtStation"] = int(m.group(1))
        res["stationCount"] = int(m.group(2))

    return res

# ===============================
# POD QUALITY EXTRACTION
# ===============================
def _is_pod_quality_pdf(pdf: pdfplumber.PDF, filename: str) -> bool:
    name = (filename or "").lower()
    if "pod-quality" in name or "pod_quality" in name:
        return True
    first_text = clean_str(pdf.pages[0].extract_text() or "").lower()
    return "photo on delivery quality report" in first_text

def _extract_pod_quality_summary(pdf: pdfplumber.PDF, filename: str) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    text = clean_str(pdf.pages[0].extract_text() or "")

    if m := re.search(r"\b([A-Z0-9]{3,6})\s*-\s*Week\s*(\d{1,2})\b", text):
        res["stationCode"] = m.group(1)
        res["weekNumber"] = int(m.group(2))
        res["weekText"] = f"Week {m.group(2)}"

    if "year" not in res and filename:
        if m := re.search(r"(20\d{2})", filename):
            res["year"] = int(m.group(1))

    pod_summary: Dict[str, Any] = {}
    pod_rejects: Dict[str, Any] = {}

    tables = pdf.pages[0].extract_tables() or []
    for table in tables:
        if not table or len(table) < 2:
            continue
        header = [clean_str(c).lower() for c in table[0]]
        if "category" in header and any("total opportunities" in h for h in header):
            for row in table[1:]:
                cat = clean_str(row[0]).lower()
                cnt = to_num(row[1])
                pct = to_percent(row[2])
                key = keyize(cat)
                if key == "success":
                    pod_summary["successCount"] = cnt
                    pod_summary["successPct"] = pct
                elif key == "bypass":
                    pod_summary["bypassCount"] = cnt
                    pod_summary["bypassPct"] = pct
                elif key == "rejects":
                    pod_summary["rejectsCount"] = cnt
                    pod_summary["rejectsPct"] = pct
                elif key == "opportunities":
                    pod_summary["opportunitiesCount"] = cnt
                    pod_summary["opportunitiesPct"] = pct
        elif "category" in header and any("total rejects" in h for h in header):
            for row in table[1:]:
                cat = clean_str(row[0]).lower()
                cnt = to_num(row[1])
                pct = to_percent(row[2])
                key = keyize(cat)
                if key == "blurryphoto":
                    pod_rejects["blurryPhotoCount"] = cnt
                    pod_rejects["blurryPhotoPct"] = pct
                elif key == "phototoodark":
                    pod_rejects["photoTooDarkCount"] = cnt
                    pod_rejects["photoTooDarkPct"] = pct
                elif key == "nopackagedetected":
                    pod_rejects["noPackageDetectedCount"] = cnt
                    pod_rejects["noPackageDetectedPct"] = pct
                elif key == "packageincar":
                    pod_rejects["packageInCarCount"] = cnt
                    pod_rejects["packageInCarPct"] = pct
                elif key == "packagetooclose":
                    pod_rejects["packageTooCloseCount"] = cnt
                    pod_rejects["packageTooClosePct"] = pct
                elif key in ("grandtotal", "total"):
                    pod_rejects["totalRejectsCount"] = cnt
                    pod_rejects["totalRejectsPct"] = pct

    if pod_summary:
        res["podQualitySummary"] = pod_summary
    if pod_rejects:
        res["podQualityRejects"] = pod_rejects

    return res

def _extract_pod_quality_drivers(pdf: pdfplumber.PDF) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def looks_like_transporter_id(x: Any) -> bool:
        s = clean_str(x)
        return bool(re.match(r"^[A-Z0-9]{8,20}$", s))

    for page in pdf.pages:
        tables = page.extract_tables() or []
        for table in tables:
            if not table:
                continue
            for row in table:
                tid = clean_str(row[0])
                if not looks_like_transporter_id(tid):
                    continue
                r = list(row) + [""] * 10
                rows.append(
                    {
                        "Transporter ID": tid,
                        "POD_Q_Opportunities": to_num(r[1]),
                        "POD_Q_Success": to_num(r[2]),
                        "POD_Q_Bypass": to_num(r[3]),
                        "POD_Q_Rejects": to_num(r[4]),
                        "POD_Q_BlurryPhoto": to_num(r[5]),
                        "POD_Q_NoPackageDetected": to_num(r[6]),
                        "POD_Q_PackageInCar": to_num(r[7]),
                        "POD_Q_PackageTooClose": to_num(r[8]),
                        "POD_Q_PhotoTooDark": to_num(r[9]),
                    }
                )

    return rows

# ===============================
# RANKING
# ===============================
def add_ranking_and_status(drivers: List[Dict[str, Any]]) -> None:
    sortable = [d for d in drivers if isinstance(d.get("FinalScore"), (int, float))]
    sortable.sort(key=lambda x: x["FinalScore"], reverse=True)

    rank = 0
    last_score = None
    score_to_rank: Dict[float, int] = {}
    for d in sortable:
        fs = d["FinalScore"]
        if fs != last_score:
            rank += 1
            last_score = fs
        score_to_rank[fs] = rank

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
        if _is_pod_quality_pdf(pdf, file.filename or ""):
            summary = _extract_pod_quality_summary(pdf, file.filename or "")
            drivers = _extract_pod_quality_drivers(pdf)
        else:
            summary = extract_summary(pdf)
            week_number = summary.get("weekNumber") if summary else None
            year = summary.get("year") if summary else None  # <-- ADDED

            # ONLY CHANGE: pass year through so FinalScore uses (year, week)
            drivers = extract_driver_rows(pdf, week_number=week_number, year=year)

            add_ranking_and_status(drivers)

    for d in drivers:
        if isinstance(d, dict) and "_cdf_mode" in d:
            d.pop("_cdf_mode", None)

    return {
        "count": len(drivers),
        "drivers": drivers,
        "summary": summary or None
    }
