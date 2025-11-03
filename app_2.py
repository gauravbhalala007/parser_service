# app.py
# Robust FastAPI Parser for Amazon DSP KPI PDFs
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import io, re
import pdfplumber
import pandas as pd

app = FastAPI(title="Amazon DSP KPI Parser")

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
    # Computed
    POD_Score: Optional[float] = None
    CC_Score: Optional[float] = None
    DCR_Score: Optional[float] = None
    CE_Score: Optional[float] = None
    LoR_Score: Optional[float] = None
    DNR_Score: Optional[float] = None
    CDF_Score: Optional[float] = None
    FinalScore: Optional[float] = None

class ParserSummary(BaseModel):
    overallScore: Optional[float] = None
    reliabilityScore: Optional[float] = None
    rankAtStation: Optional[int] = None
    stationCount: Optional[int] = None
    rankDeltaWoW: Optional[int] = None
    weekText: Optional[str] = None

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
    return str(x).replace(NBSP, " ").strip()

def to_num(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = clean_str(x)
    if not s or s == "-":
        return None
    s = s.replace("%", "").replace(" ", "")
    # German/US hybrid decimal handling
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

# ===============================
# KPI FORMULAS (Albert’s rules)
# ===============================
def compute_scores(row: Dict[str, Any]) -> Dict[str, Optional[float]]:
    pod = to_percent(row.get("POD"))
    cc  = to_percent(row.get("CC"))
    dcr = to_percent(row.get("DCR"))
    ce  = to_num(row.get("CE"))
    lor = to_num(row.get("LoR DPMO"))
    dnr = to_num(row.get("DNR DPMO"))
    cdf = to_num(row.get("CDF DPMO"))

    POD_Score = None if pod is None else clamp(pod, 0, 100)
    CC_Score  = None if cc  is None else clamp(cc,  0, 100)
    DCR_Score = None if dcr is None else clamp(dcr, 0, 100)
    CE_Score  = None if ce  is None else clamp(100 - 50.0 * ce, 50, 100)
    LoR_Score = None if lor is None else clamp(max(70.0, 100.0 - (lor / 1200.0) * 30.0), 0, 100)
    DNR_Score = None if dnr is None else clamp(max(70.0, 100.0 - (dnr / 1200.0) * 30.0), 0, 100)
    CDF_Score = None if cdf is None else clamp(134.33 - 0.01333 * cdf, 0, 100)

    valid = [v for v in [POD_Score, CC_Score, DCR_Score, CE_Score, LoR_Score, DNR_Score, CDF_Score] if v is not None]
    FinalScore = sum(valid) / len(valid) if valid else None

    return {
        "POD_Score": POD_Score, "CC_Score": CC_Score, "DCR_Score": DCR_Score,
        "CE_Score": CE_Score, "LoR_Score": LoR_Score, "DNR_Score": DNR_Score,
        "CDF_Score": CDF_Score, "FinalScore": FinalScore,
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
    "delivered": "Delivered",
    "zugestellte pakete": "Delivered",
}

def norm_col(c: Any) -> str:
    s = clean_str(c).lower()
    s = re.sub(r"\s+", " ", s)
    return HEADER_MAP.get(s, clean_str(c))

# ===============================
# PDF TABLE EXTRACTION
# ===============================
def extract_driver_rows(pdf: pdfplumber.PDF) -> List[Dict[str, Any]]:
    rows = []
    for page in pdf.pages:
        tables = page.extract_tables() or []
        for table in tables:
            if not table or len(table) < 2:
                continue

            header = [norm_col(h) for h in table[0]]
            if "Transporter ID" not in header:
                continue

            df = pd.DataFrame(table[1:], columns=header)
            if "Transporter ID" not in df.columns:
                continue

            for _, r in df.iterrows():
                tid = clean_str(r.get("Transporter ID"))
                if not tid or tid.lower() == "none":
                    continue

                rec = {
                    "Transporter ID": tid,
                    "Delivered": to_num(r.get("Delivered")),
                    "DCR": to_percent(r.get("DCR")),
                    "POD": to_percent(r.get("POD")),
                    "CC": to_percent(r.get("CC")),
                    "CE": to_num(r.get("CE")),
                    "LoR DPMO": to_num(r.get("LoR DPMO")),
                    "DNR DPMO": to_num(r.get("DNR DPMO")),
                    "CDF DPMO": to_num(r.get("CDF DPMO")),
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

    def pick(pats: List[str]) -> Optional[str]:
        for pat in pats:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                return m.group(1)
        return None

    if m := pick([r"(?:Overall\s+Score|Gesamtscore)\D*([\d.,]+)\s*%"]):
        res["overallScore"] = to_num(m)
    if m := pick([r"(?:Reliability\s+Score|Reliability)\D*([\d.,]+)\s*%"]):
        res["reliabilityScore"] = to_num(m)
    if m := re.search(r"Rank\s+(?:in\s+Station|at\s+DBY5)\D*(\d+)\D*(?:of|/|von)\D*(\d+)", text, re.IGNORECASE):
        res["rankAtStation"], res["stationCount"] = int(m.group(1)), int(m.group(2))
    if m := re.search(r"([+-]\s*\d+)\s*from\s*WoW", text, re.IGNORECASE):
        res["rankDeltaWoW"] = int(to_num(m.group(1)) or 0)
    if m := re.search(r"\bWoche\s*\d{1,2}\b.*", text, re.IGNORECASE):
        res["weekText"] = clean_str(m.group(0))

    return res

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
        summary = extract_summary(pdf)
    return {"count": len(drivers), "drivers": drivers, "summary": summary or None}
