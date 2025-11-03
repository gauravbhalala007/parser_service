from fastapi import FastAPI, File, UploadFile
import pdfplumber
import pandas as pd
from typing import Dict, Any

app = FastAPI(title="Amazon DSP KPI Parser")

# ---------------------------
# Helper functions
# ---------------------------
def to_percent(x):
    if x is None:
        return None
    s = str(x).strip().replace(",", ".")
    try:
        if s.endswith("%"):
            return float(s[:-1])
        v = float(s)
        return v * 100 if v < 1 else v
    except:
        return None

def to_number(x):
    try:
        return float(str(x).strip().replace(",", "."))
    except:
        return None

# ---------------------------
# KPI scoring rules
# ---------------------------
def compute_scores(row: Dict[str, Any]) -> Dict[str, Any]:
    pod = to_percent(row.get("POD"))
    cc = to_percent(row.get("CC"))
    dcr = to_percent(row.get("DCR"))
    ce = to_number(row.get("CE"))
    lor = to_number(row.get("LoR DPMO"))
    dnr = to_number(row.get("DNR DPMO"))
    cdf = to_number(row.get("CDF DPMO"))

    ce_s = None if ce is None else max(50.0, 100.0 - ce * 50.0)
    lor_s = None if lor is None else max(70.0, 100.0 - (lor / 1200.0) * 30.0)
    dnr_s = None if dnr is None else max(70.0, 100.0 - (dnr / 1200.0) * 30.0)
    cdf_s = None
    if cdf is not None:
        cdf_s = max(0.0, min(100.0, 134.33 - 0.01333 * cdf))

    components = [pod, cc, dcr, ce_s, lor_s, dnr_s, cdf_s]
    valid = [v for v in components if v is not None]
    final = sum(valid) / len(valid) if valid else None

    return {
        "POD_Score": pod,
        "CC_Score": cc,
        "DCR_Score": dcr,
        "CE_Score": ce_s,
        "LoR_Score": lor_s,
        "DNR_Score": dnr_s,
        "CDF_Score": cdf_s,
        "FinalScore": final
    }

# ---------------------------
# Endpoint
# ---------------------------
@app.post("/parse")
async def parse_pdf(file: UploadFile = File(...)):
    rows = []
    with pdfplumber.open(file.file) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if not table:
                continue
            # Skip header text and non-data rows
            if "Transporter ID" in str(table[0]):
                header = table[0]
                df = pd.DataFrame(table[1:], columns=header)
                if "Transporter ID" not in df.columns:
                    continue

                for _, r in df.iterrows():
                    t_id = str(r.get("Transporter ID")).strip()
                    if not t_id or t_id == "None":
                        continue

                    rec = {
                        "Transporter ID": t_id,
                        "Delivered": r.get("Delivered"),
                        "DCR": r.get("DCR"),
                        "DNR DPMO": r.get("DNR DPMO"),
                        "LoR DPMO": r.get("LoR DPMO"),
                        "POD": r.get("POD"),
                        "CC": r.get("CC"),
                        "CE": r.get("CE"),
                        "CDF DPMO": r.get("CDF DPMO"),
                    }
                    rec.update(compute_scores(rec))
                    rows.append(rec)

    return {"drivers": rows, "count": len(rows)}
