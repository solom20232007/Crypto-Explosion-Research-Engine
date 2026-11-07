"""Import the independent V6 scanner's public CSV outputs into Supabase (read-only on V6; no secrets needed there).

V6 repo: github.com/solom20232007/v6. Imports final signals and the watchlist so they can be compared with our recommendations.
"""
import io, json, os
import pandas as pd, requests

URL = os.environ["SUPABASE_URL"].rstrip("/"); KEY = os.environ["SUPABASE_SECRET_KEY"]
SB = {"apikey": KEY, "Authorization": f"Bearer {KEY}", "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates"}
RAW = "https://raw.githubusercontent.com/solom20232007/v6/{branch}/{path}"
BRANCH = os.environ.get("V6_BRANCH", "main")

def load(path):
    r = requests.get(RAW.format(branch=BRANCH, path=path), timeout=60); r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

def upsert(table, conflict, df):
    df = df.astype(object).where(df.notna(), None)
    rows = json.loads(df.to_json(orient="records"))
    for i in range(0, len(rows), 500):
        requests.post(f"{URL}/rest/v1/{table}?on_conflict={conflict}", headers=SB, data=json.dumps(rows[i:i + 500]), timeout=120).raise_for_status()
    return len(rows)

def main():
    sig = load("early_pump_signal_results_v6.csv")
    cols = ["final_time_utc", "symbol", "setup", "score", "lane", "entry_price", "sl_pct", "tp1_pct", "tp2_pct", "liquidity", "quote_volume",
            "gain_5m_at_alert", "gain_15m_at_alert", "gain_1h_at_alert", "gain_24h_at_alert", "mfe_pct", "mae_pct", "status", "outcome", "exit_pct", "outcome_min"]
    sig = sig[[c for c in cols if c in sig.columns]].dropna(subset=["final_time_utc", "symbol"])
    n1 = upsert("v6_signals_v1", "final_time_utc,symbol", sig)
    w = load("early_watch_log_v6.csv")
    wcols = ["logged_time_utc", "symbol", "setup", "score", "price", "gain_15m", "gain_1h", "gain_24h", "volume_1h", "volume_15m", "acceleration", "liquidity", "quote_volume", "is_wash"]
    w = w[[c for c in wcols if c in w.columns]].dropna(subset=["logged_time_utc", "symbol"]).drop_duplicates(["logged_time_utc", "symbol"])
    if "is_wash" in w: w["is_wash"] = w["is_wash"].astype(float).astype(bool)
    n2 = upsert("v6_watch_v1", "logged_time_utc,symbol", w)
    # Shariah/halal base list maintained by V6 (third-party dataset; not independently verified by this project)
    r = requests.get(RAW.format(branch=BRANCH, path="shariah_compliant_bases_v6.json"), timeout=60); r.raise_for_status()
    j = r.json()
    hal = pd.DataFrame({"base": sorted({str(x).upper() for x in j.get("symbols", [])})})
    hal["list_updated_at"] = j.get("updated_at_utc"); hal["source"] = j.get("source")
    n3 = upsert("v6_shariah_bases_v1", "base", hal) if len(hal) else 0
    print(json.dumps({"v6_signals": n1, "v6_watch": n2, "shariah_bases": n3, "branch": BRANCH}))
main()
