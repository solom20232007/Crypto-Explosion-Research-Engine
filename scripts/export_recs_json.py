"""Export the latest published daily recommendations as a small public JSON (research output, not a trading signal)."""
import json, os, requests

URL = os.environ["SUPABASE_URL"].rstrip("/"); KEY = os.environ["SUPABASE_SECRET_KEY"]
H = {"apikey": KEY, "Authorization": f"Bearer {KEY}"}

def get(table, params):
    r = requests.get(f"{URL}/rest/v1/{table}", headers=H, params=params, timeout=60); r.raise_for_status(); return r.json()

def recent_surge(symbol):
    """Hour in the last 24 closed hours with quote volume >= 10x trailing-24h mean AND hour return >= +10% (backtested: buying after it has median -8%/24h)."""
    c = get("candles", {"select": "open_time,open,close,quote_volume", "symbol": f"eq.{symbol}", "timeframe": "eq.1h", "order": "open_time.desc", "limit": "60"})[::-1]
    if len(c) < 30: return None
    q = [float(x["quote_volume"]) for x in c]; best = None
    for i in range(len(c) - 24, len(c)):
        base = sum(q[i - 24:i]) / 24
        if base <= 0: continue
        ret = float(c[i]["close"]) / float(c[i]["open"]) - 1; vr = q[i] / base
        if vr >= 10 and ret >= 0.10 and (best is None or vr > best["vol_ratio"]):
            best = {"hours_ago": len(c) - 1 - i, "hour_ret": round(ret, 3), "vol_ratio": round(vr, 1)}
    return best

def onchain(symbols):
    """Ethereum-only, informational (NOT validated as a predictor): recent 6h transfer count vs the prior 24h hourly average."""
    out = {}
    cons = {x["symbol"]: x for x in get("token_contracts_v1", {"select": "symbol,chain,confidence", "chain": "eq.ethereum", "limit": "1000"})}
    for s in symbols:
        o = {"eth_contract": s in cons, "eth_contract_confidence": cons[s]["confidence"] if s in cons else None}
        if s in cons:
            h = get("onchain_token_hourly_v1", {"select": "hour_utc,transfer_count,whale_volume_usd,complete", "symbol": f"eq.{s}", "order": "hour_utc.desc", "limit": "32"})
            h = [x for x in h if x.get("complete")]
            if len(h) >= 12:
                recent, prior = h[:6], h[6:30]
                avg = sum(x["transfer_count"] for x in prior) / max(len(prior), 1)
                o["onchain_ratio_6h"] = round((sum(x["transfer_count"] for x in recent) / 6) / avg, 2) if avg > 0 else None
                o["whale_usd_6h"] = round(sum(float(x["whale_volume_usd"] or 0) for x in recent))
        out[s] = o
    return out

def main():
    last = get("daily_recommendations_v1", {"select": "bar_time", "order": "bar_time.desc", "limit": 1})[0]["bar_time"]
    rows = get("daily_recommendations_v1", {"select": "symbol,rank,score_10_24,score_25_48,score_40_72,market_wide,bar_time",
                                            "bar_time": f"eq.{last}", "already_moved": "eq.false", "order": "rank.asc", "limit": 30})
    halal = {x["base"] for x in get("v6_shariah_bases_v1", {"select": "base", "limit": 5000})}
    out = [{"symbol": r["symbol"], "rank": r["rank"], "p_up10_24h": r["score_10_24"], "p_up25_48h": r["score_25_48"], "p_up40_72h": r["score_40_72"],
            "market_wide": r["market_wide"], "decision_bar_utc": r["bar_time"],
            "shariah_listed_v6": r["symbol"][:-4] in halal if r["symbol"].endswith("USDT") else False} for r in rows]
    oc = onchain([r["symbol"] for r in out])
    for r in out:
        try: r["recent_surge"] = recent_surge(r["symbol"])
        except Exception as e: r["recent_surge"] = None; print("surge check failed", r["symbol"], e)
        r.update(oc.get(r["symbol"], {}))
    json.dump(out, open("latest_recommendations.json", "w"), ensure_ascii=False, indent=1)
    print(len(out), "rows for", last)
main()
