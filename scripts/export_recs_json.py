"""Export the latest published daily recommendations as a small public JSON (research output, not a trading signal)."""
import json, os, requests

URL = os.environ["SUPABASE_URL"].rstrip("/"); KEY = os.environ["SUPABASE_SECRET_KEY"]
H = {"apikey": KEY, "Authorization": f"Bearer {KEY}"}

def get(table, params):
    r = requests.get(f"{URL}/rest/v1/{table}", headers=H, params=params, timeout=60); r.raise_for_status(); return r.json()

def main():
    last = get("daily_recommendations_v1", {"select": "bar_time", "order": "bar_time.desc", "limit": 1})[0]["bar_time"]
    rows = get("daily_recommendations_v1", {"select": "symbol,rank,score_10_24,score_25_48,score_40_72,market_wide,bar_time",
                                            "bar_time": f"eq.{last}", "already_moved": "eq.false", "order": "rank.asc", "limit": 30})
    halal = {x["base"] for x in get("v6_shariah_bases_v1", {"select": "base", "limit": 5000})}
    out = [{"symbol": r["symbol"], "rank": r["rank"], "p_up10_24h": r["score_10_24"], "p_up25_48h": r["score_25_48"], "p_up40_72h": r["score_40_72"],
            "market_wide": r["market_wide"], "decision_bar_utc": r["bar_time"],
            "shariah_listed_v6": r["symbol"][:-4] in halal if r["symbol"].endswith("USDT") else False} for r in rows]
    json.dump(out, open("latest_recommendations.json", "w"), ensure_ascii=False, indent=1)
    print(len(out), "rows for", last)
main()
