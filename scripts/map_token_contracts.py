"""Map Binance research symbols to on-chain token contracts using CoinGecko (public API, no key).

Rules: candidates share the ticker; the highest market-cap-rank candidate wins. Confidence:
high   = single candidate with an address on a supported chain, or a clear rank leader (rank < 0.5x runner-up's rank number);
medium = leader exists but runner-up rank is close; low = no market-cap rank or no candidate with a supported-chain address.
Everything is stored with reviewed=false: a human must confirm before whale data is trusted.
"""
import json, os, sys, time
from pathlib import Path
import requests

URL = os.environ["SUPABASE_URL"].rstrip("/"); KEY = os.environ["SUPABASE_SECRET_KEY"]
CG = "https://api.coingecko.com/api/v3"
CHAINS = {"ethereum": "ethereum", "binance-smart-chain": "bsc", "solana": "solana"}
S = requests.Session(); S.headers["User-Agent"] = "research-contract-mapper/1.0"

def cg(path, params=None, tries=8):
    for i in range(tries):
        r = S.get(CG + path, params=params, timeout=60)
        if r.status_code == 429:
            time.sleep(30 + 15 * i); continue
        r.raise_for_status(); return r.json()
    raise RuntimeError(f"CoinGecko kept rate-limiting {path}")

def main():
    symbols = [l.split("#")[0].strip().upper() for l in Path("config/research_symbols.txt").read_text().splitlines() if l.split("#")[0].strip()]
    bases = {s[:-4].lower(): s for s in symbols if s.endswith("USDT") and s not in {"BTCUSDT", "ETHUSDT", "SOLUSDT"}}
    coins = cg("/coins/list", {"include_platform": "true"}); print("coins", len(coins))
    cand = {}
    for c in coins:
        if c["symbol"].lower() in bases: cand.setdefault(c["symbol"].lower(), []).append(c)
    ranks = {}
    for page in range(1, 9):
        for c in cg("/coins/markets", {"vs_currency": "usd", "order": "market_cap_desc", "per_page": 250, "page": page}):
            ranks[c["id"]] = c.get("market_cap_rank")
        time.sleep(8)
    rows, report = [], []
    for base, sym in sorted(bases.items()):
        cs = cand.get(base, [])
        scored = sorted(cs, key=lambda c: ranks.get(c["id"]) or 10**9)
        usable = [c for c in scored if any(p in CHAINS and c.get("platforms", {}).get(p) for p in c.get("platforms", {}))]
        if not usable:
            report.append((sym, "no candidate with ETH/BSC/Solana address", len(cs))); continue
        best = usable[0]; rk = ranks.get(best["id"])
        second = ranks.get(usable[1]["id"]) if len(usable) > 1 else None
        if len(usable) == 1 and rk: conf = "high"
        elif rk and (second is None or rk * 2 <= second): conf = "high"
        elif rk: conf = "medium"
        else: conf = "low"
        time.sleep(12)
        detail = cg(f"/coins/{best['id']}", {"localization": "false", "tickers": "false", "market_data": "false", "community_data": "false", "developer_data": "false", "sparkline": "false"})
        dp = detail.get("detail_platforms", {})
        for plat, chain in CHAINS.items():
            addr = best.get("platforms", {}).get(plat)
            if not addr: continue
            rows.append({"symbol": sym, "chain": chain, "contract_address": addr.lower() if chain != "solana" else addr,
                         "decimals": (dp.get(plat) or {}).get("decimal_place"), "coingecko_id": best["id"], "coingecko_name": detail.get("name"),
                         "market_cap_rank": rk, "candidates_for_symbol": len(usable), "confidence": conf, "reviewed": False})
        print(sym, best["id"], rk, conf, len(usable))
    H = {"apikey": KEY, "Authorization": f"Bearer {KEY}", "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates"}
    for i in range(0, len(rows), 200):
        r = requests.post(f"{URL}/rest/v1/token_contracts_v1?on_conflict=symbol,chain", headers=H, data=json.dumps(rows[i:i+200]), timeout=120); r.raise_for_status()
    print(f"mapped {len(rows)} rows for {len({r['symbol'] for r in rows})} symbols; unmapped {len(report)}")
    for x in report: print("UNMAPPED", x)
main()
