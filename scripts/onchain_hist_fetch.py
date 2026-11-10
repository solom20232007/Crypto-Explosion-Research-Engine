"""(helpers copied from onchain_pilot) Compact on-chain pilot: per-token hourly activity + whale transfers (>= $25k) for mapped contracts.

Uses Routescan's erc20-transfers endpoint filtered by tokenAddress (Ethereum only). Only contracts with
confidence='high' in token_contracts_v1 are collected. Re-fetches the last 3 complete hours each run (idempotent upserts).
Research indicators only: transfer direction/intent is NOT inferred.
"""
import json, os, time
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse
import requests

URL = os.environ["SUPABASE_URL"].rstrip("/"); KEY = os.environ["SUPABASE_SECRET_KEY"]; RS = os.environ.get("ROUTESCAN_API_KEY", "")
SB = {"apikey": KEY, "Authorization": f"Bearer {KEY}"}
CHAIN_ID = {"ethereum": 1}  # Routescan has no BSC index (NO_BLOCKCHAINS_FOR_PARAMS); BSC needs another provider
WHALE_USD = 25_000
MAX_PAGES = int(os.environ.get("MAX_PAGES", "40")); HOURS = int(os.environ.get("HOURS", "3"))
S = requests.Session()

def get(url, params):
    for i in range(6):
        r = S.get(url, params=params, headers={"apikey": RS} if RS else {}, timeout=(10, 45))
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(min(30, 2 ** i)); continue
        if r.status_code >= 400: raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        return r.json()
    raise RuntimeError(f"Routescan kept failing: {url}")

def cursor(link):
    if link.get("nextToken"): return str(link["nextToken"])
    nxt = link.get("next")
    if not nxt: return None
    v = parse_qs(urlparse(str(nxt)).query).get("next")
    return v[-1] if v else None

def post(table, rows, conflict):
    for i in range(0, len(rows), 500):
        r = requests.post(f"{URL}/rest/v1/{table}?on_conflict={conflict}", headers={**SB, "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates"},
                          data=json.dumps(rows[i:i + 500]), timeout=120); r.raise_for_status()

def fetch_interval(base, token, start, end, depth=0):
    """Fetch all transfers for one token in [start, end). Splits the interval when the page cap is hit.
    Returns (items, complete)."""
    iso = lambda d: d.strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {"tokenAddress": token, "timestampFrom": iso(start), "timestampTo": iso(end), "sort": "desc", "limit": 100}
    items, cur = [], None
    for _ in range(MAX_PAGES):
        if cur: params["next"] = cur
        p = get(base, params); got = p.get("items") or []; items += got
        cur = cursor(p.get("link") or {})
        if not cur or not got: return items, True
        time.sleep(0.22 if RS else 0.55)
    if (end - start) <= timedelta(hours=1) or depth >= 6:
        return items, False  # dense hour: keep what we have and flag incomplete
    mid = start + (end - start) / 2
    mid = mid.replace(minute=0, second=0, microsecond=0) if (end - start) > timedelta(hours=1) else mid
    if mid <= start or mid >= end: return items, False
    a, ca = fetch_interval(base, token, mid, end, depth + 1); b, cb = fetch_interval(base, token, start, mid, depth + 1)
    return a + b, ca and cb


START_DATE = os.environ.get("START_DATE", "2023-11-01"); END_DATE = os.environ.get("END_DATE", "2026-10-28")
STEP = int(os.environ.get("STEP_DAYS", "4")); SHARD = int(os.environ.get("SHARD", "0")); NSH = int(os.environ.get("NSHARDS", "1"))
DEADLINE = time.time() + float(os.environ.get("MAX_SECONDS", "19800"))

def existing(symbol):
    done, off = set(), 0
    while True:
        r = requests.get(f"{URL}/rest/v1/onchain_hist_sample_v1", headers={**SB, "Range": f"{off}-{off+999}"}, params={"select": "bar_time", "symbol": f"eq.{symbol}"}, timeout=60); r.raise_for_status()
        b = r.json(); done |= {x["bar_time"][:10] for x in b}
        if len(b) < 1000: return done
        off += 1000

def main():
    contracts = requests.get(f"{URL}/rest/v1/token_contracts_v1", headers=SB, timeout=60,
        params={"select": "symbol,chain,contract_address", "confidence": "eq.high", "chain": "eq.ethereum", "order": "symbol", "limit": "500"}).json()
    contracts = [c for i, c in enumerate(contracts) if i % NSH == SHARD]
    days = pd_days()
    t0 = time.time(); stats = {"rows": 0, "failed": 0, "incomplete": 0}
    print("shard", SHARD, "of", NSH, "contracts", [c["symbol"] for c in contracts], "days", len(days), flush=True)
    for c in contracts:
        base = f"https://api.routescan.io/v2/network/mainnet/evm/1/erc20-transfers"
        done = existing(c["symbol"]); rows = []
        for D in days:
            if D.strftime("%Y-%m-%d") in done: continue
            if time.time() > DEADLINE: print("deadline reached; rerun to resume"); post("onchain_hist_sample_v1", rows, "symbol,bar_time"); print(json.dumps(stats)); return
            try:
                start, end = D - timedelta(hours=30), D
                items, complete = fetch_interval(base, c["contract_address"], start, end)
            except Exception as exc:
                stats["failed"] += 1; print("FAILED", c["symbol"], D.date(), str(exc)[:120], flush=True); continue
            rcv, snd, tc_r, tc_p, v_r, v_p, wn, wv = set(), set(), 0, 0, 0.0, 0.0, 0, 0.0
            for it in items:
                ts = datetime.fromisoformat(it["timestamp"].replace("Z", "+00:00")); recent = ts >= D - timedelta(hours=6)
                usd = it.get("amountInUsd"); usd = float(usd) if usd not in (None, "") else 0.0
                if recent:
                    tc_r += 1; v_r += usd; rcv.add(it["to"]); snd.add(it["from"])
                    if usd >= WHALE_USD: wn += 1; wv += usd
                else: tc_p += 1; v_p += usd
            stats["incomplete"] += (not complete)
            rows.append({"symbol": c["symbol"], "bar_time": D.isoformat(), "tc_recent6": tc_r, "tc_prior24": tc_p, "vol_recent6_usd": v_r, "vol_prior24_usd": v_p,
                         "whale_n_6": wn, "whale_usd_6": wv, "rcv_recent6": len(rcv), "snd_recent6": len(snd), "complete": complete})
            if len(rows) >= 25: post("onchain_hist_sample_v1", rows, "symbol,bar_time"); stats["rows"] += len(rows); rows = []
        if rows: post("onchain_hist_sample_v1", rows, "symbol,bar_time"); stats["rows"] += len(rows)
        print(c["symbol"], "done", int(time.time() - t0), "s", flush=True)
    print(json.dumps(stats))

def pd_days():
    d = datetime.fromisoformat(START_DATE).replace(tzinfo=timezone.utc); end = datetime.fromisoformat(END_DATE).replace(tzinfo=timezone.utc); out = []
    while d <= end: out.append(d); d += timedelta(days=STEP)
    return out
main()
