"""Compact on-chain pilot: per-token hourly activity + whale transfers (>= $25k) for mapped contracts.

Uses Routescan's erc20-transfers endpoint filtered by tokenAddress (Ethereum, BSC). Only contracts with
confidence='high' in token_contracts_v1 are collected. Re-fetches the last 3 complete hours each run (idempotent upserts).
Research indicators only: transfer direction/intent is NOT inferred.
"""
import json, os, time
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse
import requests

URL = os.environ["SUPABASE_URL"].rstrip("/"); KEY = os.environ["SUPABASE_SECRET_KEY"]; RS = os.environ.get("ROUTESCAN_API_KEY", "")
SB = {"apikey": KEY, "Authorization": f"Bearer {KEY}"}
CHAIN_ID = {"ethereum": 1, "bsc": 56}
WHALE_USD = 25_000
MAX_PAGES = int(os.environ.get("MAX_PAGES", "40")); HOURS = int(os.environ.get("HOURS", "3"))
S = requests.Session()

def get(url, params):
    for i in range(6):
        r = S.get(url, params=params, headers={"apikey": RS} if RS else {}, timeout=(10, 45))
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(min(30, 2 ** i)); continue
        r.raise_for_status(); return r.json()
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

def main():
    contracts = requests.get(f"{URL}/rest/v1/token_contracts_v1", headers=SB, timeout=60,
        params={"select": "symbol,chain,contract_address", "confidence": "eq.high", "chain": "in.(ethereum,bsc)", "limit": "500"}).json()
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0); start = end - timedelta(hours=HOURS)
    iso = lambda d: d.strftime("%Y-%m-%dT%H:%M:%SZ")
    hourly, whales, truncated, usd_missing_total = [], [], 0, 0
    for c in contracts:
        base = f"https://api.routescan.io/v2/network/mainnet/evm/{CHAIN_ID[c['chain']]}/erc20-transfers"
        params = {"tokenAddress": c["contract_address"], "timestampFrom": iso(start), "timestampTo": iso(end), "sort": "desc", "limit": 100}
        items, cur, complete = [], None, False
        for _ in range(MAX_PAGES):
            if cur: params["next"] = cur
            p = get(base, params); got = p.get("items") or []; items += got
            cur = cursor(p.get("link") or {})
            if not cur or not got: complete = True; break
            time.sleep(0.22 if RS else 0.55)
        if not complete: truncated += 1
        buckets = {}
        for it in items:
            ts = datetime.fromisoformat(it["timestamp"].replace("Z", "+00:00")); h = ts.replace(minute=0, second=0, microsecond=0)
            b = buckets.setdefault(h, {"n": 0, "vol": 0.0, "wn": 0, "wv": 0.0, "snd": set(), "rcv": set(), "miss": 0})
            usd = it.get("amountInUsd"); b["n"] += 1; b["snd"].add(it["from"]); b["rcv"].add(it["to"])
            if usd is None or usd == "": b["miss"] += 1; continue
            usd = float(usd); b["vol"] += usd
            if usd >= WHALE_USD:
                b["wn"] += 1; b["wv"] += usd
                dec = it.get("tokenDecimals"); amt = float(it["amount"]) / 10 ** int(dec) if dec is not None and it.get("amount") else None
                whales.append({"chain": c["chain"], "tx_hash": it["txHash"], "event_index": int(it.get("logIndex") or 0), "symbol": c["symbol"],
                               "block_time_utc": ts.isoformat(), "from_address": it["from"], "to_address": it["to"], "amount_token": amt, "amount_usd": usd})
        for k in range(HOURS):
            h = start + timedelta(hours=k); b = buckets.get(h, {"n": 0, "vol": 0.0, "wn": 0, "wv": 0.0, "snd": set(), "rcv": set(), "miss": 0})
            usd_missing_total += b["miss"]
            hourly.append({"symbol": c["symbol"], "chain": c["chain"], "hour_utc": h.isoformat(), "transfer_count": b["n"], "volume_usd": b["vol"],
                           "whale_count": b["wn"], "whale_volume_usd": b["wv"], "unique_senders": len(b["snd"]), "unique_receivers": len(b["rcv"]),
                           "usd_missing_count": b["miss"], "complete": complete})
    post("onchain_token_hourly_v1", hourly, "symbol,chain,hour_utc")
    if whales: post("onchain_whale_transfers_v1", whales, "chain,tx_hash,event_index")
    print(json.dumps({"contracts": len(contracts), "window": [iso(start), iso(end)], "hourly_rows": len(hourly), "whale_transfers": len(whales),
                      "tokens_truncated": truncated, "transfers_without_usd": usd_missing_total}))
main()
