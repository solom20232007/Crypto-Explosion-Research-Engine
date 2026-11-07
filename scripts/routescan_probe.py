"""One-off probe: does Routescan's global erc20-transfers endpoint support filtering by token contract?"""
import json, os, requests
URL = os.environ["SUPABASE_URL"].rstrip("/"); KEY = os.environ["SUPABASE_SECRET_KEY"]; RS = os.environ.get("ROUTESCAN_API_KEY", "")
rows = requests.get(f"{URL}/rest/v1/token_contracts_v1", headers={"apikey": KEY, "Authorization": f"Bearer {KEY}"},
                    params={"select": "symbol,chain,contract_address,decimals", "confidence": "eq.high", "chain": "in.(ethereum,bsc)", "limit": "400"}, timeout=60).json()
print("high-confidence EVM rows", len(rows), "ethereum", sum(r["chain"] == "ethereum" for r in rows), "bsc", sum(r["chain"] == "bsc" for r in rows))
cid = {"ethereum": 1, "bsc": 56}
target = next(r for r in rows if r["symbol"] == "LDOUSDT" and r["chain"] == "ethereum")
H = {"apikey": RS} if RS else {}
base = f"https://api.routescan.io/v2/network/mainnet/evm/{cid['ethereum']}/erc20-transfers"
for name in ["tokenAddresses", "tokenAddress", "token", "contractAddress", "tokens"]:
    r = requests.get(base, headers=H, params={name: target["contract_address"], "limit": 20, "sort": "desc"}, timeout=60)
    try: items = r.json().get("items", [])
    except Exception: items = []
    toks = {str(i.get("tokenAddress", "")).lower() for i in items}
    print(name, r.status_code, "items", len(items), "distinct_tokens", len(toks), "match_only", toks == {target["contract_address"].lower()} if items else None)
    if items and toks == {target["contract_address"].lower()}:
        print("WORKS:", name); print("sample keys:", sorted(items[0].keys())); print("sample:", json.dumps({k: items[0].get(k) for k in ["timestamp", "amount", "amountInUsd", "tokenSymbol", "tokenDecimals"]}))
        break
else:
    r = requests.get(base, headers=H, params={"limit": 3}, timeout=60); print("unfiltered sample keys:", sorted(r.json().get("items", [{}])[0].keys()))
