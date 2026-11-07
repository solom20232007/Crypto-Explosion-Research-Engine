"""Backfill daily futures context (open interest, long/short + taker ratios, funding) from data.binance.vision.

Binance's futures API is geo-blocked from GitHub runners (HTTP 451), but the public data archive is reachable.
Spot symbols are mapped to USDT-M perpetuals (SYMBOL, else 1000SYMBOL). Research context only.
"""
import csv, io, json, os, re, sys, zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
import requests

URL = os.environ["SUPABASE_URL"].rstrip("/"); KEY = os.environ["SUPABASE_SECRET_KEY"]
SB = {"apikey": KEY, "Authorization": f"Bearer {KEY}", "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates"}
VISION = "https://data.binance.vision/data/futures/um"
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
START = date.fromisoformat(os.environ.get("START_DATE", "2025-10-01"))
END = date.fromisoformat(os.environ["END_DATE"]) if os.environ.get("END_DATE") else date.today() - timedelta(days=1)
ONLY = [s for s in os.environ.get("SYMBOLS", "").split() if s]
NOT_CANDIDATES = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "EURIUSDT", "WBETHUSDT", "WBTCUSDT"}
sess = requests.Session()

def perps():
    out, marker = set(), ""
    while True:
        r = sess.get(S3, params={"delimiter": "/", "prefix": "data/futures/um/monthly/fundingRate/", "marker": marker, "max-keys": 1000}, timeout=60)
        found = re.findall(r"<Prefix>data/futures/um/monthly/fundingRate/([A-Z0-9]+)/</Prefix>", r.text); out.update(found)
        if "<IsTruncated>true</IsTruncated>" in r.text and found: marker = f"data/futures/um/monthly/fundingRate/{found[-1]}/"
        else: return out

def fetch_zip_rows(url):
    r = sess.get(url, timeout=60)
    if r.status_code != 200: return None
    z = zipfile.ZipFile(io.BytesIO(r.content)); return list(csv.DictReader(io.TextIOWrapper(z.open(z.namelist()[0]), encoding="utf-8")))

def f(x):
    try: return float(x)
    except (TypeError, ValueError): return None

def mean(v):
    v = [x for x in v if x is not None]; return sum(v) / len(v) if v else None

def do_symbol(spot, perp):
    days = defaultdict(dict)
    # funding (monthly files)
    m = date(START.year, START.month, 1)
    while m <= END:
        rows = fetch_zip_rows(f"{VISION}/monthly/fundingRate/{perp}/{perp}-fundingRate-{m.year}-{m.month:02d}.zip") or []
        per = defaultdict(list)
        for r in rows:
            t = r.get("calc_time") or r.get("calcTime"); rate = f(r.get("last_funding_rate") or r.get("lastFundingRate"))
            if t and rate is not None:
                per[date.fromordinal(date(1970, 1, 1).toordinal() + int(t) // 86_400_000)].append(rate)
        for d, v in per.items(): days[d].update(funding_mean=mean(v), funding_sum=sum(v), funding_n=len(v))
        m = date(m.year + (m.month == 12), m.month % 12 + 1, 1)
    # daily metrics
    def one(d):
        rows = fetch_zip_rows(f"{VISION}/daily/metrics/{perp}/{perp}-metrics-{d.isoformat()}.zip")
        if not rows: return d, None
        oi = [f(r.get("sum_open_interest_value")) for r in rows]; oi = [x for x in oi if x is not None]
        return d, {"oi_value_end": oi[-1] if oi else None, "oi_value_start": oi[0] if oi else None,
                   "ls_ratio_mean": mean([f(r.get("count_long_short_ratio")) for r in rows]),
                   "top_ls_ratio_mean": mean([f(r.get("sum_toptrader_long_short_ratio")) for r in rows]),
                   "taker_ratio_mean": mean([f(r.get("sum_taker_long_short_vol_ratio")) for r in rows]), "metrics_rows": len(rows)}
    ds = [START + timedelta(days=i) for i in range((END - START).days + 1)]
    with ThreadPoolExecutor(12) as ex:
        for d, v in ex.map(one, ds):
            if v: days[d].update(v)
    out = []
    for d, v in sorted(days.items()):
        if d < START or d > END: continue
        s, e = v.get("oi_value_start"), v.get("oi_value_end")
        out.append({"symbol": spot, "perp_symbol": perp, "day": d.isoformat(), "oi_value_end": e, "oi_value_start": s,
                    "oi_chg_1d": (e / s - 1) if s and e else None, "ls_ratio_mean": v.get("ls_ratio_mean"), "top_ls_ratio_mean": v.get("top_ls_ratio_mean"),
                    "taker_ratio_mean": v.get("taker_ratio_mean"), "funding_mean": v.get("funding_mean"), "funding_sum": v.get("funding_sum"),
                    "funding_n": v.get("funding_n"), "metrics_rows": v.get("metrics_rows")})
    for i in range(0, len(out), 400):
        requests.post(f"{URL}/rest/v1/futures_daily_v1?on_conflict=symbol,day", headers=SB, data=json.dumps(out[i:i + 400]), timeout=120).raise_for_status()
    return len(out)

def main():
    available = perps()
    syms = [l.split("#")[0].strip().upper() for l in open("config/research_symbols.txt") if l.split("#")[0].strip()]
    syms = [s for s in syms if s not in NOT_CANDIDATES and (not ONLY or s in ONLY)]
    total, unmapped = 0, []
    for s in syms:
        perp = s if s in available else ("1000" + s if "1000" + s in available else None)
        if not perp: unmapped.append(s); continue
        try: n = do_symbol(s, perp); total += n; print(s, perp, n, flush=True)
        except Exception as exc: print("FAILED", s, str(exc)[:200], flush=True)
    print(json.dumps({"symbols": len(syms), "rows": total, "no_perp": unmapped, "window": [START.isoformat(), END.isoformat()]}))
main()
