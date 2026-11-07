"""Probe: which research symbols have USDT-M perpetuals, and which futures data endpoints are reachable from the runner."""
import re, requests
syms = [l.split("#")[0].strip().upper() for l in open("config/research_symbols.txt") if l.split("#")[0].strip()]
for name, url in [("fapi exchangeInfo", "https://fapi.binance.com/fapi/v1/exchangeInfo"),
                  ("fapi fundingRate", "https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=2"),
                  ("fapi OI hist", "https://fapi.binance.com/futures/data/openInterestHist?symbol=BTCUSDT&period=1h&limit=2"),
                  ("vision listing", "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?delimiter=/&prefix=data/futures/um/monthly/fundingRate/&max-keys=1000")]:
    try:
        r = requests.get(url, timeout=30); print(name, r.status_code, r.text[:150].replace("\n", " "))
    except Exception as e: print(name, "ERR", e)
# symbols with perps per data.binance.vision (paginated S3 listing)
perps, marker = set(), ""
while True:
    r = requests.get("https://s3-ap-northeast-1.amazonaws.com/data.binance.vision", params={"delimiter": "/", "prefix": "data/futures/um/monthly/fundingRate/", "marker": marker, "max-keys": 1000}, timeout=60)
    found = re.findall(r"<Prefix>data/futures/um/monthly/fundingRate/([A-Z0-9]+)/</Prefix>", r.text); perps.update(found)
    if "<IsTruncated>true</IsTruncated>" in r.text and found: marker = f"data/futures/um/monthly/fundingRate/{found[-1]}/"
    else: break
cov = [s for s in syms if s in perps]; print("perp symbols on vision", len(perps), "research with perps", len(cov), "of", len(syms)); print("MISSING", [s for s in syms if s not in perps])
for pre in ["data/futures/um/daily/metrics/LDOUSDT/", "data/futures/um/monthly/fundingRate/LDOUSDT/"]:
    r = requests.get("https://s3-ap-northeast-1.amazonaws.com/data.binance.vision", params={"prefix": pre, "max-keys": 1000}, timeout=60)
    keys = re.findall(r"<Key>([^<]+\.zip)</Key>", r.text); print(pre, len(keys), keys[:1], keys[-1:] )
