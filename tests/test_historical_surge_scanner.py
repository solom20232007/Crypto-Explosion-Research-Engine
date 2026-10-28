import pandas as pd
from src.historical_surge_scanner import detect_surge_events, discover_target_symbols

def test_excludes_majors_stables_and_leveraged_tokens():
    p={"symbols":[{"symbol":"BTCUSDT","baseAsset":"BTC","quoteAsset":"USDT","status":"TRADING"},{"symbol":"ETHUSDT","baseAsset":"ETH","quoteAsset":"USDT","status":"TRADING"},{"symbol":"SOLUSDT","baseAsset":"SOL","quoteAsset":"USDT","status":"TRADING"},{"symbol":"USDCUSDT","baseAsset":"USDC","quoteAsset":"USDT","status":"TRADING"},{"symbol":"XYZUPUSDT","baseAsset":"XYZUP","quoteAsset":"USDT","status":"TRADING"},{"symbol":"PEPEUSDT","baseAsset":"PEPE","quoteAsset":"USDT","status":"TRADING"},{"symbol":"ABCUSDT","baseAsset":"ABC","quoteAsset":"USDT","status":"BREAK"}]}
    assert discover_target_symbols(p)==["PEPEUSDT"]

def test_detects_surge_events():
    f=pd.DataFrame({"open_time":[i*86400000 for i in range(10)],"open":[10]*10,"high":[10,10,10,10,10,10,10,13,14,15],"low":[9]*10,"close":[10,10,10,10,10,10,10,12,13,14],"volume":[100]*10,"quote_volume":[200000]*10})
    e=detect_surge_events(f,"PEPEUSDT")
    assert not e.empty and set(e.symbol)=={"PEPEUSDT"}
    assert e.event_date_utc.is_monotonic_increasing and (e.daily_high_change_pct>=25).any()
