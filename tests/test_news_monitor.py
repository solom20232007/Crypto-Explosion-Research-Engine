from datetime import datetime, timezone
import pandas as pd
from src.news_monitor import build_symbol_features, classify, match_symbols, parse_feed

def test_rss_keeps_published_and_first_seen_times():
 xml="<rss><channel><item><title>ABC listed on Binance</title><link>https://example.com/a</link><pubDate>Tue, 01 Oct 2024 10:00:00 GMT</pubDate><description>Exchange listing</description></item></channel></rss>"
 row=parse_feed(xml,"https://example.com/rss",datetime(2024,10,1,12,tzinfo=timezone.utc))[0]
 assert row["published_utc"]=="2024-10-01T10:00:00Z"
 assert row["first_seen_utc"]=="2024-10-01T12:00:00Z"
 assert row["available_at_utc"]==row["first_seen_utc"]

def test_atom_feed():
 xml='<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>ABC update</title><link href="https://example.com/a"/><updated>2024-10-01T10:00:00Z</updated><summary>Protocol update</summary></entry></feed>'
 row=parse_feed(xml,"https://example.com/feed",datetime(2024,10,1,11,tzinfo=timezone.utc))[0]
 assert row["url"]=="https://example.com/a" and row["published_utc"]=="2024-10-01T10:00:00Z"

def test_ticker_matching_uses_boundaries_and_skips_short_symbols():
 assert match_symbols("ABC announces upgrade; X is up",["ABCUSDT","XUSDT"])==["ABCUSDT"]

def test_late_discovered_news_does_not_leak_into_prior_signal():
 candles=pd.DataFrame([{"symbol":"ABCUSDT","open_time":1727740800000},{"symbol":"ABCUSDT","open_time":1727827200000}])
 news=pd.DataFrame([
  {"text":"$ABC listing","first_seen_utc":"2024-10-01T23:59:00Z","category":"exchange_announcement","sentiment_hint":1},
  {"text":"$ABC late report","first_seen_utc":"2024-10-03T00:01:00Z","category":"general","sentiment_hint":0}])
 result=build_symbol_features(news,candles).set_index("signal_time_utc")
 assert result.loc["2024-10-02T00:00:00Z","news_count_7d"]==1
 assert result.loc["2024-10-03T00:00:00Z","news_count_7d"]==1

def test_keyword_categories_are_weak_hints():
 category,score=classify("Exchange listing partnership")
 assert "exchange_announcement" in category and score==1
