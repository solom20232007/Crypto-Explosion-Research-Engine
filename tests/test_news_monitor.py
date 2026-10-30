from datetime import datetime, timezone
import pandas as pd
from src.news_monitor import build_symbol_features, classify, load_feed_config, match_symbols, parse_feed

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

def test_curated_feed_config_loads_enabled_sources(tmp_path):
 config=tmp_path/"feeds.json"
 config.write_text('{"sources":[{"name":"Primary","url":"https://example.com/rss","enabled":true},{"name":"Disabled","url":"https://example.org/rss","enabled":false}]}')
 sources=load_feed_config(config)
 assert len(sources)==1 and sources[0]["name"]=="Primary"

def test_feed_config_rejects_non_http_urls(tmp_path):
 config=tmp_path/"feeds.json"
 config.write_text('{"sources":[{"name":"Bad","url":"file:///tmp/feed.xml"}]}')
 try:
  load_feed_config(config)
  assert False, "invalid URL should fail"
 except ValueError:
  pass

def test_parsed_news_carries_source_provenance():
 row=parse_feed('<rss><channel><item><title>ABC update</title></item></channel></rss>',
  "https://example.com/rss",datetime(2024,10,1,12,tzinfo=timezone.utc),
  {"name":"Example Official","tier":"official_primary","type":"project_official"})[0]
 assert row["source_name"]=="Example Official"
 assert row["source_tier"]=="official_primary"
 assert row["source_type"]=="project_official"

from src.news_monitor import deduplicate_articles

def test_project_name_and_aliases_match_without_substring_false_positive():
 assert match_symbols("Avalanche announces a network upgrade",["AVAXUSDT"],{"AVAX":["Avalanche","avax token"]})==["AVAXUSDT"]
 assert match_symbols("NotAVAXUSDTX is unrelated",["AVAXUSDT"])==[]

def test_cross_source_duplicate_story_keeps_provenance():
 news=pd.DataFrame([
  {"title":"ABC announces major upgrade","url":"https://one.example/story","source_name":"One","source_tier":"established","source_type":"media","source_domain":"one.example","published_utc":"2025-01-01T10:00:00Z"},
  {"title":"ABC announces major upgrade","url":"https://one.example/story","source_name":"Two","source_tier":"official","source_type":"project","source_domain":"two.example","published_utc":"2025-01-01T10:00:00Z"}])
 result=deduplicate_articles(news)
 assert len(result)==1
 assert result.iloc[0]["duplicate_source_count"]==2
 assert result.iloc[0]["source_name"]=="One|Two"


from src.news_monitor import build_signal_time_features

def test_signal_time_features_use_first_seen_and_exclude_late_news():
 news=pd.DataFrame([
  {"text":"$ABC Binance listing","available_at_utc":"2024-10-02T09:00:00Z","first_seen_utc":"2024-10-02T09:00:00Z","category":"exchange_announcement","sentiment_hint":1},
  {"text":"$ABC token unlock","available_at_utc":"2024-10-02T10:00:00Z","first_seen_utc":"2024-10-02T10:00:00Z","category":"token_unlock","sentiment_hint":-1},
  {"text":"$ABC security incident","available_at_utc":"2024-10-02T11:00:00Z","first_seen_utc":"2024-10-02T11:00:00Z","category":"security","sentiment_hint":-1},
 ])
 signals=pd.DataFrame([
  {"symbol":"ABCUSDT","signal_time_utc":"2024-10-02T10:30:00Z"},
  {"symbol":"ABCUSDT","signal_time_utc":"2024-10-02T11:00:00Z"},
 ])
 result=build_signal_time_features(news,signals).set_index("signal_time_utc")
 first=result.loc["2024-10-02T10:30:00Z"]
 assert first["news_count_24h"]==2
 assert first["exchange_announcement_count_7d"]==1
 assert first["token_unlock_count_7d"]==1
 assert first["security_news_count_7d"]==0
 second=result.loc["2024-10-02T11:00:00Z"]
 # A story first seen at exactly the signal timestamp is excluded conservatively.
 assert second["news_count_24h"]==2
 assert second["security_news_count_7d"]==0

def test_signal_time_features_do_not_use_published_time_as_availability():
 news=pd.DataFrame([{
  "text":"$ABC partnership",
  "published_utc":"2024-10-01T08:00:00Z",
  "first_seen_utc":"2024-10-02T12:00:00Z",
  "available_at_utc":"2024-10-02T12:00:00Z",
  "category":"partnership","sentiment_hint":1
 }])
 signals=pd.DataFrame([{"symbol":"ABCUSDT","signal_time_utc":"2024-10-02T11:59:00Z"}])
 result=build_signal_time_features(news,signals)
 assert result.iloc[0]["news_count_7d"]==0

def test_signal_time_features_require_explicit_signal_timestamps():
 signals=pd.DataFrame([{"symbol":"ABCUSDT","timestamp":"2024-10-02T12:00:00Z"}])
 try:
  build_signal_time_features(pd.DataFrame(),signals)
  assert False, "missing signal_time_utc should fail"
 except ValueError as exc:
  assert "signal_time_utc" in str(exc)
