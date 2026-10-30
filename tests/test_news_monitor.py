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


def test_supabase_news_features_link_to_real_explosion_events_and_exclude_late_news(monkeypatch):
 import src.news_monitor as monitor
 stored=[]
 calls=[]
 monkeypatch.setattr(monitor,"_load_supabase_events",lambda url,key:[
  {"id":77,"symbol":"ABCUSDT","timeframe":"1h","event_time":"2024-10-02T10:00:00Z"}
 ])
 def fake_request(url,key,method,table,params=None,rows=None,prefer=None,offset=None,limit=1000):
  calls.append((method,table,params,rows,prefer))
  if method=="GET" and table=="news_events":
   if params and "fingerprint" in params:
    return []
   return list(stored)
  if method=="POST" and table=="news_events":
   for i,row in enumerate(rows,1):
    stored.append({"id":i,"matched_symbols":row["matched_symbols"],**row})
   return []
  if method=="PATCH" and table=="news_events":
   for row in stored:
    if params.get("id")==f"eq.{row['id']}": row.update(rows)
   return []
  if method=="POST" and table=="news_symbol_features":
   calls.append(("FEATURE_ROWS",rows))
   return []
  raise AssertionError((method,table,params))
 monkeypatch.setattr(monitor,"_supabase_request",fake_request)
 news=pd.DataFrame([
  {"title":"$ABC Binance listing","summary":"","url":"https://example.com/1",
   "source_domain":"example.com","source_name":"Example","source_tier":"official",
   "source_type":"exchange","category":"exchange_announcement","sentiment_hint":1,
   "published_utc":"2024-10-02T08:00:00Z","first_seen_utc":"2024-10-02T09:00:00Z",
   "available_at_utc":"2024-10-02T09:00:00Z","text":"$ABC Binance listing"},
  {"title":"$ABC security incident","summary":"","url":"https://example.com/2",
   "source_domain":"example.com","source_name":"Example","source_tier":"media",
   "source_type":"media","category":"security","sentiment_hint":-1,
   "published_utc":"2024-10-02T09:30:00Z","first_seen_utc":"2024-10-02T10:00:00Z",
   "available_at_utc":"2024-10-02T10:00:00Z","text":"$ABC security incident"}
 ])
 result=monitor.upload_news_to_supabase(news,"https://project.supabase.co","secret")
 feature_call=[x for x in calls if x[0]=="FEATURE_ROWS"][0]
 feature=feature_call[1][0]
 assert result["news_symbol_features_upserted"]==1
 assert feature["explosion_event_id"]==77
 assert feature["news_count_7d"]==1
 assert feature["exchange_announcement_count_7d"]==1
 assert feature["security_news_count_7d"]==0
 assert len(feature["matched_news_event_ids"])==1


def test_stale_published_articles_are_excluded_even_when_recently_discovered():
 candles=pd.DataFrame([{"symbol":"ETHUSDT","open_time":1761782400000}])
 news=pd.DataFrame([
  {"text":"Ethereum Foundation update","first_seen_utc":"2026-10-30T20:00:00Z",
   "available_at_utc":"2026-10-30T20:00:00Z","published_utc":"2020-11-26T00:00:00Z",
   "category":"general","sentiment_hint":0},
  {"text":"Ethereum network upgrade","first_seen_utc":"2026-10-30T20:00:00Z",
   "available_at_utc":"2026-10-30T20:00:00Z","published_utc":"2026-10-30T19:00:00Z",
   "category":"partnership","sentiment_hint":1}
 ])
 result=build_symbol_features(news,candles,{"ETH":["Ethereum"]})
 assert result.iloc[0]["news_count_7d"]==1

def test_major_coin_aliases_match_project_names():
 aliases={"BTC":["Bitcoin"],"ETH":["Ethereum"],"SOL":["Solana"]}
 assert match_symbols("Bitcoin and Solana market update",["BTCUSDT","ETHUSDT","SOLUSDT"],aliases)==["BTCUSDT","SOLUSDT"]
