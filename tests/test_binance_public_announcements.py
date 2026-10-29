import json
from datetime import datetime, timezone
from src.news_monitor import parse_binance_public_response, collect_binance_public_announcements

def test_public_binance_response_normalizes_article_and_receipt_time():
 payload={"data":{"catalogs":[{"catalogId":48,"catalogName":"New Cryptocurrency Listing","articles":[
  {"title":"Binance Will List ABC (ABC)","code":"abc123","releaseDate":1753257631403}]}]}}
 seen=datetime(2025,7,23,12,0,tzinfo=timezone.utc)
 rows=parse_binance_public_response(payload,seen)
 assert len(rows)==1
 row=rows[0]
 assert row["url"]=="https://www.binance.com/en/support/announcement/abc123"
 assert row["published_utc"]=="2025-07-23T08:00:31.403000Z"
 assert row["first_seen_utc"]=="2025-07-23T12:00:00Z"
 assert row["available_at_utc"]==row["first_seen_utc"]
 assert row["source_name"]=="Binance Public Announcements Page"
 assert row["catalog_name"]=="New Cryptocurrency Listing"

def test_public_binance_response_skips_malformed_articles():
 payload={"data":{"catalogs":[{"catalogName":"News","articles":[
  {"title":"Missing code"},{"code":"missing-title"},{"title":"Valid notice","code":"xyz"}]}]}}
 assert len(parse_binance_public_response(payload,datetime.now(timezone.utc)))==1

def test_public_binance_collector_uses_public_endpoint_and_parses_response():
 class Response:
  def raise_for_status(self): pass
  def json(self): return {"data":{"catalogs":[{"catalogName":"News","articles":[{"title":"ABC update","code":"abc"}]}]}}
 class Session:
  def get(self,url,**kwargs):
   assert "binance.com" in url
   assert kwargs["params"]["pageSize"]=="10"
   return Response()
 rows=collect_binance_public_announcements(Session(),limit=10,now=datetime(2025,1,1,tzinfo=timezone.utc))
 assert len(rows)==1 and rows[0]["url"].endswith("/abc")

def test_public_binance_collector_rejects_invalid_limit():
 try:
  collect_binance_public_announcements(limit=0)
  assert False,"invalid limit should fail"
 except ValueError:
  pass
