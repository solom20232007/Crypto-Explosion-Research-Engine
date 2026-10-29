import json
from datetime import datetime, timezone
from src.news_monitor import parse_binance_announcement_message, build_binance_ws_url

def test_binance_announcement_payload_is_normalized_with_receipt_time():
 frame={"type":"DATA","topic":"com_announcement_en","data":json.dumps({
  "catalogId":161,"catalogName":"Delisting","publishDate":1753257631403,
  "title":"ABCUSDT delisting notice","body":"Trading will stop"})}
 seen=datetime(2025,7,23,12,0,tzinfo=timezone.utc)
 row=parse_binance_announcement_message(frame,seen)
 assert row["source_name"]=="Binance Announcements WebSocket"
 assert row["source_tier"]=="official_primary"
 assert row["source_type"]=="exchange_official"
 assert row["first_seen_utc"]=="2025-07-23T12:00:00Z"
 assert row["available_at_utc"]==row["first_seen_utc"]
 assert row["published_utc"]=="2025-07-23T09:20:31.403000Z"
 assert row["catalog_name"]=="Delisting" and row["catalog_id"]==161
 assert row["url"]==""

def test_binance_announcement_parser_ignores_non_data_and_other_topics():
 assert parse_binance_announcement_message({"type":"COMMAND","data":"SUCCESS"},datetime.now(timezone.utc)) is None
 assert parse_binance_announcement_message({"type":"DATA","topic":"other","data":"{}"},datetime.now(timezone.utc)) is None

def test_binance_websocket_url_contains_valid_hmac_signature():
 from urllib.parse import parse_qs,urlparse
 import hashlib,hmac
 url=build_binance_ws_url("secret",1753257631403,"fixed-random")
 query=parse_qs(urlparse(url).query)
 params={k:v[0] for k,v in query.items() if k!="signature"}
 from urllib.parse import urlencode
 payload=urlencode(sorted(params.items()))
 expected=hmac.new(b"secret",payload.encode(),hashlib.sha256).hexdigest()
 assert query["signature"][0]==expected
 assert query["topic"]==["com_announcement_en"]
