"""Timestamp-safe RSS/Atom news ingestion for pre-breakout research."""
from __future__ import annotations
import argparse, hashlib, json, os, re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd
import requests

OUT = Path("reports/news_monitor")
CATEGORY_TERMS = {
 "exchange_announcement": ("binance","coinbase","bybit","okx","listing","delisting","trading pairs"),
 "token_unlock": ("token unlock","vesting","unlocking","cliff","token release"),
 "security": ("exploit","hack","hacked","security incident","bridge attack"),
 "partnership": ("partnership","integration","collaboration"),
 "regulatory": ("sec ","lawsuit","regulation","legal action"),
 "fundraising": ("funding","raises","investment round"),
}
POSITIVE = ("partnership","approval","listing","funding","upgrade","launch")
NEGATIVE = ("hack","exploit","delist","lawsuit","ban","breach","unlock","suspension")

def utc_iso(dt):
 if dt.tzinfo is None: dt=dt.replace(tzinfo=timezone.utc)
 return dt.astimezone(timezone.utc).isoformat().replace("+00:00","Z")

def parse_datetime(value):
 if not value: return None
 try: dt=datetime.fromisoformat(value.replace("Z","+00:00"))
 except ValueError:
  try: dt=parsedate_to_datetime(value)
  except (TypeError,ValueError,OverflowError): return None
 if dt.tzinfo is None: dt=dt.replace(tzinfo=timezone.utc)
 return dt.astimezone(timezone.utc)

def lname(tag): return tag.rsplit("}",1)[-1].lower()

def child_text(el,names):
 for node in el.iter():
  if lname(node.tag) in names and node.text: return node.text.strip()
 return ""

def parse_feed(xml_text,source_url,first_seen,source_meta=None):
 root=ET.fromstring(xml_text); rows=[]
 entries=[e for e in root.iter() if lname(e.tag) in {"item","entry"}]
 for entry in entries:
  title=child_text(entry,{"title"}); link=""
  for node in entry.iter():
   if lname(node.tag)=="link":
    link=node.attrib.get("href") or (node.text or "").strip()
    if link: break
  pubraw=child_text(entry,{"pubdate","published","updated","date"})
  published=parse_datetime(pubraw); summary=child_text(entry,{"description","summary","encoded","content"})
  if not title: continue
  rows.append({"title":title,"summary":summary,"url":link,"source_domain":urlparse(source_url).netloc.lower(),
   "published_utc":utc_iso(published) if published else "","first_seen_utc":utc_iso(first_seen),
   "available_at_utc":utc_iso(first_seen),"published_time_missing":published is None,"text":(title+" "+summary).strip(),
   "source_name":(source_meta or {}).get("name",urlparse(source_url).netloc.lower()),"source_tier":(source_meta or {}).get("tier","unrated"),"source_type":(source_meta or {}).get("type","media")})
 return rows

def classify(text):
 low=text.lower()
 cats=[name for name,terms in CATEGORY_TERMS.items() if any(t in low for t in terms)]
 score=int(any(t in low for t in POSITIVE))-int(any(t in low for t in NEGATIVE))
 return "|".join(cats) if cats else "general",score

def _terms_for_symbol(symbol,aliases):
 base=symbol[:-4] if symbol.endswith("USDT") else symbol
 values=aliases.get(base,[])
 if isinstance(values,str): values=[values]
 terms={base}
 for value in values:
  if isinstance(value,dict):
   terms.update(str(x).upper() for x in value.get("aliases",[]) if x)
   if value.get("name"): terms.add(str(value["name"]).upper())
  else: terms.add(str(value).upper())
 return {x.strip() for x in terms if len(x.strip())>=3}

def match_symbols(text,symbols,aliases=None):
 aliases=aliases or {}; upper=text.upper(); found=[]
 for symbol in symbols:
  for term in _terms_for_symbol(symbol,aliases):
   if re.search(r"(?<![A-Z0-9])\$?"+re.escape(term)+r"(?![A-Z0-9])",upper):
    found.append(symbol); break
 return found

def deduplicate_articles(news):
 """Collapse cross-publisher copies while retaining all source provenance."""
 if news.empty: return news
 frame=news.copy()
 frame["_url_key"]=frame.get("url",pd.Series("",index=frame.index)).fillna("").astype(str).str.lower().str.rstrip("/")
 frame["_title_key"]=frame.get("title",pd.Series("",index=frame.index)).fillna("").astype(str).str.lower()
 frame["_title_key"]=frame["_title_key"].str.replace(r"[^a-z0-9]+"," ",regex=True).str.strip()
 frame["_published_key"]=pd.to_datetime(frame.get("published_utc",pd.Series("",index=frame.index)),utc=True,errors="coerce").dt.floor("D")
 frame["_fallback_key"]=frame["_title_key"]+"|"+frame["_published_key"].astype(str)
 frame["_fallback_key"]=frame["_fallback_key"].where(frame["_published_key"].notna(),frame["_title_key"]+"|"+frame.get("source_domain",pd.Series("",index=frame.index)).fillna("").astype(str))
 frame["_dedup_key"]=frame["_url_key"].where(frame["_url_key"]!="",frame["_fallback_key"])
 rows=[]
 for _,group in frame.groupby("_dedup_key",dropna=False,sort=False):
  row=group.iloc[0].drop(labels=["_url_key","_title_key","_published_key","_fallback_key","_dedup_key"]).to_dict()
  for col in ("source_name","source_tier","source_type","source_domain"):
   vals=sorted({str(x) for x in group[col].dropna() if str(x)})
   row[col]="|".join(vals) if vals else row.get(col,"")
  row["duplicate_source_count"]=len(group)
  rows.append(row)
 return pd.DataFrame(rows)

def load_feed_config(path):
 data=json.loads(Path(path).read_text(encoding="utf-8"))
 sources=data.get("sources",[])
 enabled=[x for x in sources if x.get("enabled",True)]
 if not enabled: raise ValueError("Feed config has no enabled sources")
 for source in enabled:
  if not source.get("url","").startswith(("https://","http://")):
   raise ValueError(f"Invalid feed URL for source: {source.get('name','unnamed')}")
 return enabled

def collect_feeds(feed_urls,session=requests,now=None):
 seen=now or datetime.now(timezone.utc); rows=[]; errors=[]
 for item in feed_urls:
  meta=item if isinstance(item,dict) else {"url":item}
  url=meta["url"]
  try:
   response=session.get(url,timeout=20,headers={"User-Agent":"PreBreakoutResearchBot/1.0"})
   response.raise_for_status(); rows.extend(parse_feed(response.text,url,seen,meta))
  except Exception as exc: errors.append({"feed":url,"source_name":meta.get("name",url),"error":str(exc)})
 return rows,errors

def build_symbol_features(news,candles,aliases=None):
 missing={"symbol","open_time"}-set(candles.columns)
 if missing: raise ValueError(f"candles missing columns: {sorted(missing)}")
 n=news.copy()
 if n.empty: n=pd.DataFrame(columns=["text","first_seen_utc","category","sentiment_hint"])
 n["available_at"]=pd.to_datetime(n.get("available_at_utc",n.get("first_seen_utc",pd.Series(dtype=str))),utc=True,errors="coerce")
 n["published_at"]=pd.to_datetime(n.get("published_utc",pd.Series(index=n.index,dtype=str)),utc=True,errors="coerce")
 n["category"]=n.get("category",pd.Series(dtype=str)).fillna("general")
 n["sentiment_hint"]=pd.to_numeric(n.get("sentiment_hint",pd.Series(dtype=int)),errors="coerce").fillna(0)
 n["text"]=n.get("text",pd.Series(dtype=str)).fillna("")
 symbols=sorted(candles.symbol.dropna().astype(str).unique()); matched=[]
 for _,item in n.iterrows():
  for symbol in match_symbols(str(item.get("text","")),symbols,aliases):
   matched.append({**item.to_dict(),"symbol":symbol})
 m=pd.DataFrame(matched); results=[]
 for row in candles[["symbol","open_time"]].drop_duplicates().itertuples(index=False):
  signal=pd.to_datetime(row.open_time,unit="ms",utc=True)+pd.Timedelta(days=1)
  subset=m[m.symbol==row.symbol].copy() if not m.empty else m.copy()
  if not subset.empty:
   # A feed snapshot may expose years-old articles today. Receipt time alone
   # must not make stale publications look like fresh news.
   known_pub=subset.published_at.notna()
   subset=subset[(subset.available_at<signal)&
    (~known_pub | ((subset.published_at<=signal)&(subset.published_at>=signal-pd.Timedelta(days=7))))]
   subset=subset[subset.available_at>=signal-pd.Timedelta(days=7)]
  r24=subset[subset.available_at>=signal-pd.Timedelta(hours=24)]
  r7=subset
  cats=r7.category.astype(str)
  results.append({"symbol":row.symbol,"signal_time_utc":signal.isoformat().replace("+00:00","Z"),
   "news_count_24h":len(r24),"news_count_7d":len(r7),
   "exchange_announcement_count_7d":cats.str.contains("exchange_announcement").sum(),
   "token_unlock_count_7d":cats.str.contains("token_unlock").sum(),
   "security_news_count_7d":cats.str.contains("security").sum(),
   "positive_news_hint_count_7d":(r7.sentiment_hint>0).sum(),
   "negative_news_hint_count_7d":(r7.sentiment_hint<0).sum()})
 return pd.DataFrame(results)

def parse_binance_announcement_message(message, first_seen):
 """Normalize one official Binance Announcements WebSocket DATA frame."""
 if isinstance(message, bytes):
  message=message.decode("utf-8")
 frame=json.loads(message) if isinstance(message,str) else message
 if not isinstance(frame,dict) or frame.get("type")!="DATA":
  return None
 if frame.get("topic")!="com_announcement_en":
  return None
 payload=frame.get("data",{})
 if isinstance(payload,str):
  payload=json.loads(payload)
 if not isinstance(payload,dict) or not payload.get("title"):
  return None
 published_ms=payload.get("publishDate")
 published=None
 if published_ms is not None:
  try: published=datetime.fromtimestamp(int(published_ms)/1000,tz=timezone.utc)
  except (TypeError,ValueError,OverflowError,OSError): published=None
 title=str(payload.get("title","")).strip()
 body=str(payload.get("body","")).strip()
 text_value=(title+" "+body).strip()
 category,score=classify(text_value)
 seen=first_seen if isinstance(first_seen,datetime) else parse_datetime(str(first_seen))
 if seen is None:
  raise ValueError("first_seen must be a valid timestamp")
 return {"title":title,"summary":body,"url":"","source_domain":"binance.com",
  "published_utc":utc_iso(published) if published else "",
  "first_seen_utc":utc_iso(seen),"available_at_utc":utc_iso(seen),
  "published_time_missing":published is None,"text":text_value,
  "source_name":"Binance Announcements WebSocket","source_tier":"official_primary",
  "source_type":"exchange_official","category":category,"sentiment_hint":score,
  "catalog_id":payload.get("catalogId"),"catalog_name":str(payload.get("catalogName",""))}


def parse_binance_public_response(payload, first_seen, language="en"):
 """Normalize Binance's public CMS article-list response (best-effort, undocumented endpoint)."""
 if isinstance(payload,str):
  payload=json.loads(payload)
 if not isinstance(payload,dict):
  raise ValueError("Binance public response must be a JSON object")
 data=payload.get("data") or {}
 catalogs=data.get("catalogs") or []
 if not isinstance(catalogs,list):
  raise ValueError("Unexpected Binance public CMS catalogs structure")
 seen=first_seen if isinstance(first_seen,datetime) else parse_datetime(str(first_seen))
 if seen is None:
  raise ValueError("first_seen must be a valid timestamp")
 rows=[]
 for catalog in catalogs:
  if not isinstance(catalog,dict): continue
  category_name=str(catalog.get("catalogName") or catalog.get("name") or "").strip()
  articles=catalog.get("articles") or []
  if not isinstance(articles,list): continue
  for article in articles:
   if not isinstance(article,dict): continue
   title=str(article.get("title") or "").strip()
   code=str(article.get("code") or article.get("articleCode") or "").strip()
   if not title or not code: continue
   release=article.get("releaseDate") or article.get("publishDate")
   published=None
   if release is not None:
    try:
     numeric=float(release)
     if numeric > 1e12: numeric /= 1000
     published=datetime.fromtimestamp(numeric,tz=timezone.utc)
    except (TypeError,ValueError,OverflowError,OSError): published=None
   url=f"https://www.binance.com/{language}/support/announcement/{code}"
   text_value=title
   category,score=classify(text_value+" "+category_name)
   rows.append({"title":title,"summary":"","url":url,"source_domain":"binance.com",
    "published_utc":utc_iso(published) if published else "",
    "first_seen_utc":utc_iso(seen),"available_at_utc":utc_iso(seen),
    "published_time_missing":published is None,"text":text_value,
    "source_name":"Binance Public Announcements Page","source_tier":"official_primary",
    "source_type":"exchange_official","category":category,"sentiment_hint":score,
    "catalog_id":catalog.get("catalogId",catalog.get("id")),
    "catalog_name":category_name})
 return rows

def collect_binance_public_announcements(session=requests, limit=50, now=None, endpoint=None):
 """Fetch latest Binance announcements without credentials; endpoint is best-effort and undocumented."""
 if not 1 <= int(limit) <= 100:
  raise ValueError("limit must be between 1 and 100")
 url=endpoint or "https://www.binance.com/bapi/apex/v1/public/apex/cms/article/list/query"
 params={"type":"1","pageNo":"1","pageSize":str(int(limit))}
 headers={"User-Agent":"Mozilla/5.0 (compatible; PreBreakoutResearchBot/1.0)",
  "Accept":"application/json, text/plain, */*","Accept-Language":"en-US,en;q=0.9",
  "Referer":"https://www.binance.com/en/support/announcement"}
 response=session.get(url,params=params,timeout=20,headers=headers)
 response.raise_for_status()
 return parse_binance_public_response(response.json(),now or datetime.now(timezone.utc))

def build_binance_ws_url(api_secret, now_ms, random_value, topic="com_announcement_en", recv_window=30000):
 """Build the signed URL documented for Binance Announcements WebSocket."""
 import hashlib, hmac
 from urllib.parse import urlencode
 params={"random":str(random_value),"topic":topic,"recvWindow":str(recv_window),"timestamp":str(int(now_ms))}
 payload=urlencode(sorted(params.items()))
 signature=hmac.new(api_secret.encode("utf-8"),payload.encode("utf-8"),hashlib.sha256).hexdigest()
 return "wss://api.binance.com/sapi/wss?"+payload+"&signature="+signature

def collect_binance_announcements(api_key,api_secret,duration_seconds=60,now_fn=None,ws_factory=None):
 """Collect live official announcements for a bounded interval; requires a Binance API key."""
 import time, uuid
 try:
  import websocket
 except ImportError as exc:
  raise RuntimeError("Install websocket-client to collect Binance announcements") from exc
 factory=ws_factory or websocket.create_connection
 now_fn=now_fn or (lambda: int(time.time()*1000))
 url=build_binance_ws_url(api_secret,now_fn(),uuid.uuid4().hex[:24])
 ws=factory(url,header=[f"X-MBX-APIKEY: {api_key}"],timeout=1)
 rows=[]; started=time.monotonic(); last_ping=started
 try:
  while time.monotonic()-started < max(1,int(duration_seconds)):
   current=time.monotonic()
   if current-last_ping>=30:
    ws.ping(); last_ping=current
   try:
    raw=ws.recv()
   except websocket.WebSocketTimeoutException:
    continue
   if not raw: continue
   row=parse_binance_announcement_message(raw,datetime.now(timezone.utc))
   if row: rows.append(row)
 finally:
  try: ws.close()
  except Exception: pass
 return rows


def _supabase_request(url, key, method, table, *, params=None, rows=None, prefer=None, offset=None, limit=1000):
 headers={"apikey":key,"Authorization":f"Bearer {key}","Content-Type":"application/json"}
 if prefer: headers["Prefer"]=prefer
 if offset is not None: headers["Range"]=f"{offset}-{offset+limit-1}"
 response=requests.request(method,f"{url.rstrip('/')}/rest/v1/{table}",headers=headers,
  params=params,json=rows,timeout=60)
 if not response.ok:
  raise RuntimeError(f"Supabase {table} {method} failed: HTTP {response.status_code}: {response.text[:1000]}")
 return response.json() if response.text.strip() else []

def _load_supabase_events(url,key):
 events=[]; offset=0
 while True:
  page=_supabase_request(url,key,"GET","explosion_events",
   params={"select":"id,symbol,timeframe,event_time","order":"event_time.asc"},
   offset=offset,limit=1000)
  events.extend(page)
  if len(page)<1000: break
  offset+=1000
 return events

def upload_news_to_supabase(news, supabase_url=None, secret_key=None):
 """Persist timestamped news and point-in-time features linked to existing explosion_events."""
 url=(supabase_url or os.getenv("SUPABASE_URL","")).strip()
 key=(secret_key or os.getenv("SUPABASE_SECRET_KEY","")).strip()
 if not url or not key:
  raise RuntimeError("SUPABASE_URL and SUPABASE_SECRET_KEY are required for Supabase news upload")
 events=_load_supabase_events(url,key)
 symbols=sorted({str(e["symbol"]) for e in events if e.get("symbol")})
 aliases_path=os.getenv("NEWS_ALIASES_PATH","").strip()
 if not aliases_path and Path("config/news_symbol_aliases.json").exists():
  aliases_path="config/news_symbol_aliases.json"
 aliases=json.loads(Path(aliases_path).read_text(encoding="utf-8")) if aliases_path else {}
 prepared=[]
 for _,item in news.iterrows():
  text_value=str(item.get("text") or (str(item.get("title",""))+" "+str(item.get("summary","")))).strip()
  matched=match_symbols(text_value,symbols,aliases)
  raw_url=str(item.get("url") or "").strip()
  identity=raw_url.lower().rstrip("/") or "|".join([
   str(item.get("source_domain","")).lower(),str(item.get("title","")).lower(),
   str(item.get("published_utc","") or item.get("first_seen_utc",""))])
  fingerprint=hashlib.sha256(identity.encode("utf-8")).hexdigest()
  prepared.append({"fingerprint":fingerprint,"title":str(item.get("title","")),
   "summary":str(item.get("summary","") or ""),"url":raw_url,
   "source_name":str(item.get("source_name","") or ""),
   "source_domain":str(item.get("source_domain","") or ""),
   "source_tier":str(item.get("source_tier","unrated") or "unrated"),
   "source_type":str(item.get("source_type","media") or "media"),
   "category":str(item.get("category","general") or "general"),
   "sentiment_hint":int(item.get("sentiment_hint",0) or 0),
   "published_utc":item.get("published_utc") or None,
   "first_seen_utc":item.get("first_seen_utc"),
   "available_at_utc":item.get("available_at_utc") or item.get("first_seen_utc"),
   "matched_symbols":matched})
 # Preserve the original first-seen time on repeated collection runs.
 existing_by_fingerprint={}
 for start in range(0,len(prepared),250):
  batch=prepared[start:start+250]
  fingerprints=",".join(x["fingerprint"] for x in batch)
  existing=_supabase_request(url,key,"GET","news_events",
   params={"select":"id,fingerprint,first_seen_utc,available_at_utc,matched_symbols",
           "fingerprint":f"in.({fingerprints})"})
  for row in existing: existing_by_fingerprint[row["fingerprint"]]=row
 for row in prepared:
  previous=existing_by_fingerprint.get(row["fingerprint"])
  if previous:
   row["first_seen_utc"]=previous["first_seen_utc"]
   row["available_at_utc"]=previous["available_at_utc"]
   row["matched_symbols"]=sorted(set(row["matched_symbols"])|set(previous.get("matched_symbols") or []))
 for start in range(0,len(prepared),250):
  batch=prepared[start:start+250]
  _supabase_request(url,key,"POST","news_events",
   params={"on_conflict":"fingerprint"},rows=batch,
   prefer="resolution=merge-duplicates,return=minimal")
 # Re-read the full persisted event history: features must include earlier archived stories.
 persisted_news=[]; offset=0
 while True:
  page=_supabase_request(url,key,"GET","news_events",
   params={"select":"id,fingerprint,title,summary,source_domain,category,sentiment_hint,published_utc,first_seen_utc,available_at_utc,matched_symbols"},
   offset=offset,limit=1000)
  persisted_news.extend(page)
  if len(page)<1000: break
  offset+=1000
 # Recompute symbol matches so newly added event symbols can match previously archived stories.
 for row in persisted_news:
  text_value=(str(row.get("title") or "")+" "+str(row.get("summary") or "")).strip()
  matched=match_symbols(text_value,symbols,aliases)
  previous=sorted(set(row.get("matched_symbols") or []))
  row["matched_symbols"]=matched
  if matched!=previous:
   _supabase_request(url,key,"PATCH","news_events",
    params={"id":f"eq.{row['id']}"},rows={"matched_symbols":matched},
    prefer="return=minimal")
 # Compute each feature row against the actual persisted explosion-event timestamp.
 news_by_id=[(row,row["id"]) for row in persisted_news]
 feature_rows=[]
 for event in events:
  event_time=pd.to_datetime(event["event_time"],utc=True,errors="coerce")
  if pd.isna(event_time): continue
  eligible=[]
  for row,news_id in news_by_id:
   if event["symbol"] not in row["matched_symbols"]: continue
   available=pd.to_datetime(row.get("available_at_utc") or row.get("first_seen_utc"),utc=True,errors="coerce")
   if pd.isna(available) or available>=event_time or event_time-available>pd.Timedelta(days=7): continue
   published=pd.to_datetime(row.get("published_utc"),utc=True,errors="coerce")
   # Old articles first discovered in a recent feed snapshot are not fresh news.
   if not pd.isna(published) and (published>event_time or event_time-published>pd.Timedelta(days=7)): continue
   eligible.append((row,news_id,available))
  last24=[x for x in eligible if event_time-x[2]<=pd.Timedelta(hours=24)]
  cats=[x[0]["category"] for x in eligible]
  feature_rows.append({"explosion_event_id":event["id"],"symbol":event["symbol"],
   "timeframe":event.get("timeframe") or "1h","signal_time_utc":event_time.isoformat(),
   "news_count_24h":len(last24),"news_count_7d":len(eligible),
   "exchange_announcement_count_7d":sum("exchange_announcement" in c for c in cats),
   "token_unlock_count_7d":sum("token_unlock" in c for c in cats),
   "security_news_count_7d":sum("security" in c for c in cats),
   "positive_news_hint_count_7d":sum(x[0]["sentiment_hint"]>0 for x in eligible),
   "negative_news_hint_count_7d":sum(x[0]["sentiment_hint"]<0 for x in eligible),
   "matched_news_event_ids":[x[1] for x in eligible]})
 for start in range(0,len(feature_rows),250):
  _supabase_request(url,key,"POST","news_symbol_features",
   params={"on_conflict":"explosion_event_id"},rows=feature_rows[start:start+250],
   prefer="resolution=merge-duplicates,return=minimal")
 summary={"news_events_upserted":len(prepared),"persisted_news_events_read":len(persisted_news),"explosion_events_read":len(events),
  "news_symbol_features_upserted":len(feature_rows),
  "events_with_any_news":sum(r["news_count_7d"]>0 for r in feature_rows),
  "availability_rule":"available_at_utc < event_time AND (published_utc is null OR published_utc is within the prior 7 days and not in the future); published_utc is never treated as availability",
  "events_with_any_news":sum(r["news_count_7d"]>0 for r in feature_rows),
  "stale_published_articles_excluded":True}
 print(json.dumps({"supabase_news_upload":summary},indent=2))
 return summary

def run(feed_urls,candle_path,aliases_path=None,output_dir=OUT,extra_articles=None,extra_errors=None,upload_supabase=False):
 candles=pd.read_csv(candle_path); aliases=json.loads(Path(aliases_path).read_text()) if aliases_path else {}
 raw,errors=collect_feeds(feed_urls)
 raw.extend(extra_articles or [])
 errors.extend(extra_errors or [])
 for item in raw: item["category"],item["sentiment_hint"]=classify(item["text"])
 news=deduplicate_articles(pd.DataFrame(raw))
 if not news.empty:
  classified=news["text"].fillna("").map(classify)
  news["category"]=[x[0] for x in classified]
  news["sentiment_hint"]=[x[1] for x in classified]
 features=build_symbol_features(news,candles,aliases)
 output_dir.mkdir(parents=True,exist_ok=True)
 news.to_csv(output_dir/"news_events.csv",index=False)
 features.to_csv(output_dir/"news_symbol_daily_features.csv",index=False)
 supabase_summary=upload_news_to_supabase(news) if upload_supabase else None
 summary={"feeds_requested":len(feed_urls),"articles_collected":len(news),"symbol_feature_rows":len(features),"feed_errors":errors,
  "unique_stories_after_deduplication":len(news),
  "duplicate_source_copies_collapsed":int(news.get("duplicate_source_count",pd.Series(dtype=int)).sum()-len(news)) if not news.empty else 0,
  "availability_rule":"available_at_utc (fallback: first_seen_utc); article excluded when availability is equal to or later than signal time",
  "supabase_upload":supabase_summary,
  "limitations":["RSS coverage is source-dependent","ticker matching can miss aliases or create false matches",
   "keyword sentiment is a triage hint, not a verified model","historical first-seen times cannot be reconstructed from a current feed snapshot"]}
 (output_dir/"news_monitor_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
 return summary

def main():
 p=argparse.ArgumentParser(); p.add_argument("--candles",default="reports/surge_scan/daily_candles.csv")
 p.add_argument("--aliases",default=""); p.add_argument("--output-dir",default=str(OUT))
 p.add_argument("--feeds",default=os.getenv("NEWS_FEED_URLS","")); p.add_argument("--feed-config",default=""); p.add_argument("--binance-public-limit",type=int,default=50); p.add_argument("--binance-ws-seconds",type=int,default=0)
 p.add_argument("--candle-days",type=int,default=30); p.add_argument("--upload-supabase",action="store_true"); a=p.parse_args()
 if not a.aliases and Path("config/news_symbol_aliases.json").exists():
  a.aliases="config/news_symbol_aliases.json"
 feeds=[x.strip() for x in a.feeds.split(",") if x.strip()]
 if a.feed_config:
  configured=load_feed_config(a.feed_config)
  feeds=configured+[{"url":url} for url in feeds]
 if not feeds: p.error("Provide --feeds, --feed-config, or NEWS_FEED_URLS")
 public_articles=[]; public_errors=[]
 try: public_articles=collect_binance_public_announcements(limit=a.binance_public_limit)
 except Exception as exc: public_errors.append({"feed":"Binance public announcements","source_name":"Binance Public Announcements Page","error":str(exc)})
 extra_articles=list(public_articles)
 if a.binance_ws_seconds>0:
  key=os.getenv("BINANCE_API_KEY",""); secret=os.getenv("BINANCE_API_SECRET","")
  if key and secret:
   try: extra_articles.extend(collect_binance_announcements(key,secret,a.binance_ws_seconds))
   except Exception as exc: public_errors.append({"feed":"Binance announcements WebSocket","source_name":"Binance Announcements WebSocket","error":str(exc)})
  else: public_errors.append({"feed":"Binance announcements WebSocket","source_name":"Binance Announcements WebSocket","error":"Skipped: API credentials are not configured"})
 candle_path=Path(a.candles)
 candles=pd.read_csv(candle_path) if candle_path.exists() else pd.DataFrame()
 if candles.empty:
  from src.news_candle_collector import collect_recent_daily_candles
  print("Daily candles missing/empty; collecting recent Binance Spot daily candles...")
  collect_recent_daily_candles(candle_path,days=a.candle_days)
  candles=pd.read_csv(candle_path)
 if candles.empty or not {"symbol","open_time"}.issubset(candles.columns):
  raise RuntimeError(f"Daily candle input is empty or invalid: {candle_path}")
 print(json.dumps(run(feeds,candle_path,a.aliases or None,Path(a.output_dir),extra_articles,public_errors,a.upload_supabase),indent=2))
if __name__=="__main__": main()
