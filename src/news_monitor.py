"""Timestamp-safe RSS/Atom news ingestion for pre-breakout research."""
from __future__ import annotations
import argparse, json, os, re
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

def parse_feed(xml_text,source_url,first_seen):
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
   "available_at_utc":utc_iso(first_seen),"published_time_missing":published is None,"text":(title+" "+summary).strip()})
 return rows

def classify(text):
 low=text.lower()
 cats=[name for name,terms in CATEGORY_TERMS.items() if any(t in low for t in terms)]
 score=int(any(t in low for t in POSITIVE))-int(any(t in low for t in NEGATIVE))
 return "|".join(cats) if cats else "general",score

def match_symbols(text,symbols,aliases=None):
 aliases=aliases or {}; upper=text.upper(); found=[]
 for symbol in symbols:
  base=symbol[:-4] if symbol.endswith("USDT") else symbol
  for term in {base,*[str(x).upper() for x in aliases.get(base,[])]}:
   if len(term)>=3 and re.search(r"(?<![A-Z0-9])\$?"+re.escape(term)+r"(?![A-Z0-9])",upper):
    found.append(symbol); break
 return found

def collect_feeds(feed_urls,session=requests,now=None):
 seen=now or datetime.now(timezone.utc); rows=[]; errors=[]
 for url in feed_urls:
  try:
   response=session.get(url,timeout=20,headers={"User-Agent":"PreBreakoutResearchBot/1.0"})
   response.raise_for_status(); rows.extend(parse_feed(response.text,url,seen))
  except Exception as exc: errors.append({"feed":url,"error":str(exc)})
 return rows,errors

def build_symbol_features(news,candles,aliases=None):
 missing={"symbol","open_time"}-set(candles.columns)
 if missing: raise ValueError(f"candles missing columns: {sorted(missing)}")
 n=news.copy()
 if n.empty: n=pd.DataFrame(columns=["text","first_seen_utc","category","sentiment_hint"])
 n["first_seen"]=pd.to_datetime(n.get("first_seen_utc",pd.Series(dtype=str)),utc=True,errors="coerce")
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
  subset=m[(m.symbol==row.symbol)&(m.first_seen<signal)] if not m.empty else m
  r24=subset[subset.first_seen>=signal-pd.Timedelta(hours=24)]
  r7=subset[subset.first_seen>=signal-pd.Timedelta(days=7)]
  cats=r7.category.astype(str)
  results.append({"symbol":row.symbol,"signal_time_utc":signal.isoformat().replace("+00:00","Z"),
   "news_count_24h":len(r24),"news_count_7d":len(r7),
   "exchange_announcement_count_7d":cats.str.contains("exchange_announcement").sum(),
   "token_unlock_count_7d":cats.str.contains("token_unlock").sum(),
   "security_news_count_7d":cats.str.contains("security").sum(),
   "positive_news_hint_count_7d":(r7.sentiment_hint>0).sum(),
   "negative_news_hint_count_7d":(r7.sentiment_hint<0).sum()})
 return pd.DataFrame(results)

def run(feed_urls,candle_path,aliases_path=None,output_dir=OUT):
 candles=pd.read_csv(candle_path); aliases=json.loads(Path(aliases_path).read_text()) if aliases_path else {}
 raw,errors=collect_feeds(feed_urls)
 for item in raw: item["category"],item["sentiment_hint"]=classify(item["text"])
 news=pd.DataFrame(raw)
 if not news.empty: news=news.drop_duplicates(subset=["url","title","source_domain"])
 features=build_symbol_features(news,candles,aliases)
 output_dir.mkdir(parents=True,exist_ok=True)
 news.to_csv(output_dir/"news_events.csv",index=False)
 features.to_csv(output_dir/"news_symbol_daily_features.csv",index=False)
 summary={"feeds_requested":len(feed_urls),"articles_collected":len(news),"symbol_feature_rows":len(features),"feed_errors":errors,
  "availability_rule":"first_seen_utc; article excluded before first_seen_utc",
  "limitations":["RSS coverage is source-dependent","ticker matching can miss aliases or create false matches",
   "keyword sentiment is a triage hint, not a verified model","historical first-seen times cannot be reconstructed from a current feed snapshot"]}
 (output_dir/"news_monitor_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
 return summary

def main():
 p=argparse.ArgumentParser(); p.add_argument("--candles",default="reports/surge_scan/daily_candles.csv")
 p.add_argument("--aliases",default=""); p.add_argument("--output-dir",default=str(OUT))
 p.add_argument("--feeds",default=os.getenv("NEWS_FEED_URLS","")); a=p.parse_args()
 feeds=[x.strip() for x in a.feeds.split(",") if x.strip()]
 if not feeds: p.error("Provide --feeds or NEWS_FEED_URLS (comma-separated RSS/Atom URLs)")
 print(json.dumps(run(feeds,a.candles,a.aliases or None,Path(a.output_dir)),indent=2))
if __name__=="__main__": main()
