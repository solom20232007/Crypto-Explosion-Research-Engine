"""Historical surge discovery for small Binance Spot USDT altcoins."""
import argparse, json, time
from pathlib import Path
import pandas as pd
import requests
BASES=("https://data-api.binance.vision","https://api.binance.com")
EXCLUDED={"BTC","ETH","SOL","USDT","USDC","FDUSD","TUSD","USDP","DAI","BUSD","USDE","USDD","EUR","TRY","BRL","GBP","AUD","BIDR","IDRT","UAH","RUB","ZAR","PLN","RON","ARS","JPY"}
COLS=["open_time","open","high","low","close","volume","close_time","quote_volume","trade_count","taker_buy_base_volume","taker_buy_quote_volume","ignore"]
def get_json(path,params=None):
    last=None
    for base in BASES:
        for attempt in range(4):
            try:
                r=requests.get(base+path,params=params,timeout=30)
                if r.status_code==451: last=requests.HTTPError("HTTP 451",response=r); break
                if r.status_code in (418,429): time.sleep(max(int(r.headers.get("Retry-After",2**attempt)),1)); continue
                r.raise_for_status(); return r.json()
            except (requests.RequestException,ValueError) as e:
                last=e
                if attempt<3: time.sleep(2**attempt)
    raise last or RuntimeError("Binance endpoints unavailable")
def discover_target_symbols(info):
    out=[]
    for x in info.get("symbols",[]):
        base=str(x.get("baseAsset","")).upper(); sym=str(x.get("symbol","")).upper()
        if x.get("status")!="TRADING" or x.get("quoteAsset")!="USDT" or not x.get("isSpotTradingAllowed",True): continue
        if base in EXCLUDED or base.endswith(("UP","DOWN","BULL","BEAR")) or any(t in base for t in ("2L","2S","3L","3S","5L","5S")): continue
        if sym.endswith("USDT"): out.append(sym)
    return sorted(set(out))
def fetch_klines(symbol,interval,start_ms,end_ms,pause=.08):
    rows=[]; cursor=start_ms; step=3600000 if interval=="1h" else 86400000
    while cursor<end_ms:
        p=get_json("/api/v3/klines",{"symbol":symbol,"interval":interval,"startTime":cursor,"endTime":end_ms-1,"limit":1000})
        if not p: break
        rows.extend(p); nxt=int(p[-1][0])+step
        if nxt<=cursor: break
        cursor=nxt
        if len(p)<1000: break
        time.sleep(max(0,pause))
    if not rows: return pd.DataFrame(columns=COLS)
    f=pd.DataFrame(rows,columns=COLS).drop_duplicates("open_time").sort_values("open_time")
    for c in ("open_time","open","high","low","close","volume","quote_volume","trade_count"): f[c]=pd.to_numeric(f[c],errors="coerce")
    return f.dropna(subset=["open_time","close","high"])
def detect_surge_events(daily,symbol,daily_high_pct=25,daily_close_pct=20,three_day_pct=40,seven_day_pct=75):
    if daily.empty: return pd.DataFrame()
    f=daily.copy().sort_values("open_time").reset_index(drop=True)
    for c in ("open","high","low","close","volume","quote_volume"): f[c]=pd.to_numeric(f[c],errors="coerce")
    prev=f.close.shift(1); f["daily_close_change_pct"]=(f.close/prev-1)*100; f["daily_high_change_pct"]=(f.high/prev-1)*100
    f["close_change_3d_pct"]=(f.close/f.close.shift(3)-1)*100; f["close_change_7d_pct"]=(f.close/f.close.shift(7)-1)*100
    f["volume_change_1d_pct"]=(f.volume/f.volume.shift(1)-1)*100
    tests={"daily_high":f.daily_high_change_pct>=daily_high_pct,"daily_close":f.daily_close_change_pct>=daily_close_pct,
           "three_day_close":f.close_change_3d_pct>=three_day_pct,"seven_day_close":f.close_change_7d_pct>=seven_day_pct}
    mask=pd.Series(False,index=f.index)
    for v in tests.values(): mask|=v
    r=f.loc[mask].copy()
    if r.empty: return pd.DataFrame()
    r["symbol"]=symbol; r["event_date_utc"]=pd.to_datetime(r.open_time,unit="ms",utc=True).dt.strftime("%Y-%m-%d")
    r["trigger_types"]=[",".join(k for k,v in tests.items() if bool(v.loc[i])) for i in r.index]
    for c in ("daily_close_change_pct","daily_high_change_pct","close_change_3d_pct","close_change_7d_pct","volume_change_1d_pct"): r[c]=r[c].round(4)
    return r[["symbol","event_date_utc","open_time","open","high","low","close","volume","quote_volume","daily_high_change_pct","daily_close_change_pct","close_change_3d_pct","close_change_7d_pct","volume_change_1d_pct","trigger_types"]].reset_index(drop=True)
def summarize_context(hourly,symbol,date):
    if hourly.empty: return {"symbol":symbol,"event_date_utc":date,"hourly_rows":0}
    f=hourly.copy()
    for c in ("open_time","close","high","low","quote_volume"): f[c]=pd.to_numeric(f[c],errors="coerce")
    t=int(pd.Timestamp(date,tz="UTC").timestamp()*1000); pre=f[f.open_time<t].sort_values("open_time"); last=pre.tail(24); prior=pre.tail(72).head(48)
    a=last.quote_volume.mean() if len(last) else float("nan"); b=prior.quote_volume.mean() if len(prior) else float("nan")
    return {"symbol":symbol,"event_date_utc":date,"hourly_rows":len(f),
      "pre_event_24h_return_pct":round((last.close.iloc[-1]/last.close.iloc[0]-1)*100,4) if len(last)>1 else None,
      "pre_event_72h_return_pct":round((pre.close.iloc[-1]/pre.close.iloc[-min(72,len(pre))]-1)*100,4) if len(pre)>1 else None,
      "pre_event_24h_range_pct":round((last.high.max()/last.low.min()-1)*100,4) if len(last) and last.low.min()>0 else None,
      "pre_event_quote_volume_24h":round(float(last.quote_volume.sum()),4) if len(last) else None,
      "pre_event_volume_ratio_24h_vs_prior48h":round(float(a/b),4) if pd.notna(a) and pd.notna(b) and b>0 else None}
def run_scan(start,end,out_dir,max_symbols=None,max_events=250,min_quote_volume_usdt=100000,pause_seconds=.08):
    start=pd.Timestamp(start); end=pd.Timestamp(end)
    start=start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end=end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    if end<=start: raise ValueError("--end must be later than --start")
    out_dir.mkdir(parents=True,exist_ok=True); symbols=discover_target_symbols(get_json("/api/v3/exchangeInfo"))
    if max_symbols is not None:
        if max_symbols<1: raise ValueError("--max-symbols must be positive")
        symbols=symbols[:max_symbols]
    a,b=int(start.timestamp()*1000),int(end.timestamp()*1000); events=[]; failures=[]
    for i,sym in enumerate(symbols,1):
        try:
            d=fetch_klines(sym,"1d",a,b,pause_seconds); e=detect_surge_events(d,sym)
            if not e.empty: e=e[e.quote_volume>=min_quote_volume_usdt]; events.extend(e.to_dict("records"))
            print(json.dumps({"progress":f"{i}/{len(symbols)}","symbol":sym,"daily_rows":len(d),"events":len(e),"status":"ok"}))
        except Exception as exc:
            failures.append({"symbol":sym,"error":f"{type(exc).__name__}: {exc}"}); print(json.dumps({"symbol":sym,"status":"error","error":str(exc)}))
        time.sleep(max(0,pause_seconds))
    ev=pd.DataFrame(events)
    if not ev.empty: ev=ev.sort_values(["daily_high_change_pct","daily_close_change_pct","close_change_3d_pct"],ascending=False)
    ep=out_dir/"surge_events.csv"; ev.to_csv(ep,index=False); selected=ev.head(max(1,max_events)) if not ev.empty else ev
    hourly_parts=[]; contexts=[]; hfail=[]
    for i,x in enumerate(selected.to_dict("records"),1):
        sym,date=x["symbol"],x["event_date_utc"]; t=int(pd.Timestamp(date,tz="UTC").timestamp()*1000)
        try:
            h=fetch_klines(sym,"1h",t-72*3600000,t+24*3600000,pause_seconds)
            if not h.empty: h.insert(0,"event_date_utc",date); h.insert(0,"event_symbol",sym); hourly_parts.append(h)
            contexts.append(summarize_context(h,sym,date))
        except Exception as exc: hfail.append({"symbol":sym,"event_date_utc":date,"error":str(exc)})
        if i%25==0: print(json.dumps({"hourly_context_progress":f"{i}/{len(selected)}"}))
    hp=out_dir/"surge_event_hourly_context.csv"; (pd.concat(hourly_parts,ignore_index=True) if hourly_parts else pd.DataFrame()).to_csv(hp,index=False)
    context=pd.DataFrame(contexts)
    if not context.empty:
        cols=["symbol","event_date_utc","daily_high_change_pct","daily_close_change_pct","close_change_3d_pct","close_change_7d_pct","trigger_types"]
        context=selected[cols].merge(context,on=["symbol","event_date_utc"],how="left")
    cp=out_dir/"surge_pre_event_context.csv"; context.to_csv(cp,index=False)
    summary={"requested_start_utc":start.isoformat(),"requested_end_exclusive_utc":end.isoformat(),
      "universe":"currently active Binance Spot USDT pairs excluding BTC/ETH/SOL, stablecoins and leveraged tokens",
      "survivorship_bias_warning":"Current exchangeInfo omits delisted pairs; this first pass may miss historical winners that were later delisted.",
      "symbols_scanned":len(symbols),"symbols_with_errors":len(failures),"event_rows":len(ev),
      "selected_events_with_hourly_context":len(selected),"minimum_event_quote_volume_usdt":min_quote_volume_usdt,
      "thresholds_pct":{"daily_high":25,"daily_close":20,"close_3d":40,"close_7d":75},
      "failed_symbols":failures,"hourly_context_failures":hfail,"outputs":[str(ep),str(hp),str(cp)]}
    (out_dir/"surge_scan_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2)); return summary
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--start",default="2023-10-28T00:00:00Z"); p.add_argument("--end",default="2026-10-28T00:00:00Z")
    p.add_argument("--out-dir",type=Path,default=Path("reports/surge_scan")); p.add_argument("--max-symbols",type=int,default=None)
    p.add_argument("--max-events",type=int,default=250); p.add_argument("--min-quote-volume-usdt",type=float,default=100000); p.add_argument("--pause-seconds",type=float,default=.08)
    x=p.parse_args(); run_scan(x.start,x.end,x.out_dir,x.max_symbols,x.max_events,x.min_quote_volume_usdt,x.pause_seconds)
if __name__=="__main__": main()
