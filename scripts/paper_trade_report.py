"""Paper-trading report (markdown to stdout). Research only; no orders are placed.

Fixed rules, decided in advance and never tuned on live results:
  * picks = the day's top-5 recommendations (rank<=5, not already_moved);
  * entry = close of the 08:00 UTC hourly bar of the recommendation day (about when the user sees it);
  * exit  = +3.5% target (an hourly high touching entry*1.035; fill assumed at the target) or the close 24h after entry. No stop.
  * costs = 0.10% fee per side + slippage per side by prior-24h quote volume (<200k: 0.5%, 200k-1M: 0.3%, >=1M: 0.15%).
Only days whose full 24h window is in the candles table are counted.
"""
import os, requests, numpy as np, pandas as pd
URL = os.environ["SUPABASE_URL"].rstrip("/"); KEY = os.environ["SUPABASE_SECRET_KEY"]; H = {"apikey": KEY, "Authorization": f"Bearer {KEY}"}
TP, FEE = 0.035, 0.001

def get(t, p):
    rows, off = [], 0
    while True:
        r = requests.get(f"{URL}/rest/v1/{t}", headers={**H, "Range": f"{off}-{off+999}"}, params=p, timeout=60); r.raise_for_status(); b = r.json(); rows += b
        if len(b) < 1000: return rows
        off += 1000
def slip(q): return 0.005 if q < 2e5 else 0.003 if q < 1e6 else 0.0015

recs = pd.DataFrame(get("daily_recommendations_v1", {"select": "bar_time,symbol,rank,already_moved,market_wide", "rank": "lte.5", "order": "bar_time,rank"}))
recs = recs[~recs.already_moved.astype(bool)] if len(recs) else recs
out = []
for _, r in recs.iterrows():
    t0 = pd.Timestamp(r.bar_time).tz_convert("UTC") if pd.Timestamp(r.bar_time).tzinfo else pd.Timestamp(r.bar_time, tz="UTC")
    c = pd.DataFrame(get("candles", {"select": "open_time,high,close,quote_volume", "symbol": f"eq.{r.symbol}", "timeframe": "eq.1h",
                                     "open_time": f"gte.{(t0 - pd.Timedelta(hours=16)).isoformat()}", "and": f"(open_time.lte.{(t0 + pd.Timedelta(hours=33)).isoformat()})", "order": "open_time"}))
    if c.empty: continue
    c["open_time"] = pd.to_datetime(c.open_time, utc=True); c = c.set_index("open_time"); e_open = t0 + pd.Timedelta(hours=8)
    if e_open not in c.index or (e_open + pd.Timedelta(hours=24)) not in c.index: continue
    w = c.loc[e_open + pd.Timedelta(hours=1): e_open + pd.Timedelta(hours=24)]
    if len(w) < 24: continue
    entry = float(c.loc[e_open, "close"]); hit = (w.high.astype(float) >= entry * (1 + TP)).any()
    gross = TP if hit else float(w.close.iloc[-1]) / entry - 1
    q = float(c.loc[e_open - pd.Timedelta(hours=23): e_open, "quote_volume"].astype(float).sum()); cost = 2 * (FEE + slip(q))
    out.append({"day": t0.date(), "symbol": r.symbol, "rank": r["rank"], "hit": hit, "gross": gross, "net": gross - cost})
print("# تداول ورقي (محاكاة فقط، لا أوامر حقيقية)\n")
print("القواعد ثابتة: أفضل 5 يوميًا، دخول إغلاق شمعة 08:00 UTC، هدف +3.5% أو إغلاق بعد 24 ساعة، بلا وقف، بعد الرسوم والانزلاق.\n")
if not out: print("لا توجد أيام مكتملة بعد."); raise SystemExit
d = pd.DataFrame(out); day = d.groupby("day").net.mean()
print(f"أيام مكتملة: {len(day)} | صفقات: {len(d)}")
print(f"بلغت الهدف: {d.hit.mean():.1%} | متوسط صافي الصفقة: {d.net.mean()*100:.2f}% | وسيط: {d.net.median()*100:.2f}%")
print(f"أيام رابحة: {(day>0).mean():.1%} | متوسط الصافي اليومي: {day.mean()*100:.2f}% | أسوأ يوم: {day.min()*100:.1f}% | أفضل يوم: {day.max()*100:.1f}%")
print(f"مجموع الصافي التراكمي (جمع بسيط): {day.sum()*100:.1f}%")
print("\nالمرجع التاريخي (اختبار 3 سنوات): متوسط صافي الصفقة بهذا النمط ≈ −0.5% (لا ميزة عن الشراء العشوائي).")
if len(day) < 30: print(f"\n⚠️ العينة صغيرة ({len(day)} يومًا): لا تستنتج شيئًا قبل 30–60 يومًا.")
print("\nآخر 10 أيام:\n"); print(day.tail(10).mul(100).round(2).rename("صافي اليوم %").to_markdown())
