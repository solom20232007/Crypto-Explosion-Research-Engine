"""Weekly live-performance report of the daily recommendations (markdown to stdout). Research only."""
import os, requests, pandas as pd
URL = os.environ["SUPABASE_URL"].rstrip("/"); KEY = os.environ["SUPABASE_SECRET_KEY"]
H = {"apikey": KEY, "Authorization": f"Bearer {KEY}"}
def get(t, p): r = requests.get(f"{URL}/rest/v1/{t}", headers=H, params=p, timeout=60); r.raise_for_status(); return r.json()
perf = pd.DataFrame(get("recommendation_performance_v1", {"select": "*"}))
hist = get("daily_recommendations_v1", {"select": "bar_time", "order": "bar_time.asc", "limit": 1})
print("# تقرير الأداء الفعلي للتوصيات (أسبوعي)\n")
print(f"أول توصية مسجلة: {hist[0]['bar_time'][:10] if hist else '-'}\n")
if perf.empty or int(perf.days.max()) < 1:
    print("لا توجد أيام مكتملة النتائج بعد (النتيجة تحتاج 72 ساعة بعد كل شمعة). لا تستنتج شيئًا بعد.")
else:
    n = int(perf.days.max())
    print(f"الأيام المكتملة: {n}\n")
    cols = [c for c in ["top_k", "days", "recs", "hit_10_24", "universe_base_10_24", "lift_10_24", "hit_25_48", "universe_base_25_48", "hit_40_72", "universe_base_40_72", "avg_best_24h_gain", "avg_worst_24h_drop"] if c in perf.columns]
    print(perf[cols].round(3).to_markdown(index=False) if hasattr(perf, "to_markdown") else perf[cols].round(3).to_string(index=False))
    print("\nالمرجع التاريخي (3 سنوات): أفضل 5 يوميًا ≈ 22–26% لهدف +10%/24س مقابل ≈ 9–11% لأي عملة (2.2–2.9 ضعف).")
    if n < 30: print(f"\n⚠️ العينة صغيرة ({n} يومًا): الفروق الحالية ضجيج على الأرجح. الحد الأدنى للحكم ≈ 60 يومًا.")
