# Supabase-backed point-in-time news features

The news monitor writes collected articles into the existing Supabase `public.news_events` table and computes one feature row per existing `public.explosion_events` row in `public.news_symbol_features`.

## Required GitHub Actions secrets

- `SUPABASE_URL`
- `SUPABASE_SECRET_KEY` (server-side secret/service-role key; never expose it in browser code)

## Run

```bash
python -m src.news_monitor \
  --feed-config config/news_feeds.json \
  --candles reports/surge_scan/daily_candles.csv \
  --upload-supabase
```

The daily CSV artifact remains available for diagnostics. Supabase is the durable store and the source of event timestamps.

## Point-in-time rule

For each row in `explosion_events`, a news story is eligible only when:

```text
news_events.available_at_utc < explosion_events.event_time
```

Equality is excluded. `published_utc` is retained as publisher metadata and is never substituted for availability. Repeated collection preserves the original `first_seen_utc` and `available_at_utc` values.

The uploader:
1. Upserts news into `news_events` by stable fingerprint.
2. Matches article title/summary to symbols in `explosion_events`.
3. Reads the persisted news archive, including previously collected articles.
4. Computes 24-hour and 7-day counts and category/sentiment hints per event.
5. Upserts the result by `explosion_event_id` in `news_symbol_features`, including matched news UUIDs.

## Historical limitation

The current Supabase database has a finite set of historical explosion events. News first seen after an event cannot contribute to that event, even if the publisher's displayed publication date is earlier. To create useful nonzero features for future events, the news collector must run before those events occur and preserve its first-seen timestamps.
