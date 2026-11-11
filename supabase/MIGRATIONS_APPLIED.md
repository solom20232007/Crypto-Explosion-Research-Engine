# Migrations applied in Supabase (project crypto-research-engine)

Only two of these exist as SQL files in `supabase/migrations/`. The rest were applied through the Supabase migration tool; their SQL is kept in Supabase's migration history (`supabase_migrations.schema_migrations`) and is **not yet exported here**, so the database cannot be rebuilt from this repo alone. Exporting them is a tracked TODO.

| version | name |
|---|---|
| 20261030205127 | add_explosion_episode_model |
| 20261030205142 | tune_episode_indexes |
| 20261030205200 | add_explosion_episode_refresh |
| 20261030211033 | add_point_in_time_features_and_forward_labels |
| 20261030211423 | restrict_research_feature_tables_to_service_role |
| 20261030212010 | create_research_training_views |
| 20261030212046 | enforce_forward_label_integrity_and_backfill_events |
| 20261030212525 | add_scanner_signal_tracking |
| 20261030222922 | add_onchain_whale_research_tables |
| 20261105191802 | onchain_collection_checkpoints |
| 20261106225935 | create_market_context_and_episodes_v1 |
| 20261106230059 | create_prebreakout_samples_v1_batch0 |
| 20261106230420 | finalize_prebreakout_samples_v1 |
| 20261106231424 | add_symbol_relative_features_to_samples_v1 |
| 20261107090331 | create_daily_recommendations_v1 |
| 20261107092041 | add_raw_score_to_daily_recommendations |
| 20261107093357 | create_news_symbol_daily_v1 |
| 20261107093815 | create_token_contracts_v1 |
| 20261107100737 | create_onchain_pilot_tables_v1 |
| 20261107112130 | recommendation_outcome_tracking_v1 |
| 20261107113455 | create_futures_daily_v1 |
| 20261107131729 | create_api_schema_views_v1 |
| 20261107131759 | api_views_hide_stale_already_moved_rows |
| 20261107132054 | lock_down_legacy_objects_and_create_research_reader |
| 20261107134305 | pipeline_health_monitoring_v1 |
| 20261107141026 | pipeline_health_thresholds_for_throttled_schedules |
| 20261107143004 | create_v6_feedback_tables_v1 |
| 20261107143016 | api_v6_comparison_views_v1 |
| 20261107145511 | v6_shariah_list_and_unified_eval |
| 20261107161425 | history_daily_v1 |
| 20261110223354 | onchain_hist_sample_v1 |
| 20261111212905 | health_onchain_collection_stopped |
