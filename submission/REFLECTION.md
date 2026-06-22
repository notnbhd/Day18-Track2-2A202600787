# Reflection — Lakehouse Anti-Pattern Risk Assessment

**Project:** Modeling Influenza Spread Across the Globe  
**Team:** Group 4 — Đặng Tuấn Phong · Trần Minh Hiếu · Nguyễn Bạch Hải Đăng · Nguyễn Thế Khiêm  
**Date:** 2026-06-22

---

## The Anti-Pattern We Are Most at Risk Of

**Anti-pattern #4: `VACUUM 0 HOURS` để "tiết kiệm storage"**

Out of the five anti-patterns on the slide, this is the one that would most naturally occur to our team — and the one with the most catastrophic consequence relative to the damage it appears to cause.

---

## Why This One, Not the Others

It is worth being explicit about why the other four are less dangerous for us before making the case for #4.

**#1 (Dump everything to S3, no schema)** is our current state — we have no Delta, no schema enforcement, raw parquet files with versioned filenames. This is a real problem and the architecture document addresses it. But it is also a visible problem: you can see the swamp accumulating. It degrades gradually.

**#2 (High-cardinality partitioning)** is a latent risk. Our proposed partition key `(iso_year, country_iso3)` gives 189 × 25 = 4,725 partitions — borderline but manageable. The dangerous version would be adding `iso_week` as a third level (189 × 52 × 25 = 245,700 partitions), which is tempting because our query pattern is week-level. The ARCHITECTURE.md explicitly guards against this, so it is a known risk with a documented mitigation.

**#3 (Skipping OPTIMIZE → small-file problem)** is real for the weekly FluNet ingestion path. Our Module 1 already produces 75 parquet files across 25 years (3 per year), which is fine. But a weekly-cadence pipeline — 52 appends per year × 5 tables = 260 new files/year — would hit the small-file threshold within two years without a compaction schedule. This is operationally annoying but not silent: queries become measurably slow and the symptom is obvious.

**#5 (Spark cluster for 5 GB query)** is our current setup. We built the ERA5 pipeline on Spark because 130 GB of NetCDF genuinely requires it. The risk is that we reach for the same Spark environment when querying the Silver or Gold tables, which are 5–50 MB. This wastes money but does not destroy data.

**#4 is different from all of these.** The failure is silent, the motivation is rational, and the loss is irreversible.

---

## The Mechanism

Our FinOps analysis found that ERA5 raw NetCDF storage ($2.88/month) was 95% of total storage cost. The correct response — decision D8 in the architecture — was to discard the raw `.nc` files after a validated Parquet conversion. This was the right call: 125 GB of read-once source data that ECMWF will re-serve for free is genuinely wasteful to store.

The danger is that this mental model carries over to the Delta transaction log.

A team member who has just internalized "large source files we don't query should be discarded" will look at the Delta `_delta_log/` directory — a few hundred KB of JSON commit files — and apply the same logic. `VACUUM RETAIN 0 HOURS` frees that space immediately. The table still reads. Current queries still work. Nothing appears broken.

What is gone is the entire time-travel history.

For most projects, losing time travel is an inconvenience. For ours, it is the core reproducibility guarantee we built the whole architecture around.

---

## Why This Would Be Uniquely Damaging for Our Project

Our models were compared on a shared temporal split: train < 2018, val 2018–2019, test ≥ 2020. The leaderboard result — Ridge R² 0.802 vs GNN R² 0.263 — is only meaningful if we can prove all five models were scored on the same corpus. Currently we cannot prove this: the parquet files have no version history. The entire motivation for Delta time travel in our architecture is to make that proof possible for every future training run.

The specific scenarios where we need time travel:

1. **Reproducing a past model.** Six months from now, someone asks: "What exact training data produced `flu_model.pkl` version 1?" Without the Delta snapshot pointer in MLflow and the corresponding transaction log, the answer is "we don't know."

2. **The COVID regime-shift rollback (FM1).** When the GNN is retrained on new WHO data and its RMSE spikes, the recovery plan is to roll back `mobility_edges/` to the pre-collapse airline topology. `VACUUM 0 HOURS` makes this impossible — the old snapshot no longer exists.

3. **Late WHO reporting (FM2).** When week-48 data is incomplete, the recovery plan is to re-run Module 4 against the last complete Silver snapshot. Same problem.

In all three cases, the `VACUUM` command does not feel like destroying data. The Delta log files are small, the storage savings are real ($0.001/month at best), and the operation succeeds silently. The loss only surfaces when you try to use time travel — potentially months after the damage was done, after many subsequent commits have overwritten the affected partitions.

---

## Why the Temptation Is Rational, Not Careless

The reason this anti-pattern is worth writing about is that it does not arise from negligence. It arises from the same discipline — FinOps awareness, storage hygiene — that produces correct decisions elsewhere.

Our team made a correct call discarding 125 GB of ERA5 raw NetCDF. A team that thinks carefully about storage costs is also a team that will notice Delta log directories and ask whether they need to be retained. The answer is yes — not because the files are large, but because they are the audit trail for an irreversible processing pipeline. The distinction is:

- ERA5 `.nc` files: large, read-once, re-downloadable from ECMWF, zero query value after Bronze → discard.
- Delta transaction log: tiny, write-once, not re-creatable, the only record of which data version produced which model → retain permanently.

The fix the slide prescribes — retain at minimum 168 hours (7 days, the default) — is actually too conservative for a weekly training pipeline. We should retain at minimum 104 versions (2 years), as specified in the architecture document, because our training window spans multi-year history and model audits can be requested long after a training run completes.

---

## What We Would Actually Write in the Code

```python
# Correct: VACUUM with retention that preserves our training audit window
delta_table.vacuum(retention_hours=24 * 7 * 14)  # 14 weeks minimum

# What we are at risk of writing after reading the ERA5 FinOps analysis:
delta_table.vacuum(retention_hours=0)  # DO NOT DO THIS — destroys time travel
```

The guard that prevents this is not documentation — it is a CI assertion:

```python
# Assert no Delta table has retention < 168h before any pipeline run
for table_path in SILVER_TABLES + GOLD_TABLES:
    dt = DeltaTable(table_path)
    props = dt.protocol()
    retention = dt.history(1)[0].get("operationParameters", {}).get("retentionCheckEnabled")
    assert retention != "false", f"{table_path}: time travel retention check is disabled"
```

---

## Summary

Anti-pattern #4 is our highest risk because it is the only one where:
- The motivation is a direct extension of a correct decision we already made (D8: discard ERA5 raw files)
- The immediate symptom is success (storage freed, table still works)
- The delayed symptom is the loss of our primary reproducibility guarantee
- The loss is irreversible — no rollback is possible once the transaction log is vacuumed

The other four anti-patterns degrade performance or accumulate technical debt visibly. This one silently destroys the audit trail that makes our model comparison trustworthy.
