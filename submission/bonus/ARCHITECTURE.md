# Architecture Decision Record — Global Influenza Surveillance Lakehouse

**Topic:** H (self-defined) — Storage architecture for a multi-source, multi-model influenza forecasting system  
**Author:** A Dang  
**Date:** 2026-06-22

---

## 1. Problem Statement

This system ingests three heterogeneous sources to produce two-week-ahead national influenza forecasts for 119 countries:

| Source | Raw size | Cadence | Format |
|---|---|---|---|
| WHO FluNet surveillance | 30 MB CSV, 181,195 rows, 189 countries, 1995–2026 | Weekly pull | CSV (53 columns) |
| **ERA5 climate reanalysis (ECMWF)** | **125 GB NetCDF, 2001–2025, global 0.25° grid** | **Yearly download** | **.nc (gridded arrays: t2m, d2m, tp)** |
| IATA airline mobility | ~50 MB, route counts per country pair | Yearly snapshot | CSV |

Module 1 runs a **Spark MapReduce** over the ERA5 grid: `(lat, lon, time) → (country_iso3, year, week)` with area-weighted aggregation, yielding `climate_weekly.parquet` per year. Modules 2–4 then join climate, FluNet, and airline graph features into a 28-feature master table (89,980 rows × 119 countries) for Ridge, XGBoost, and A3T-GCN models.

Why this is hard:

- **ERA5 dominates storage cost.** 125 GB of raw NetCDF dwarfs every other source. After Spark processing the outputs are ~50 MB total, so the raw files are "read-once." The architecture must decide whether to keep them or discard them — the choice is irreversible.
- **Late data is structural, not exceptional.** ~42% of country-weeks in the GNN's dense `(T, N, F)` tensor are absent — countries report flu with 4–12 week lags. Any retraining on a different WHO export date silently produces a different training corpus.
- **Reproducibility is a safety property.** When the GNN underperformed on test (RMSE 1034 vs Ridge 552), we needed to prove the split was identical across all five model runs. Currently impossible — the parquet files have no version history.
- **Four modules × five contributors produce features independently.** If `mobility_features.parquet` (Module 3) changes schema, Module 4's join silently drops rows. No lineage system catches this.
- **Regime shifts break model assumptions.** The 2020 COVID travel collapse invalidated the static airline graph that the GNN was built on. A proper lakehouse would detect this drift and support pinned-corpus retraining.

---

## 2. Architecture Diagram

```
INGESTION
  WHO FluNet (weekly)          ERA5 ECMWF CDS (yearly)         IATA (yearly)
  30 MB CSV, 189 countries     125 GB NetCDF, global 0.25°      ~50 MB CSV
         │                     t2m / d2m / tp variables                │
         │                              │                              │
         ▼                              ▼                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  BRONZE  (append-only, schema-on-read, raw fidelity preserved)               │
│                                                                              │
│  flunet_raw/        ← Delta, partitioned by iso_year                         │
│    iso_year=1995/ … iso_year=2026/   ← ZSTD, ~4 MB total                    │
│                                                                              │
│  era5_raw/          ← Delta, partitioned by year                             │
│    year=2001/ … year=2025/                                                   │
│    Format: Parquet converted from .nc at ingest (see D8)                     │
│    Size: ~1 GB after ZSTD (125 GB raw → 125× compression on gridded floats) │
│    Raw .nc files discarded after validated Parquet write (irreversible)      │
│                                                                              │
│  mobility_raw/      ← Delta, yearly snapshot (~50 MB)                       │
│                                                                              │
│  Schema evolution: additive only (new columns → NULL backfill)              │
│  Retention: flunet RETAIN 52 VERSIONS; era5 RETAIN 5 VERSIONS (yearly)     │
└────────────────┬──────────────────────────┬──────────────────────────────────┘
                 │  Module 1: FluNet dedup  │  Module 1: ERA5 Spark MapReduce
                 │  + MinHash-LSH           │  (lat,lon,time)→(country,week)
                 │                          │  area-weighted mean, unit convert
                 ▼                          ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  SILVER  (validated, deduplicated, schema-enforced, country-week grain)      │
│                                                                              │
│  flunet_clean/      ← Delta, partitioned by (iso_year, country_iso3)        │
│  climate_weekly/    ← Delta, same partition key                              │
│    temperature_c, humidity_pct, precipitation_mm per (country, year, week)  │
│  climate_anomaly/   ← Delta, anomaly + lags 1–4 + climate-stress index      │
│  mobility_features/ ← Delta (Module 3: mobility-weighted neighbour flu)     │
│  graph_metrics/     ← Delta (Module 3: PageRank, TSPR, betweenness)        │
│  mobility_edges/    ← Delta (airline graph edges for GNN)                   │
│                                                                              │
│  ACID MERGE for late FluNet arrivals:                                        │
│    MERGE INTO flunet_clean AS tgt USING new_batch AS src                    │
│    ON  tgt.country_iso3 = src.country_iso3                                  │
│    AND tgt.iso_year = src.iso_year AND tgt.iso_week = src.iso_week          │
│    WHEN MATCHED AND src.report_ts > tgt.report_ts THEN UPDATE SET …        │
│    WHEN NOT MATCHED THEN INSERT …                                           │
│                                                                              │
│  ERA5 is immutable once written (no late-arriving climate data)             │
│  Data contracts: schema validated against expected_schema.json at write     │
│  Retention: RETAIN 104 VERSIONS (2 years → covers any training window)     │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │  Module 4: feature join + model training
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  GOLD  (model-ready: master features + predictions + risk scores)            │
│                                                                              │
│  master_features/   ← Delta, 89,980 rows × 28 features                      │
│    partitioned by (train_split, iso_year)                                   │
│    Training run records delta_snapshot_version in MLflow params             │
│                                                                              │
│  predictions/       ← Delta, (model_id, country_iso3, year, week, pred)    │
│  country_risk_scores/ ← Delta, weekly risk ranking                          │
│                                                                              │
│  model_registry/    ← MLflow: ridge_enhanced v1 → flu_model.pkl            │
│    snapshot_version → Silver Delta version → Bronze ERA5 version           │
│                                                                              │
│  Retention: RETAIN ALL VERSIONS (Gold is ~2 MB; reproducibility > cost)    │
└──────────────────────────────────────────────────────────────────────────────┘

QUERY PATHS
  Hot  (dashboard, last 12 weeks):  DuckDB on local Silver parquet,  <1 s
  Warm (ad-hoc, 1–5 yrs):          DuckDB + delta-rs time-travel,   <30 s
  Cold (full ERA5 history/audit):   Spark full scan,                 no SLA
```

---

## 3. Key Decisions

### D1 — Table format: Delta Lake over Iceberg and raw Parquet

**Chosen: Delta Lake.**

I chose Delta because `delta-rs` provides a pure-Python, zero-JVM implementation that integrates directly with our DuckDB + pandas pipeline. The Python ecosystem was a non-negotiable constraint: all four modules are Jupyter notebooks, and spinning up a JVM for Iceberg catalog operations would require restructuring every notebook.

I rejected **Apache Iceberg**: Iceberg's time travel and branching (Nessie) are architecturally superior for multi-writer scenarios, but Python support (`pyiceberg`) lags delta-rs on stability, and we have a single-writer pipeline — one notebook writes each layer.

I rejected **raw Parquet with versioned filenames** (e.g. `flunet_clean_v3.parquet`): this is what the project currently does. It breaks reproducibility because `v3` has no machine-readable link to the WHO export date that produced it, and there is no `MERGE`-on-late-data support.

### D2 — Catalog: file-based (no server) with upgrade path to Apache Polaris

**Chosen: file-based catalog (Delta transaction log as the catalog).**

At current scale (5 Delta tables, 1 team, 1 compute environment), a catalog server adds operational overhead without benefit. `delta-rs` reads the `_delta_log/` transaction log directly; DuckDB's `delta` extension reads the same files.

I rejected **AWS Glue**: requires AWS account + IAM setup; overkill for a research project that must run from a clean laptop checkout.

I rejected **Databricks Unity Catalog**: vendor lock-in, monthly cost, and incompatible with our local DuckDB query path.

**Upgrade path**: when the project moves to multi-team or multi-compute, register Delta tables in **Apache Polaris** (REST Catalog spec) without changing the underlying file layout. Delta with UniForm can expose the same files as Iceberg to Polaris simultaneously.

### D3 — Partitioning: `(iso_year, country_iso3)` for Silver, `iso_year` only for Bronze

**Chosen: `(iso_year, country_iso3)` for Silver.**

The dominant query pattern is "all weeks for country X in year Y" — fetching a country's full time series for feature engineering. Partitioning on both year and country_iso3 gives DuckDB partition pruning on the most common filter and keeps each partition file small (~5–20 KB after ZSTD).

I rejected **`(iso_year, iso_week)`**: this matches the ingestion cadence but scatters a single country's time series across 52 partitions, destroying the sequential read pattern that Module 1 and Module 4 rely on.

I rejected **single `iso_year` partition for Silver**: directory-level scans would read all 189 country files when only one is needed.

Bronze uses `iso_year` only because the raw data arrives as full-year weekly batches from WHO, and Bronze readers always scan full years.

### D4 — Compression: ZSTD over Snappy and LZ4

**Chosen: ZSTD (level 3).**

FluNet records are highly repetitive: `WHOREGION`, `HEMISPHERE`, `COUNTRY_CODE` repeat thousands of times. ZSTD at level 3 achieves ~8–10× compression on these columns vs the raw CSV's ~3–4× gzip, at decompression speeds competitive with Snappy. The 164 KB `flunet_clean.parquet` (from 30 MB raw CSV) already demonstrates this ratio.

I rejected **Snappy**: better decompression throughput, but 20–30% larger files. At our scale, storage cost is irrelevant, but the habit of max compression matters when scaling to clinic-level data.

I rejected **LZ4**: fastest decompression, worst ratio. Only justified for hot-path streaming writes, which this pipeline does not have.

### D5 — Late-data handling: ACID MERGE with timestamp guard over append-and-deduplicate

**Chosen: ACID MERGE with `report_ts` guard (shown in diagram).**

WHO country uploads arrive with arbitrary lag. A naïve append would duplicate rows every time a country re-submits a corrected weekly report. The MERGE with `src.report_ts > tgt.report_ts` ensures Silver always holds the most recent authoritative value while retaining the old version in the Delta transaction log for audit.

I rejected **append-only + downstream dedup**: this is what Module 2's MinHash-LSH currently does. It works but loses the correction — we cannot distinguish "new record" from "corrected record," so the Silver layer drifts from ground truth silently.

I rejected **overwrite-partition**: blows away time travel history for the affected partition and risks a partial-write failure leaving the partition corrupt.

### D6 — Feature/model versioning: Delta time travel + MLflow over DVC and manual pkl naming

**Chosen: Delta snapshot version pinned in MLflow run metadata.**

Each training run records `delta_snapshot_version: 42` (or equivalent commit hash) in MLflow's `run.data.params`. To reproduce any model, check out `DeltaTable.forVersion(42)` and re-run the notebook. The model artifact (`flu_model.pkl`) is stored in the MLflow artifact store alongside the snapshot pointer.

I rejected **DVC**: DVC tracks file hashes but does not natively track the Delta snapshot that produced them. Two sources of truth for the same lineage question.

I rejected **versioned filenames** (`master_features_v3.parquet`): the current approach. It has no machine-readable link between a file version and the WHO export date or the model metrics.

### D8 — ERA5 Bronze format: convert NetCDF → Parquet at ingest, discard raw .nc

**Chosen: convert ERA5 `.nc` to Parquet at the Bronze boundary; discard raw NetCDF after a validated write.**

The 125 GB of raw ERA5 NetCDF files are grid-native: `(time, lat, lon)` float32 arrays optimized for array slicing, not for the country-week aggregation our pipeline performs. After the Module 1 Spark MapReduce, the gridded data is never queried directly again — it has been fully reduced to 189 country-week rows per year. Retaining 125 GB of `.nc` files at ~$2.88/month (S3 Standard) in perpetuity to support a reprocessing path we have never needed is not defensible.

The conversion yields ~1 GB of ZSTD Parquet (125× compression on repetitive float arrays), preserving the three variables we use — `t2m`, `d2m`, `tp` — at full precision. If we ever need to reprocess with different spatial aggregation, we re-download from ECMWF CDS (free for registered users) rather than maintaining a local archive.

I rejected **keeping raw .nc in Bronze**: $2.88/month × 12 = $34.56/year with no query or reproducibility benefit. ERA5 is not a mutable source — ECMWF does not publish corrections to historical reanalysis — so there is no "late data" case for raw ERA5.

I rejected **Zarr**: Zarr is the ideal format for cloud-native gridded array access (chunked along lat/lon/time), but our access pattern is always already-aggregated country-week rows, never raw grid slices. Zarr adds no value over Parquet here and would require an additional conversion step.

I rejected **skipping Bronze for ERA5** (Spark reads `.nc` directly → Silver): loses the audit trail of which ERA5 version fed each model run. Storing the Parquet-converted Bronze preserves Delta time travel for the ERA5 source, so a future "retrain from 2018 ERA5 data" query is deterministic.

### D7 — Ingestion cadence: weekly batch over real-time streaming

**Chosen: weekly batch pull.**

WHO FluNet updates weekly. There is no sub-weekly signal in the data. Streaming infrastructure (Kafka, Debezium) would add >$200/month in managed service cost for zero analytical benefit — the freshest possible data is still one week old.

I rejected **daily pull**: WHO does not publish daily diffs; a daily pull just re-downloads the same data with slightly higher egress cost.

I rejected **event-driven (push from WHO)**: WHO does not offer webhooks or Change Data Capture.

---

## 4. Failure Modes

### FM1 — COVID regime shift (the incident that already happened)

**What breaks at 3 AM:** The GNN is retrained on a new WHO export. Because airline mobility edges are static (fixed 2017 topology), the model continues to diffuse flu along routes that collapsed in 2020. Test RMSE degrades from 552 (Ridge) to 1034 (GNN) — a 1.9× blowup that is not caught until the model dashboard is reviewed manually.

**Detection:** Compare val RMSE at training time against a rolling baseline threshold stored in the MLflow model registry. Flag any new run where `val_rmse > 1.2 × best_registered_val_rmse`.

**Rollback (ties to time travel):** The Delta transaction log for `mobility_edges/` retains every snapshot. Roll back to version N-1 (pre-collapse routes), retrain, and re-evaluate. The pinned snapshot version in MLflow identifies exactly which topology version each historical model was trained on.

### FM2 — Late WHO reporting corrupts the training window

**What breaks at 3 AM:** A weekly pipeline run lands on a date when 30+ countries have not yet uploaded their week-48 data. Module 4 trains on the incomplete Silver snapshot. `ridge_enhanced` achieves artificially high val RMSE because the validation set is missing the outbreak peaks from late-reporting countries (historically the highest-burden ones).

**Detection:** A row-count check before any Gold write: `expected_rows(year, week) = historical_median_countries_reporting(week) × 0.85`. If the Silver week-48 partition falls below 85% of historical median, abort the pipeline and alert.

**Rollback:** Delta time travel — re-run Module 4 against `DeltaTable.forVersion(last_complete_version)`. The snapshot pointer in MLflow identifies which version was used in each prior run, so the rollback is deterministic.

### FM3 — MinHash-LSH over-deduplication removes valid records

**What breaks at 3 AM:** Module 2's MinHash-LSH fingerprints two genuinely distinct country-week records as duplicates (e.g., two countries share an identical zero-count week). The Silver `flunet_clean` table silently loses rows, and downstream coverage gaps appear in the GNN's observed-mask channel — inflating the 42% sparsity figure and degrading GNN convergence.

**Detection:** After every Silver write, assert `silver_row_count(year) ≥ bronze_row_count(year) × 0.98` (at most 2% dedup rate is plausible; higher signals a collision burst). Also assert per-country row continuity: no country should go from 52 weeks to 0 weeks in a single year.

**Rollback:** Delta time travel on `flunet_clean/` to the pre-dedup Silver version. Tune the MinHash threshold, re-run Module 2, re-write Silver. Because Bronze is append-only and immutable, the source data is always recoverable.

### FM5 — ERA5 spatial aggregation silently wrong for a country

**What breaks at 3 AM:** Module 1's Spark MapReduce joins each ERA5 grid cell `(lat, lon)` to a country polygon via `geopandas`. If the country polygon lookup is wrong for a specific country — e.g., a coastline that leaves offshore grid cells unmatched, or a bounding-box rounding error for a small island nation — all climate features for that country across 25 years are silently biased or zero. The model trains without error; the bias only surfaces if someone inspects per-country feature distributions.

**Detection:** After every Silver `climate_weekly` write, assert: (1) no country has all-zero temperature for any full year (physically impossible), (2) the row count per country matches the Count-Min Sketch estimate from Bronze profiling within 10%, (3) the Stratified Reservoir QA sample — already computed in Module 1 — produces a country-week mean temperature within ±3°C of the ERA5 documented climatology for that region.

**Rollback (ties to time travel):** Because `era5_raw/` in Bronze is a Delta table, roll back to the Bronze ERA5 version N-1, reprocess Module 1 with the corrected polygon lookup, and overwrite `climate_weekly/` in Silver. The Delta transaction log shows exactly which Bronze ERA5 version produced each Silver write, so the lineage chain is complete.

### FM4 — Schema drift from WHO column additions

**What breaks at 3 AM:** WHO adds three new columns to FluNet (this already happened: `PSOURCE_SUBTYPE_INF`, `PSOURCE_PPOS_INF`, `PSOURCE_RSV` appeared in recent exports). The Bronze Delta write succeeds (additive schema evolution), but the Silver transformation script references column positions by index, not name, causing a silent column-shift misalignment in `flunet_clean`.

**Detection:** Enforce a schema contract at the Silver write boundary: compare the output schema against a registered `expected_schema.json`. Any unexpected column or type change raises a schema mismatch error before the write commits.

**Rollback:** Delta's schema evolution audit shows exactly which transaction introduced the new columns. Bronze data is intact; reprocess Silver from any Bronze snapshot using the corrected transformation script.

---

## 5. Cost Estimate

ERA5 dominates every other storage line by two orders of magnitude. The FinOps story is entirely about whether you keep raw NetCDF.

### Scenario A — Keep raw ERA5 .nc (naive)

```
ERA5 raw NetCDF (125 GB):
  125 GB × $0.023/GB-month (S3 Standard) = $2.88/month
  × 12 months = $34.56/year  ← entire budget for one source

FluNet Bronze parquet (4 MB):         ~$0.0001/month
IATA mobility (50 MB):                ~$0.001/month
Silver (6 Delta tables, ~50 MB):      ~$0.001/month
Gold (features + predictions, ~2 MB): ~$0.00005/month
MLflow artifacts (5 × 50 MB pkl):     ~$0.006/month

Spark compute (Module 1, yearly ERA5 reprocess, 4-node cluster, 2 hr):
  4 × r5.xlarge spot × $0.06/hr × 2 hr = $0.48/run × 25 years = $12 one-time

TOTAL (ongoing): ~$3/month  — ERA5 raw storage is 95% of the bill
```

### Scenario B — Convert ERA5 to Parquet at Bronze, discard raw .nc (chosen, D8)

```
ERA5 Bronze Parquet (125 GB raw → ~1 GB ZSTD Parquet):
  1 GB × $0.023/GB-month = $0.023/month

  After 90 days, ERA5 Bronze moves to S3 Infrequent Access (query cadence < 1/month):
  1 GB × $0.0125/GB-month (S3-IA) = $0.0125/month

FluNet + mobility + Silver + Gold + MLflow: ~$0.01/month (as above)

Spark compute (Module 1, yearly, 4-node spot, 2 hr): $0.48/run
  Amortized over 52 weekly runs: $0.009/week = ~$0.04/month

TOTAL (ongoing): ~$0.06/month  — 50× cheaper than keeping raw .nc
```

### FinOps math summary

| Decision | Monthly cost | Driver |
|---|---|---|
| Keep raw ERA5 .nc | ~$3.00 | 125 GB S3 Standard |
| Convert + discard .nc (D8) | ~$0.06 | 1 GB Parquet on S3-IA |
| Savings | **$2.94/month = $35/year** | |

**The FinOps answer:** storage tiering matters exactly one place — the ERA5 Bronze layer. Everything else (FluNet, Silver, Gold, model artifacts) is so small that tiering adds operational complexity for negligible savings. Compute for GNN training (GPU-hours) is a separate budget line that dwarfs all storage costs regardless of scenario.

---

## 6. MVP Slice — One Week

The smallest shippable slice that proves the architecture works is:

**Convert `flunet_clean.parquet` to a Delta table with time-travel and a training snapshot API.**

```python
# poc/delta_snapshot.py  (~50 lines)
from deltalake.writer import write_deltalake
from deltalake import DeltaTable
import pandas as pd

# Write: Bronze → Silver as Delta
df = pd.read_parquet("flunet_clean.parquet")
write_deltalake(
    "silver/flunet_clean",
    df,
    partition_by=["iso_year", "country_iso3"],
    mode="overwrite",
    configuration={"delta.dataSkippingNumIndexedCols": "3"},
)

# Pin a training snapshot (Module 4 records this version number in MLflow)
def load_training_snapshot(version: int) -> pd.DataFrame:
    dt = DeltaTable("silver/flunet_clean", version=version)
    return dt.to_pandas()

# Demonstrate time travel: compare version 0 vs current
v0 = load_training_snapshot(0)
current = load_training_snapshot(DeltaTable("silver/flunet_clean").version())
print(f"v0 rows: {len(v0)}, current rows: {len(current)}, delta: {len(current) - len(v0)}")
```

This proves:
1. The Delta write works on the existing data without any schema changes.
2. `DeltaTable.forVersion(N)` can deterministically reproduce the exact training corpus used for any past model run (FM1, FM2 rollback paths).
3. The integration with the existing pandas/parquet pipeline is zero-friction — `to_pandas()` returns the same DataFrame Module 4 already consumes.

**What this does NOT include** (deferred to week 2+): ACID MERGE for late-data upserts, MLflow run parameter tagging, schema contract enforcement at Silver write, Bronze layer for raw CSV. Those require ~2 more weeks but are blocked on nothing from week 1.

---

## Concepts Applied (Self-Checklist)

| Concept | Where applied |
|---|---|
| **Medallion (Bronze/Silver/Gold)** | Three-layer layout with two distinct Bronze paths: FluNet (weekly, MERGE-capable) and ERA5 (yearly, immutable Parquet conversion) |
| **ACID + MERGE** | Late-data upsert for FluNet in Silver with `report_ts` guard (D5, FM2); ERA5 is immutable (no late-arriving reanalysis data) |
| **Time travel** | Delta `RETAIN N VERSIONS`; snapshot version pinned in MLflow; Bronze ERA5 version traceable through Silver to Gold (D6, D8, FM1, FM2, FM3, FM5) |
| **Catalog** | File-based today; upgrade path to Apache Polaris REST Catalog without changing file layout (D2) |
| **Lineage** | MLflow `delta_snapshot_version` links model artifact → Gold → Silver → Bronze ERA5 version; schema contracts at Silver write catch upstream drift (FM4, FM5) |
| **Security / governance** | No PII (country-level only); schema contracts enforce data quality; ERA5 discard policy documented and irreversible — requires explicit decision (D8) |
| **FinOps** | ERA5 raw .nc (125 GB) is the single dominant cost driver; D8 converts to Parquet + S3-IA tiering saves $35/year (50× reduction); all other sources are negligible |
