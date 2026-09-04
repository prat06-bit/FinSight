# FinSight — Step 2: Risk-Model Evaluation, Threshold Calibration, & Explainability

FinSight implements a leak-free Machine Learning framework for corporate financial deterioration prediction using historical SEC EDGAR filings.

---

## 1. Temporal Split Methodology

To prevent temporal data leakage and forward-looking bias, FinSight enforces a **Three-Stage Temporal Split** across historical SEC filing years:

```
┌───────────────────────────────────┬─────────────────────────┬───────────────────────────┐
│     TRAINING SET (2007–2017)      │  VALIDATION (2018–2020) │    FINAL TEST (2021–2025)  │
│  Model Fitting & Feature Weights  │ Threshold Calibration   │  Untouched Out-of-Time    │
│            397 samples            │       119 samples       │        170 samples        │
└───────────────────────────────────┴─────────────────────────┴───────────────────────────┘
```

### Leakage Prevention Rules
- **No Year Overlap:** `Train`, `Validation`, and `Test` sets share zero filing years.
- **Untouched Final Test Set:** The 2021–2025 test period is strictly held out. It is **never** used during feature selection, hyperparameter tuning, threshold selection, or probability calibration.
- **Backward-Looking Features:** All rolling-window features (3-year moving averages, momentum) use strictly backward-looking observations (`rows[max(0, i-2): i+1]`).

---

## 2. Class Distribution & Regime Shift Analysis

| Dataset Period | Filing Years | Total Samples | Healthy (0) | Deteriorated (1) | Positive Rate (%) | Macroeconomic Regime |
|---|---|---|---|---|---|---|
| **Train** | 2007–2017 | 397 | 256 | 141 | 35.5% | Pre-COVID baseline |
| **Validation** | 2018–2020 | 119 | 78 | 41 | 34.5% | Pre-COVID baseline |
| **Final Test** | 2021–2025 | 170 | 88 | 82 | **48.2%** | **Post-COVID Regime Shift** |

> **Regime Shift Note:** The post-2020 test period experienced a higher rate of corporate financial deterioration (48.2% vs. 34.5%) driven by Federal Reserve rate hikes (0% to 5.25%), post-COVID inflation, and supply chain realignments.

---

## 3. Threshold Calibration Methodology

In financial risk management, **false negatives** (missing a company undergoing financial distress) are significantly more costly than false positives.

The classification threshold was evaluated across `0.10` to `0.90` **strictly on the Validation set (2018–2020)**:

### Validation Set (2018–2020) Threshold Calibration

| Objective / Candidate | Threshold | Accuracy | Precision | **Recall** | **F1-Score** | TP | FP | TN | **FN (Missed Risk)** |
|---|---|---|---|---|---|---|---|---|---|
| Maximize F1 / Default 0.50 | `0.50` | 67.2% | 51.9% | 65.9% | **0.581** | 27 | 25 | 53 | 14 |
| Target ~90% Recall | `0.25` | 50.4% | 40.4% | **92.7%** | 0.563 | 38 | 56 | 22 | 3 |
| **Selected (Target ~80% Recall)** | **`0.30`** | **50.4%** | **39.5%** | **82.9%** | **0.535** | **34** | **52** | **26** | **7** |

> **Selection Objective:** Threshold `0.30` was selected on the Validation set to prioritize risk recall (~83% on Val) while maintaining reasonable precision before locking the threshold for final test evaluation.

---

## 4. Final Evaluation Results (Untouched 2021–2025 Test Set)

The locked threshold (`0.30`) selected on Validation was evaluated **exactly once** on the untouched 2021–2025 Test set:

```
Locked Threshold : 0.30
Accuracy         : 58.8%
Precision        : 54.5%
Recall           : 87.8%  (Caught 72 out of 82 actual deterioration cases)
F1-Score         : 0.673
ROC-AUC          : 0.707  (Out-of-time discriminative ranking)
PR-AUC           : 0.747  (Precision-Recall area)
```

### Final Test Confusion Matrix (at locked 0.30 threshold)

$$\begin{pmatrix} \text{True Negatives (TN): 28} & \text{False Positives (FP): 60} \\ \text{False Negatives (FN): 10} & \text{True Positives (TP): 72} \end{pmatrix}$$

---

## 5. Comparative Benchmark

### Random Split Benchmark (Reference Only)
> *"Random split benchmark — potentially optimistic and not used as the production evaluation."*

* **Train Accuracy:** 93.4%
* **Test Accuracy:** 90.4%
* **Precision:** 0.852
* **Recall:** 0.784

> **Methodological Note:** Random splitting leaks cross-company and macro-economic period signals between train and test sets, yielding overly optimistic metrics. Temporal splitting is required for realistic financial evaluation.

---

## 6. Explainability Methodology (SHAP)

FinSight uses **SHAP (SHapley Additive exPlanations)** with `shap.TreeExplainer` to interpret XGBoost feature attributions.

### Global Top 10 Features (by Mean $|SHAP|$)

| Rank | Feature | Mean $|SHAP|$ | Direction of Effect |
|---|---|---|---|
| 1 | `current_ratio` | 0.3928 | Higher values decrease risk |
| 2 | `dupont_equity_multiplier` | 0.2470 | Higher values increase risk |
| 3 | `net_margin_3yr_avg` | 0.2365 | Higher values decrease risk |
| 4 | `margin_momentum` | 0.2336 | Higher values decrease risk |
| 5 | `return_on_equity` | 0.1451 | Higher values decrease risk |
| 6 | `return_on_assets` | 0.1381 | Higher values decrease risk |
| 7 | `operating_margin` | 0.1132 | Higher values decrease risk |
| 8 | `cpi_inflation` | 0.0969 | Higher values increase risk |
| 9 | `current_ratio_trend` | 0.0772 | Higher values decrease risk |
| 10 | `revenue_growth` | 0.0766 | Higher values decrease risk |

### Local Explanation Example — Apple Inc. (AAPL, 2025)
* **Predicted Risk Probability:** 51.1%
* **Top Local Attributions:**
  * `current_ratio` = 0.8933 $\rightarrow$ SHAP = `+0.8255` (Increases risk due to low liquidity ratio)
  * `return_on_assets` = 0.3118 $\rightarrow$ SHAP = `-0.3719` (Decreases risk due to high asset profitability)
  * `dupont_equity_multiplier` = 4.8722 $\rightarrow$ SHAP = `+0.3544` (Increases risk due to elevated leverage multiplier)

> **Causality Disclaimer:** SHAP values quantify statistical feature attribution in the model's learned predictions. They do not assert direct structural or causal economic relationships.

---

---

## 7. Step 3 — Financial Forecasting & Walk-Forward Validation Methodology

FinSight implements a leakage-safe time-series forecasting pipeline using an out-of-time walk-forward backtest engine.

### Why Random Splitting is Inappropriate for Time-Series Forecasting
Random train/test splits disrupt the temporal sequence of financial filings, allowing future revenue values to leak into training feature sets. Walk-forward (expanding-window / rolling-origin) validation is required:
- **Train through 2015** $\rightarrow$ Predict 2016
- **Train through 2016** $\rightarrow$ Predict 2017
- ...
- **Train through 2024** $\rightarrow$ Predict 2025

### Evaluation Scopes: Aggregate 40-Company Benchmark vs. Individual AAPL Benchmark

#### Scope A: Aggregate 40-Company Pooled Walk-Forward Benchmark (2016–2025)

| Forecasting Model | Average MAE ($B) | Average RMSE ($B) | **Average sMAPE (%)** | Model Selection Status |
|---|---|---|---|---|
| **Holt Exponential Smoothing** | **$8.650B** | **$19.387B** | **6.94%** | **Selected Model (Best sMAPE)** |
| **Naive Last-Year Baseline** | $9.940B | $21.054B | 8.35% | Baseline |
| **XGBoost Regressor (With Macro)** | $15.659B | $36.964B | 13.25% | ML Model |
| **XGBoost Regressor (Without Macro)** | $16.483B | $38.307B | 13.91% | ML Model |

#### Scope B: Individual AAPL Walk-Forward Backtest (2016–2025)

| Forecasting Model | MAE ($B) | RMSE ($B) | **sMAPE (%)** | Selection Status |
|---|---|---|---|---|
| **Holt Exponential Smoothing** | **$6.603B** | **$11.837B** | **10.47%** | **Selected Model for AAPL** |
| **Naive Last-Year Baseline** | $8.966B | $17.044B | 12.51% | Baseline |
| **XGBoost (Without Macro)** | $9.493B | $19.675B | 11.71% | ML Model |
| **XGBoost (With Macro)** | $9.743B | $20.584B | 12.00% | ML Model |

### Macro-Feature Ablation Findings
- **Aggregate 40-Company ML Regressor:** Adding macro features (`fed_funds_rate`, `treasury_10y_yield`, `yield_curve_slope`, `cpi_inflation`) improved cross-company XGBoost performance (sMAPE improved from **13.91% to 13.25%**, a +0.66 percentage point gain).
- **AAPL Individual Forecast:** Macro features slightly increased XGBoost error (**11.71% without macro vs. 12.00% with macro**). Macro features were not forced into AAPL's model.
- **Model Selection Winner:** **Holt Exponential Smoothing** remained the top-performing model in both evaluation scopes (10.47% sMAPE for AAPL, 6.94% sMAPE Aggregate).

### AAPL 2026 Future Revenue Forecast
- **Historical Latest Year:** 2025 (Revenue: $416.16B)
- **Forecast Target Year:** 2026
- **Selected Model:** Holt Exponential Smoothing
- **Point Forecast:** **$450.93B** (+8.3% YoY growth)
- **95% Prediction Interval:** **$403.83B – $498.02B**

---

## 8. Execution Commands

```bash
# Run complete test suite (41 tests)
.\.venv311\Scripts\python.exe -m pytest -q

# Run Step 2 risk evaluation audit
.\.venv311\Scripts\python.exe -m ml.audit

# Run Step 2 SHAP explainability audit
.\.venv311\Scripts\python.exe -m ml.explainability

# Run Step 3 financial forecast for Apple
.\.venv311\Scripts\python.exe main.py --ticker AAPL forecast

# Run risk prediction for Apple
.\.venv311\Scripts\python.exe main.py --ticker AAPL risk
```

---

## Step 4 — Sector-Relative Ratio Normalization

### Motivation & Objectives
Raw financial metrics (margins, ratios, leverage) vary structurally across industries. For example, software firms naturally operate at 70%+ gross margins, whereas retail firms operate at much lower margins. Comparing companies against global unnormalized thresholds introduces industry bias. Step 4 implements **leakage-safe sector-relative normalization** to benchmark each company against its point-in-time industry peer median.

### Deterministic SIC Industry Grouping
Standard Industrial Classification (SIC) codes extracted from SEC EDGAR filings map deterministically into standard sector groups:
- **Technology (SIC 3570-3579, 3600-3699, 7370-7379):** 18 tickers (AAPL, MSFT, GOOGL, META, NVDA, AMD, INTC, etc.)
- **Healthcare & Pharma (SIC 2830-2836, 3820-3849, 6300-6399):** 11 tickers (JNJ, PFE, UNH, ABBV, MRK, LLY, etc.)
- **Retail & Consumer (SIC 5200-5999, 2080-2089, 3000-3099, 5800-5899):** 10 tickers (AMZN, KO, PEP, WMT, COST, HD, etc.)
- **Industrial & Automotive (SIC 3700-3799):** 1 ticker (TSLA)

### Sector Delta Features (6 Features Added, Total 33 Features)
Features are computed as the difference from the sector median: $\text{Company Metric} - \text{Sector Median Metric}$:
1. `net_margin_sector_delta`
2. `gross_margin_sector_delta`
3. `operating_margin_sector_delta`
4. `roa_sector_delta`
5. `current_ratio_sector_delta`
6. `debt_to_equity_sector_delta`

### Leakage-Safe Peer Median Rules
- **Point-In-Time:** For a record at year $T$, only peer observations from year $T$ are evaluated.
- **Leave-One-Out (LOO):** The focal company's own observation is excluded from its sector peer median calculation.
- **Minimum Sample Rule ($N \ge 3$):** Requires $\ge 3$ eligible peer observations per sector-year. If $N < 3$ (e.g., TSLA as sole automotive ticker), sector deltas fall back safely to `0.0`.

### Model Ablation Experiment Results (Untouched Test 2021–2025)
Comparison of **Model A** (Baseline 27 features) vs. **Model B** (Enhanced 33 features) under three-stage temporal split (Train 2007–2017, Val 2018–2020, Test 2021–2025):

| Evaluation Metric | Model A (27 Feats) | Model B (33 Feats) | Performance Delta |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 55.9% | **57.6%** | **+1.7%** |
| **Precision** | 52.6% | **54.0%** | **+1.4%** |
| **Recall** | 85.4% | **82.9%** | -2.5% |
| **F1-Score** | 0.651 | **0.654** | **+0.003** |
| **ROC-AUC** | 0.707 | **0.706** | -0.001 |
| **PR-AUC** | 0.747 | **0.749** | **+0.002** |

### Global SHAP Explainability Rankings
Two sector-relative delta features entered the **Global Top 10 SHAP Rankings** for risk attribution:
- **Rank #3:** `operating_margin_sector_delta` (Mean $|SHAP| = 0.1897$, Higher values decrease risk)
- **Rank #9:** `current_ratio_sector_delta` (Mean $|SHAP| = 0.0996$, Higher values decrease risk)

*Disclaimer: SHAP values represent empirical feature attribution within model predictions and do not imply direct causal economic relationships.*

### AAPL 2025 Sector-Relative Case Study
- **Sector:** Technology | **Peer Count:** 17 point-in-time peers
- **Net Margin Delta:** +0.0407 (+4.07% above tech peers)
- **Gross Margin Delta:** -0.1803 (-18.03% below tech peers due to high-margin software peer benchmarks)
- **Operating Margin Delta:** +0.0275 (+2.75% above tech peers)
- **ROA Delta:** +0.2013 (+20.13% asset efficiency above tech peers)
- **Current Ratio Delta:** -1.1120 (-1.11 liquidity below tech peers)
- **Debt to Equity Delta:** +0.9594 (+0.96 leverage above tech peers)

---

## Step 5 — Hybrid RAG for SEC 10-K / 10-Q Financial Filings

### Architecture Overview
FinSight implements a leakage-safe Hybrid Retrieval-Augmented Generation (RAG) architecture over SEC EDGAR financial filings. It combines dense semantic retrieval (vector similarity) with sparse lexical search (BM25) using Reciprocal Rank Fusion (RRF).

```
┌────────────────────────────────────────────────────────────────────────┐
│                        User Search / Query                              │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                 ┌─────────────────┴─────────────────┐
                 │                                   │
                 ▼                                   ▼
   ┌───────────────────────────┐       ┌───────────────────────────┐
   │ Dense Semantic Retriever  │       │   Sparse BM25 Retriever   │
   │ HuggingFace all-MiniLM-v2 │       │ Financial Term Tokenizer  │
   │ ChromaDB Metadata Filter  │       │ Persistent Document Index │
   └─────────────┬─────────────┘       └─────────────┬─────────────┘
                 │ (Top-K Dense Ranks)               │ (Top-K BM25 Ranks)
                 └─────────────────┬─────────────────┘
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │ Reciprocal Rank Fusion (RRF)  │
                   │ RRF Score = 1/(60 + Rank)     │
                   └───────────────┬───────────────┘
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │    Top-K Context Results      │
                   │ (Exact Section & Year Format) │
                   └───────────────────────────────┘
```

### Key RAG Features
1. **Local Dense Embedding:** Uses `sentence-transformers/all-MiniLM-L6-v2` (384-dim local embedding) — zero dependency on external cloud APIs or proprietary quota limits.
2. **Metadata-Filtered Vector Store:** Persistent `ChromaDB` collection with strict metadata filtering on `ticker`, `section`, `filing_year`, and `filing_type`.
3. **Financial-Aware BM25 Tokenization:** Tailored tokenizer preserving financial numeric formats, percentages (`15.2%`), dollar metrics (`$10.5B`), and SEC section titles (`Item 1A`, `Item 7`).
4. **Reciprocal Rank Fusion (RRF):** Standardized RRF ($k=60$) combining top-$N$ dense and sparse retrieval ranks with score normalization.
5. **Retrieval Benchmark Engine:** 25-query financial evaluation set measuring **Hit@K**, **MRR (Mean Reciprocal Rank)**, **Precision@K**, and **Recall@K** across Dense, BM25, and Hybrid modes.

### Step 5 RAG Execution Commands

```bash
# Ingest 10-K filings for a company into Hybrid RAG
.\.venv311\Scripts\python.exe main.py ingest --ticker AAPL

# Perform hybrid search across SEC filings
.\.venv311\Scripts\python.exe main.py search --ticker AAPL "supply chain disruption risk" --method hybrid

# Run retrieval benchmark evaluation (Dense vs. BM25 vs. Hybrid)
.\.venv311\Scripts\python.exe -m rag.evaluation
```

