# 🏦 Market Risk Structure & Stability Analysis Using Statistical and Linear Algebra Methods

A quantitative research pipeline investigating the structure and stability of financial market risk — using log return analysis, formal normality testing, eigendecomposition, PCA factor extraction, rolling correlation regime analysis, and variance-threshold feature selection across 13 years of S&P 500 data.

---

## 🧩 Project Overview

This project asks a precise question: **which statistical properties of financial markets are stable enough to be relied upon in a model, and which change over time?** It builds a full analytical pipeline from raw price data to PCA-reduced feature sets, with a focus on three things that are often glossed over in surface-level analyses:

1. **Formal normality testing** — not just visual inspection, but a Shapiro-Wilk test with p-values
2. **Variance-threshold feature selection** — systematically identifying which assets carry enough signal to include in a factor model
3. **Rolling correlation regime analysis** — specifically tracking how the IBM–MSFT relationship evolves across 13 years, exposing the time-varying nature of cross-asset dependence

Where [Project 6](../market-risk-factor-analysis-pca) uses simple returns and focuses on distribution characterisation and PCA explained variance, this project uses **log returns throughout** (time-additive, theoretically grounded) and extends into **quantitative feature selection** as a precursor to model-ready data preparation.

---

## 🎯 What It Investigates

| Section | Analysis |
|---|---|
| **1. Data Pipeline** | Long-format CSV ingestion with Ticker tagging; pivot to price matrix |
| **2. Return Construction** | Both simple and log returns computed; log returns used for all analysis |
| **3. Distribution Analysis** | IBM log return histogram + KDE; visual fat-tail identification |
| **4. Normality Testing** | Shapiro-Wilk test on IBM log returns (sample n=1,000) |
| **5. Heavy-Tail Analysis** | Kurtosis computation; interpretation against normal baseline |
| **6. Descriptive Statistics** | Mean, volatility (std), skewness, kurtosis — full 10-asset table |
| **7. Confidence Intervals** | 95% CI on IBM mean log return using t-distribution and SEM |
| **8. Hypothesis Testing** | Two-sample t-test: AAPL vs IBM mean log returns |
| **9. Linear Algebra** | Return vectors and T×N matrix representation |
| **10. Covariance Matrix** | Log-return covariance; full 10×10 heatmap |
| **11. Eigendecomposition** | `np.linalg.eig()` on log-return covariance matrix |
| **12. PCA Factor Extraction** | StandardScaler + full PCA; cumulative explained variance curve |
| **13. Feature Stability** | 20-day rolling covariance and rolling correlation |
| **14. IBM–MSFT Regime Analysis** | Time-series plot of 20-day rolling pairwise correlation, 2010–2023 |
| **15. Variance-Threshold Selection** | Filters to assets with `variance > 0.0005` |
| **16. PCA Noise Reduction** | 3-component PCA on filtered feature set (AMD, NFLX, TSLA) |

---

## 📦 Dataset

- **Source:** S&P 500 Stock Prices (Kaggle) — individual OHLCV CSVs per ticker
- **Tickers:** AAPL, AMD, AMZN, CSCO, GOOG, IBM, MSFT, NFLX, SBUX, TSLA
- **Raw coverage:** AAPL from 1980, IBM from 1962, TSLA from 2010 (most recent IPO)
- **Analysis window:** June 2010 – August 2023 (3,297 concurrent price days; 3,296 return observations after differencing)
- **Missing values pre-alignment:** TSLA 12,206, GOOG 10,731 — all pre-IPO periods, dropped via `dropna()`

---

## 🗂️ Project Structure

```
market-risk-stability-analysis/
│
├── Market_Risk_Structure_Stability_Analysis_Using_Statistical_and_Linear_Algebra_Methods.ipynb
└── README.md
```

---

## 🔧 Technical Stack

| Library | Purpose |
|---|---|
| `pandas` | CSV ingestion, pivot table, rolling covariance/correlation |
| `numpy` | Log return formula, matrix representation, `linalg.eig()` |
| `scipy.stats` | Shapiro-Wilk test, t-distribution CI, two-sample t-test |
| `sklearn.decomposition.PCA` | Full PCA + 3-component noise reduction |
| `sklearn.preprocessing.StandardScaler` | Return standardisation before PCA |
| `matplotlib` / `seaborn` | Distribution histograms, covariance heatmap, rolling correlation plot, cumulative variance curve |

---

## 📐 Methodology

### Data Architecture

Raw CSVs are loaded with an explicit `Ticker` column tagged at ingestion, concatenated into a long-format panel (all tickers × all dates), then pivoted to a wide price matrix with `Date` as index. This is a more explicit and auditable approach than the `globals()` method — the ticker label travels with the data through the ingestion step.

After pivoting and `dropna()`, the working dataset is 3,297 dates × 10 tickers, all priced concurrently from TSLA's IPO date onward.

### Log Returns Throughout

This project uses log returns exclusively for all statistical analysis:

$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

Log returns are **time-additive** — a property that makes them the correct choice for multi-period aggregation and for comparing returns across assets with different price levels. Simple returns are computed for reference but log returns drive all downstream analysis.

### Normality Testing — Shapiro-Wilk

Beyond visual inspection, normality is formally tested using the **Shapiro-Wilk test** on a 1,000-observation random sample of IBM log returns:

```
Shapiro-Wilk statistic: 0.8640
p-value: 1.45 × 10⁻²⁸
```

The p-value is effectively zero — normality is rejected at any conventional significance level. The Shapiro-Wilk test is more powerful than Jarque-Bera for detecting non-normality in moderate sample sizes, making it the appropriate choice here.

**IBM kurtosis: 10.20** — extreme events in IBM occur at a rate consistent with a distribution with far heavier tails than the normal. For context, a normal distribution has excess kurtosis of 0; IBM's 10.20 reflects the kind of tail behaviour that caused the failure of Value-at-Risk models during the 2008 financial crisis.

### Descriptive Statistics (Log Returns)

| Ticker | Mean | Volatility (σ) | Skewness | Excess Kurtosis |
|---|---|---|---|---|
| AAPL | 0.000922 | 0.017837 | −0.260 | **5.61** |
| AMD | 0.000827 | 0.035416 | +0.290 | **9.94** |
| AMZN | 0.000962 | 0.020733 | +0.032 | **6.34** |
| CSCO | 0.000273 | 0.016798 | −0.867 | **16.84** |
| GOOG | 0.000739 | 0.017161 | +0.238 | **7.69** |

Note: the `Variance` column in the notebook reports standard deviation — `log_returns.std()` — rather than variance proper. The distinction matters: variance is `std²`. The actual variances used for feature selection are reported correctly in the `log_returns.var()` output below.

### Confidence Interval — IBM

95% CI for IBM mean daily log return (t-distribution, SEM-based, n=3,296):

```
CI: (−0.000429, +0.000544)
```

Zero lies inside this interval — IBM's mean daily log return is **not statistically distinguishable from zero** over the 13-year period. This contrasts with GOOG in Project 6 (p = 0.003, zero outside CI), reflecting IBM's lower growth profile during this period.

### Hypothesis Testing — AAPL vs IBM

Two-sample t-test on log returns (H₀: mean_AAPL = mean_IBM):
- t-statistic: 2.176, p-value: **0.030**
- Reject H₀ at the 5% level — AAPL and IBM have statistically different mean daily log returns over 2010–2023

This is a meaningful result: AAPL's compound growth story over this period produced a measurably higher mean daily return than IBM's, and that difference is statistically detectable with 13 years of data.

### Covariance Matrix

Log-return covariance for selected pairs:

| | AAPL | AMD | TSLA | IBM |
|---|---|---|---|---|
| **AAPL** | 0.000318 | 0.000248 | 0.000232 | 0.000102 |
| **AMD** | 0.000248 | 0.001254 | 0.000401 | 0.000152 |
| **TSLA** | 0.000232 | 0.000401 | 0.001289 | 0.000111 |
| **IBM** | 0.000102 | 0.000152 | 0.000111 | 0.000203 |

IBM has the lowest covariances with all other names — consistent with its role as the most defensive, least tech-growth-correlated asset in the basket.

### Eigendecomposition

Eigenvalues of the log-return covariance matrix:
```
λ₁ = 0.002546  ← dominant market factor
λ₂ = 0.000879
λ₃ = 0.000818
...
λ₁₀ = 0.000113  ← noise floor
```

λ₁ is 2.9× λ₂, confirming a single dominant risk factor. The eigenvector associated with λ₁ has positive loadings for AMD (0.514), AMZN (0.279), and AAPL (0.228) — the high-growth names drive the market factor most strongly.

### PCA Factor Analysis

Full PCA on standardised log returns:

| PC | Variance Explained | Cumulative |
|---|---|---|
| PC1 | **46.47%** | 46.47% |
| PC2 | 10.24% | 56.71% |
| PC3 | 7.63% | 64.34% |
| PC4 | 6.84% | 71.18% |
| PC5 | 6.40% | 77.58% |

The cumulative explained variance curve (plotted with `np.cumsum()`) shows a pronounced elbow after PC1, then a gradual flatten — 5 components explain ~78% of all return variance. The first component at 46.47% is the most direct evidence of a dominant market factor in this dataset.

### Feature Stability — IBM–MSFT Rolling Correlation

20-day rolling pairwise correlation between IBM and MSFT log returns across 2010–2023 reveals significant regime changes. During certain periods — particularly around broad market events — the correlation spikes toward +0.7 or higher. During calmer periods it can drop toward zero or even turn negative (as seen at the end of the sample period where the 20-day IBM–MSFT correlation reaches approximately −0.26).

This is a critical insight for risk modelling: a covariance matrix estimated on a long historical window will systematically understate correlation during stress periods and overstate it during calm ones.

### Variance-Threshold Feature Selection

Log-return variances across the basket:

| Ticker | Variance | Selected? |
|---|---|---|
| TSLA | 0.001289 | ✅ |
| AMD | 0.001254 | ✅ |
| NFLX | 0.001054 | ✅ |
| AMZN | 0.000430 | ❌ |
| GOOG | 0.000295 | ❌ |
| AAPL | 0.000318 | ❌ |
| MSFT | 0.000272 | ❌ |
| CSCO | 0.000282 | ❌ |
| SBUX | 0.000281 | ❌ |
| IBM | 0.000203 | ❌ |

Threshold: `variance > 0.0005`. Only **AMD, NFLX, and TSLA** pass — the three highest-volatility names in the basket. The 7 names below threshold (including AAPL, MSFT, GOOG) are excluded as insufficiently volatile to contribute meaningful signal in a high-variance factor model.

This is a deliberate modelling choice: for a strategy focused on capturing high-volatility regime signals, lower-variance names add noise rather than signal.

### PCA on Filtered Features

3-component PCA on the filtered AMD–NFLX–TSLA subspace produces a 3,296 × 3 factor score matrix. With only 3 input features and 3 components, this is a full representation — the value here is the compression into an orthogonal factor space where the components are uncorrelated by construction, suitable as inputs to downstream models (regression, clustering, regime detection).

---

## 📊 Key Findings

- **IBM log returns formally reject normality** at p = 1.45 × 10⁻²⁸ (Shapiro-Wilk). This is not a borderline result.
- **IBM kurtosis of 10.20** — extreme daily moves in a "safe" blue-chip stock occur 10× more often than a normal model predicts.
- **AAPL and IBM have statistically different mean returns** over 2010–2023 (p = 0.030) — AAPL's growth premium is detectable in the data.
- **PC1 explains 46.47% of log-return variance** — a single latent market factor drives nearly half of all daily price variation across 10 large-cap stocks.
- **IBM–MSFT rolling correlation is highly unstable**, ranging from strongly positive to slightly negative across the sample period — any static correlation estimate is a snapshot, not a stable structural fact.
- **Only 3 of 10 assets pass a variance threshold of 0.0005** — AMD, NFLX, TSLA. Lower-volatility names including AAPL, GOOG, and MSFT do not carry enough daily variance to contribute to a high-signal factor model under this criterion.

---

## 🧠 Concepts Demonstrated

| Concept | Implementation |
|---|---|
| Log vs simple returns | Both computed; log returns used throughout for time-additivity |
| Formal normality testing | Shapiro-Wilk test with p-value interpretation |
| Excess kurtosis as tail risk measure | IBM kurtosis = 10.20 |
| Statistical moments | Mean, std, skewness, kurtosis — full 10-asset DataFrame |
| t-distribution confidence intervals | SEM-based 95% CI on IBM mean log return |
| Two-sample hypothesis testing | AAPL vs IBM mean return comparison |
| Covariance matrix | Log-return covariance; annotated heatmap |
| Eigendecomposition | `np.linalg.eig()` — eigenvalues ranked by magnitude |
| PCA factor extraction | Full PCA + cumulative variance curve |
| Rolling correlation regime analysis | 20-day IBM–MSFT pairwise rolling correlation, 2010–2023 |
| Variance-threshold feature selection | Explicit threshold filtering; 3 of 10 assets selected |
| PCA noise reduction | 3-component PCA on variance-filtered subspace |

---

## 🚀 How to Run

**Requirements:**
```
pandas numpy matplotlib seaborn scipy scikit-learn
```

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

1. Clone the repo and open the notebook in Jupyter or Google Colab
2. Mount Google Drive or update `path` to your local data directory
3. Place the 10 ticker CSVs from the Kaggle S&P 500 dataset in the data folder
4. Run all cells top to bottom

---

## 📌 Context

Seventh project in a Python for Quantitative Finance programme, and a companion to [Project 6](../market-risk-factor-analysis-pca) which uses the same dataset. Where Project 6 focuses on distribution characterisation and PCA explained variance using simple returns, this project uses log returns throughout and extends into **quantitative feature selection** — applying a variance threshold to identify which assets carry enough signal for high-volatility factor modelling. The IBM–MSFT rolling correlation analysis is a direct demonstration of why time-varying covariance models (DCC-GARCH, regime-switching) exist: static correlation estimates are misleading over long horizons.
