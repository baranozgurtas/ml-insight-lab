# 🧪 ML Insight Lab

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?style=flat-square&logo=scikit-learn)
![SciPy](https://img.shields.io/badge/SciPy-1.11+-blue?style=flat-square&logo=scipy)
![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75?style=flat-square&logo=plotly)

An interactive ML system for analyzing model failure, generalization, and statistically sound decision-making.

Most ML projects stop at `accuracy_score`. Real-world ML systems require understanding *why* models fail   
and whether improvements are statistically significant.

---

## Modules

### ⚡ Module 1 — Model Behavior Playground

Visualize how ML models succeed — and fail — under different data conditions.

**What you can explore:**
- Dataset complexity: Moons, Circles, Blobs, Linear
- Noise level, sample size, train/test split
- Five algorithms: Logistic Regression, Decision Tree, SVM (RBF), Random Forest, Gradient Boosting
- Model-specific hyperparameters: max depth, C, n_estimators, learning rate

**Key insights the tool surfaces:**
- Why linear models fail on non-linear data
- How Decision Trees overfit as depth increases
- How more data reduces the train/test gap

**Outputs:**

*Decision Boundary* — Probability contour overlaid with train (circles) and test (diamonds) points. The boundary shape reveals model complexity directly: Logistic Regression produces a linear partition, Random Forests produce increasingly irregular boundaries as depth grows.

*Learning Curves* — Train and cross-validated accuracy plotted against training set size, with ±1 std confidence bands. Reveals data efficiency and how much of the generalization gap closes with more samples.

*Bias/Variance Diagnosis* — Automatic classification with plain-English interpretation:
- **High Bias**: train accuracy < 75% — underfitting, model too simple
- **High Variance**: train-CV gap > 15% — overfitting, model memorizes data
- **Moderate Variance**: gap 7–15% — consider regularization
- **Well Balanced**: gap < 7% — good generalization

---

### 📊 Module 2 — A/B Testing Simulator

Make statistically sound model deployment decisions.

Simulates a real-world scenario: *Should you deploy Model B over Model A?*

**Tab 1: Run A/B Test**

Runs a two-proportion z-test and outputs:

- **p-value** and **z-statistic**
- **Lift**: relative improvement of B over A — `(p_B - p_A) / p_A × 100%`
- **Cohen's h**: effect size for proportions — `h = 2·arcsin(√p_B) - 2·arcsin(√p_A)`, interpreted as Small (< 0.2), Medium (0.2–0.5), Large (> 0.5)
- **Confidence interval** for the difference `(p_B - p_A)`
- **Z-distribution plot** with rejection regions and observed z-statistic
- **Statistical power** of the test
- **Verdict**: deploy / keep baseline / inconclusive

The test statistic:

```
z = (p_B - p_A) / sqrt(p̂(1 - p̂)(1/n_A + 1/n_B))
```

where `p̂ = (successes_A + successes_B) / (n_A + n_B)` is the pooled proportion under H₀.

**Tab 2: Sample Size Calculator**

Computes the minimum samples required per group before running a test, given baseline rate, MDE, α, and target power:

```
n = (z_{α/2} · √(2p̄(1-p̄)) + z_β · √(p_A(1-p_A) + p_B(1-p_B)))² / (p_B - p_A)²
```

Includes an MDE vs. sample size tradeoff chart — making explicit the cost of chasing small improvements.

**Tab 3: Concepts Reference**

Concise definitions of p-value, α, power, confidence intervals, Cohen's h, Type I/II error, lift, and MDE — framed in the context of model comparison.

---

## What This Demonstrates

- Why high accuracy ≠ good model
- How to detect and diagnose overfitting vs. underfitting
- When a performance difference is statistically significant vs. noise
- How much data you need before trusting an A/B test result
- How ML deployment decisions are made in production environments

---

## Demo 
<img width="1438" height="640" alt="Screenshot 2026-04-01 at 2 36 31 PM" src="https://github.com/user-attachments/assets/ffc4d725-8121-4c71-8f3a-c9876269e68f" />
<img width="1436" height="713" alt="Screenshot 2026-04-01 at 2 37 03 PM" src="https://github.com/user-attachments/assets/943c4e65-8e74-4791-8f5c-f66c397fc04f" />
<img width="1433" height="687" alt="Screenshot 2026-04-01 at 2 38 00 PM" src="https://github.com/user-attachments/assets/2288db4d-358d-4e48-a1d5-af9d0b9dd0ae" />
<img width="1439" height="708" alt="Screenshot 2026-04-01 at 2 38 21 PM" src="https://github.com/user-attachments/assets/c8075a26-bbe6-438a-b070-353e0ab5cbf5" />



## Tech Stack

| Layer | Library | Purpose |
|-------|---------|---------|
| UI | Streamlit 1.32+ | Multi-page app with `st.navigation` API |
| Visualization | Plotly | Decision boundaries, learning curves, CI plots, z-distribution |
| ML | scikit-learn | Classification algorithms, learning curve computation |
| Statistics | SciPy | Normal distribution, p-value, power calculation |
| Numerics | NumPy / Pandas | Data generation, mesh grids, array ops |

---

## Project Structure

```
ml-insight-lab/
├── app.py                  # Entry point — st.navigation routing
├── pages/
│   ├── home.py             # Landing page
│   ├── model_playground.py # Module 1
│   └── ab_testing.py       # Module 2
└── requirements.txt
```

---

## Getting Started

```bash
git clone https://github.com/baranozgurtas/ml-insight-lab.git
cd ml-insight-lab
pip install -r requirements.txt
streamlit run app.py
```

App runs at `http://localhost:8501`

---


## Positioning

This project mirrors how ML systems are evaluated in production,  
where model selection depends on reliability, generalization, and robustness — not a single metric.

Core focus:
- Understanding model behavior under changing data conditions  
- Quantifying uncertainty and variance in performance  
- Making deployment decisions backed by statistical evidence  

Inspired by production ML systems, where incorrect deployment decisions carry measurable cost.
