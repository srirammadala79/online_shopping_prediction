# Machine Learning Capstone Project: Online Shopping Prediction Pipeline

**Course:** 23CSE301 Machine Learning  
**Degree:** B.Tech. Computer Science and Engineering (III Year) — Academic Year 2026-27  
**Team:** Team 16  
**Repository:** [https://github.com/srirammadala79/online_shopping_prediction](https://github.com/srirammadala79/online_shopping_prediction)

---

## 1. Repository Directory Structure

This repository follows the required capstone project directory structure:

```text
/ (root)
│
├── README.md                          # Project overview, results, setup instructions
├── requirements.txt                   # All Python dependencies with explicit pinned versions
│
├── data/                              # Raw dataset file(s) and verification utility
│   ├── shopping2.csv                  # Raw clickstream dataset (165,474 records)
│   └── download_data.py               # Dataset validation & integrity inspection utility
│
├── notebooks/                         # Core Jupyter Notebooks
│   ├── regression.ipynb               # Continuous price prediction pipeline (Review 1 Track)
│   ├── classification.ipynb           # [Under development by teammate — Classification Track]
│   └── clustering.ipynb               # [Under development by teammate — Clustering Track]
│
├── models/                            # Serialized Joblib (.pkl) trained model artifacts
│   ├── best_regressor.pkl             # Trained Random Forest Regressor
│   └── reg_preprocessor.pkl           # ColumnTransformer preprocessing pipeline for regression
│
└── figures/                           # Benchmark results and diagnostic CSV exports
    └── regression_model_comparison.csv
```

---

## 2. Project Overview & Scope

This capstone project analyzes customer interaction clickstream records (`shopping2.csv`) containing **165,474 interaction events** across customer sessions:

1. **Regression Track (Continuous Price Prediction — Complete):**
   - Predicts product monetary value (`Price`) based on customer browsing behavior (page sequence, location on page, photography type, session depth, and time of week).
   - Evaluates and benchmarks 10 distinct regression algorithms on the held-out test set.
   - Tunes champion hyperparameters using 5-Fold Cross Validation.

2. **Classification Track (Customer Navigation Category Prediction — In Progress by Teammate):**
   - Multi-class product category prediction (`Main_Category`) with session-grouped splitting.

3. **Clustering Track (Unsupervised Customer Segmentation — In Progress by Teammate):**
   - Behavioral segmentation using K-Means and Agglomerative Hierarchical clustering.

---

## 3. Regression Track Benchmark Results

All 10 regression algorithms were evaluated on the held-out test split using $R^2$, Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE):

| Rank | Algorithm | $R^2$ Score (↑) | RMSE (↓) | MAE (↓) | Status |
|:---:|:---|:---:|:---:|:---:|:---|
| 1 | **Random Forest Regressor** | **0.6335** | **7.1919** | **4.1970** | **Champion Model** |
| 2 | **Decision Tree Regressor** | 0.5855 | 7.6478 | 4.9720 | Runner-Up |
| 3 | **Gradient Boosting Regressor** | 0.5097 | 8.3178 | 6.2314 | Ensemble Baseline |
| 4 | **KNN Regressor** | 0.2340 | 10.3971 | 7.6732 | Non-Parametric |
| 5 | **Support Vector Regressor (SVR)** | 0.1310 | 11.0742 | 7.8999 | Kernel RBF |
| 6 | **Polynomial Regression (Degree 2)** | 0.1192 | 11.1487 | 9.1297 | Polynomial Expansion |
| 7 | **ElasticNet** | 0.0510 | 11.5727 | 9.4652 | Regularized Linear |
| 8 | **Lasso Regression** | 0.0509 | 11.5727 | 9.4641 | L1 Regularization |
| 9 | **Ridge Regression** | 0.0509 | 11.5732 | 9.4645 | L2 Regularization |
| 10 | **Linear Regression** | 0.0509 | 11.5732 | 9.4645 | Baseline Linear |

### Key Regression Findings & Hyperparameter Optimization
- **Champion Architecture:** **Random Forest Regressor** achieved top rank with $R^2 = 0.6335$ and lowest MAE ($4.20$).
- **5-Fold Cross Validation:** Evaluated parameter grid across tree depths and estimators, finding optimal configuration at `n_estimators=100`, `max_depth=12`.
- **Residual Diagnostics:** Errors are symmetrically centered around zero, indicating homoscedastic variance and lack of systemic bias.

---

## 4. Setup & Execution Instructions

### Step 1: Clone the Repository
```bash
git clone https://github.com/srirammadala79/online_shopping_prediction.git
cd online_shopping_prediction
```

### Step 2: Set Up Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Dataset Integrity
```bash
python data/download_data.py
```

### Step 5: Run the Regression Notebook
```bash
jupyter notebook notebooks/regression.ipynb
```

---

## 5. Team 16 Members & Responsibilities
- **Sriram Madala:** Repository structure, data pipeline, feature engineering, and regression track modeling (`notebooks/regression.ipynb`).
- **Teammates:** Classification track (`notebooks/classification.ipynb`) and clustering track (`notebooks/clustering.ipynb`).
