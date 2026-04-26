# 🏦 Nigerian Bank Fraud Detection System

A complete machine learning project that detects fraudulent bank transactions in real time with **97.11% accuracy** and **80% fraud recall**.

## 🌐 Live Demo
**[Try the app →](https://your-deployment-url.up.railway.app)**

---

## 📌 Project Overview
Nigerian banking fraud is a growing challenge — with millions of transactions processed daily, manual detection is impossible. This system uses machine learning to flag suspicious transactions automatically, catching **4 out of every 5 actual fraud cases**.

**What makes this project stand out:**
- Only 5% of transactions are fraud → extreme class imbalance solved with **SMOTE**
- Financial amounts normalized using **log transformation** (np.log1p)
- Custom **probability threshold of 0.7** instead of default 0.5 → boosted fraud recall from 47% to 80%
- 6 engineered features capturing behavioral fraud patterns

---

## 📊 Dataset
| Property | Value |
|---|---|
| Rows | 10,050 |
| Columns | 20 (raw) → 26 (after engineering) |
| Fraud Rate | 5% (extreme imbalance) |
| Target | IsFraud: Yes / No |

---

## 🧹 Data Cleaning Challenges
| Column | Problem | Solution |
|---|---|---|
| TransactionAmount | NGN/₦ formats, commas, outliers ×1000 | Strip symbols, IQR clip |
| TransactionHour | "14:00", "2AM", "3 PM" — 3 formats | Custom parsing function |
| AccountAgeMonths | "24 months" and "2.0 years" mixed | Detect 'years' → ×12 |
| FailedAttempts | Values stored ×100 (700 instead of 7) | ÷100 if ≥100 |
| IsFraud | 16 different formats | str.capitalize() + dict map |

---

## ⚙️ Feature Engineering
| Feature | Formula | Insight |
|---|---|---|
| n_TransactionAmount | np.log1p(Amount) | Normalizes skewed distribution |
| n_AccountBalance | np.log1p(Balance) | Normalizes skewed distribution |
| UnusualTransactionHour | 1 if hour in [22,23,0,1,2,3,4] | Nighttime = higher fraud risk |
| RedFlag_Attempt | PrevFraudFlag + FailedAttempts | Combined behavioral risk score |
| AgeTransactionsRatio | NumTransactionsDay / (1 + AccountAgeMonths) | Unusual activity vs maturity |
| AmountToBalance_nRatio | n_Amount / n_Balance | Log-normalized drain ratio |

---

## 🤖 Models Trained
| Model | Accuracy | Fraud Recall | Notes |
|---|---|---|---|
| Logistic Regression (default) | 94.66% | 0.47 | Default threshold too lenient |
| Decision Tree | 91.14% | 0.73 | Better recall, lower accuracy |
| Random Forest | 96.15% | 0.25 | Poor fraud recall — rejected |
| **GridSearch LR (threshold=0.7)** | **97.11%** | **0.80** | **Champion ✅** |

### Custom Probability Threshold
```python
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob > 0.7).astype(int)
```
Raising threshold from 0.5 → 0.7 reduced false alarms while maintaining recall.

---

## 🧪 Class Imbalance — SMOTE
```
Before SMOTE: Clean=9,547  Fraud=503  (19:1 ratio)
After SMOTE:  Clean=9,547  Fraud=9,547 (balanced)
```
SMOTE applied to **training data only** — never on test data (data leakage prevention).

---

## 🏗️ Tech Stack
- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Imbalance:** imbalanced-learn (SMOTE)
- **ML:** Scikit-learn (LogisticRegression, GridSearchCV)
- **Web Backend:** Flask
- **Frontend:** HTML5, CSS3
- **Deployment:** Railway.app
- **Version Control:** GitHub

---

## 📁 Project Structure
```
BankFraudDetector/
├── data/
│   └── nigerian_bank_fraud_messy.csv
├── models/
│   └── fraud_model.pkl
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── app.py
├── requirements.txt
└── Procfile
```

---

## 🚀 Run Locally
```bash
git clone https://github.com/DavidGabriel213/BankFraudDetector
cd BankFraudDetector
pip install -r requirements.txt
python app.py
```

---

## 💡 Key Learnings
1. **SMOTE on training only** — applying to test set causes data leakage
2. **Custom threshold** — predict_proba gives more control than predict()
3. **For fraud: Recall > Accuracy** — missing fraud is costly
4. **LocationMatch** — coefficient 3.808, strongest fraud predictor
5. **Log transformation** — critical for skewed financial distributions

---

## 👨‍💻 About
**Gabriel David** | Mathematics Undergraduate | ATBU Bauchi
Self-taught ML Engineer — built during Industrial Training placement.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-gabriel--david--ds-blue)](https://linkedin.com/in/gabriel-david-ds)
[![GitHub](https://img.shields.io/badge/GitHub-DavidGabriel213-black)](https://github.com/DavidGabriel213)

