# 🛒 DeepCSAT — E-Commerce Customer Satisfaction Predictor

A Deep Learning ANN model that predicts whether a customer interaction will
result in **Satisfied (CSAT 4-5)** or **Dissatisfied (CSAT 1-3)** feedback,
built on 85,907 e-commerce service interaction records from the Shopzilla platform.

🔗 **Live App:** [deepcsat-ecommerce.streamlit.app](https://deepcsat-ecommerce-aks21.streamlit.app/)

---

## 📌 Project Overview

Customer satisfaction (CSAT) scores are a critical metric for e-commerce
businesses. This project builds a binary classifier using a Deep Learning
Artificial Neural Network that combines:

- **Structural features** — response time, agent shift, tenure, category, channel
- **TF-IDF text features** — extracted from free-text Customer Remarks (200 features, bigrams)

The model was developed and refined across 5 versions, improving accuracy
from **29.3% → 72.9%** and F1-Macro from **0.193 → 0.637**.

---

## 📊 Final Model Performance

| Metric | Score |
|---|---|
| Test Accuracy | 72.90% |
| F1-Score (Macro) | 0.637 |
| F1-Score (Weighted) | 0.756 |
| ROC-AUC | 0.768 |
| Dissatisfied Recall | 64% |
| Satisfied Precision | 91% |

---

## 🗂️ Repository Structure
```
deepcsat-ecommerce/
│
├── app.py                        # Streamlit web app
├── requirements.txt              # Python dependencies
├── DeepCSAT_Final_Complete.ipynb # Full project notebook (Colab)
│
├── deepcsat_model.keras          # Trained ANN model
├── scaler.pkl                    # MaxAbsScaler fitted on training data
├── tfidf.pkl                     # TF-IDF vectoriser (200 features)
└── struct_feature_cols.pkl       # Feature column names for inference
```

---

## 🧠 Model Architecture
```
Input (335 features: 135 structural + 200 TF-IDF)
    ↓
Dense(256) + BatchNorm + ReLU + Dropout(0.15)  ← L2(0.0001)
    ↓
Dense(128) + BatchNorm + ReLU + Dropout(0.15)
    ↓
Dense(64)  + BatchNorm + ReLU + Dropout(0.10)
    ↓
Dense(32)  + ReLU
    ↓
Dense(2, softmax)  →  [P(Dissatisfied), P(Satisfied)]
```

- **Optimiser:** Adam (LR = 0.0005)
- **Loss:** Sparse Categorical Crossentropy
- **Imbalance handling:** `compute_class_weight('balanced')` — no SMOTE
- **Early stopping:** patience=10, restore best weights (stopped at Epoch 20, best Epoch 10)

---

## 🔧 Tech Stack

| Component | Library / Tool |
|---|---|
| Model | TensorFlow / Keras |
| Text features | scikit-learn TF-IDF |
| Preprocessing | pandas, NumPy, scipy |
| Imbalance | sklearn `compute_class_weight` |
| Web app | Streamlit |
| Hosting | Streamlit Cloud |
| Notebook | Google Colab |

---

## 📁 Dataset

- **Source:** Shopzilla e-commerce platform (provided for academic use)
- **Records:** 85,907
- **Features:** 20 (timestamps, agent details, category, channel, free-text remarks)
- **Target:** CSAT Score (1–5), remapped to binary — Dissatisfied (1–3) / Satisfied (4–5)
- **Class split:** 82.5% Satisfied / 17.5% Dissatisfied

**Key missing data:**
- `connected_handling_time` — 99.7% missing (dropped)
- `Customer Remarks` — 66.5% missing (TF-IDF applied; missing rows get zero vectors)
- `Customer_City`, `Agent_name` — 80%+ missing or high cardinality (dropped)

---

## 🚀 Running Locally
```bash
# 1. Clone the repo
git clone https://github.com/your-username/deepcsat-ecommerce.git
cd deepcsat-ecommerce

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

---

## ☁️ Deploying on Streamlit Cloud

1. Fork or push this repo to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select this repo → set main file to `app.py`
4. Click **Deploy** — live in ~2 minutes

---

## 💡 Key Business Findings

| Finding | Detail |
|---|---|
| Strongest signal | Customer Remarks text (99.97% confidence on clear sentiment) |
| Worst channel | Email — 26.5% dissatisfaction rate vs 16.8% for Outcall |
| Worst sub-categories | Commission related (avg 2.33) and Unable to Login (avg 2.43) |
| Best shift | Split shift agents (avg CSAT 4.43) |
| Worst shift | Morning shift agents (avg CSAT 4.19) |

---

## 🔄 Version History

| Version | Change | Accuracy | F1-Macro |
|---|---|---|---|
| v1 | Original 5-class ANN | 29.3% | 0.193 |
| v2 | 3-class + TF-IDF + SMOTE | 66.2% | 0.431 |
| v3 | Binary classification | 75.7% | 0.647 |
| v4 | Partial SMOTE (0.5 ratio) | 72.4% | 0.632 |
| **v5** | **No SMOTE + compute_class_weight** | **72.9%** | **0.637** |

---

## ⚠️ Limitations

- Customer Remarks are missing for 66.5% of records — the strongest feature is absent for most predictions
- Dissatisfied precision is 0.35 — high false positive rate is a deliberate tradeoff to maximise recall on unhappy customers
- The Streamlit app session loads the model fresh each time; inference takes ~1–2 seconds on first prediction

---

## 🔮 Future Improvements

- BERT or Sentence Transformer embeddings for richer text features
- Train a separate high-performance model only on the ~28,742 records with remarks
- NLP preprocessing — stopword removal, lemmatisation on Customer Remarks
- Add agent-level historical CSAT as a feature

---

## 👤 Author

**[AKSHAY SOM]**
[https://www.linkedin.com/in/akshaysom21/] | [akshaysom21@gmail.com]

---

## 📄 License

This project is for academic and educational purposes.
Dataset provided by course instructor for the DeepCSAT project.
