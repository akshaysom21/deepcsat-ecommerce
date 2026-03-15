import streamlit as st
import pandas as pd
import numpy as np
import pickle
import onnxruntime as ort
from scipy.sparse import hstack, csr_matrix
import matplotlib.pyplot as plt

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepCSAT – CSAT Predictor",
    page_icon="🛒",
    layout="wide"
)

# ── Load artefacts (cached — only loads once per session) ──────────────────
@st.cache_resource
def load_artefacts():
    model               = ort.InferenceSession("deepcsat_model.onnx")
    scaler              = pickle.load(open("scaler.pkl", "rb"))
    tfidf               = pickle.load(open("tfidf.pkl", "rb"))
    struct_feature_cols = pickle.load(open("struct_feature_cols.pkl", "rb"))
    return model, scaler, tfidf, struct_feature_cols

model, scaler, tfidf, struct_feature_cols = load_artefacts()

CLASS_LABELS = {0: "Dissatisfied (1-3)", 1: "Satisfied (4-5)"}
CLASS_COLORS = {0: "#e74c3c",            1: "#27ae60"}

# ── Prediction function ────────────────────────────────────────────────────
def predict_csat(input_dict, customer_remark=""):
    row = pd.DataFrame([input_dict])
    for col in struct_feature_cols:
        if col not in row.columns:
            row[col] = 0
    row        = row[struct_feature_cols].astype(np.float32)
    remark_vec = tfidf.transform([customer_remark])
    combined   = hstack([csr_matrix(row.values), remark_vec])
    scaled     = scaler.transform(combined).toarray().astype(np.float32)
    input_name = model.get_inputs()[0].name
    proba      = model.run(None, {input_name: scaled})[0][0]
    pred_class = int(np.argmax(proba))
    return pred_class, float(proba[0]), float(proba[1])

# ── Header ─────────────────────────────────────────────────────────────────
st.title("🛒 DeepCSAT — E-Commerce CSAT Predictor")
st.markdown(
    "Predicts whether a customer interaction will result in "
    "**Satisfied (CSAT 4-5)** or **Dissatisfied (CSAT 1-3)** feedback, "
    "using a trained Deep Learning ANN with TF-IDF text features."
)
st.divider()

# ── Two-column layout ──────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📝 Enter Interaction Details")

    customer_remark = st.text_area(
        "Customer Remark (optional but strongly recommended)",
        placeholder="e.g. Very bad service, problem not resolved at all...",
        height=120,
        help="Free-text feedback from the customer. This is the strongest signal in the model."
    )

    response_time = st.slider(
        "Response Time (minutes)",
        min_value=0, max_value=300, value=20, step=5,
        help="Minutes between issue reported and issue responded"
    )

    issue_hour = st.slider(
        "Hour of Day Issue Was Reported",
        min_value=0, max_value=23, value=10,
        help="0 = midnight, 12 = noon, 23 = 11pm"
    )

    day_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2,
        "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
    }
    issue_day = st.selectbox("Day of Week", options=list(day_map.keys()))

    st.divider()
    predict_btn = st.button("🔍  Predict CSAT", use_container_width=True, type="primary")

with col2:
    st.markdown("### 📊 Prediction Result")

    if predict_btn:
        input_dict = {
            "response_time_minutes": response_time,
            "issue_hour":            issue_hour,
            "issue_dayofweek":       day_map[issue_day]
        }

        with st.spinner("Running prediction..."):
            pred_class, prob_dis, prob_sat = predict_csat(input_dict, customer_remark)

        label      = CLASS_LABELS[pred_class]
        color      = CLASS_COLORS[pred_class]
        confidence = prob_sat if pred_class == 1 else prob_dis
        icon       = "✅" if pred_class == 1 else "❌"

        st.markdown(
            f"<div style='background:{color}18; border:2px solid {color}; "
            f"border-radius:12px; padding:24px; text-align:center;'>"
            f"<h2 style='color:{color}; margin:0;'>{icon} {label}</h2>"
            f"<p style='color:{color}; margin:10px 0 0; font-size:18px;'>"
            f"Confidence: <strong>{confidence:.1%}</strong></p>"
            f"</div>",
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("**Class Probabilities**")
        fig, ax = plt.subplots(figsize=(6, 2))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        ax.barh(
            ["Satisfied (4-5)", "Dissatisfied (1-3)"],
            [prob_sat, prob_dis],
            color=["#27ae60", "#e74c3c"],
            height=0.45, edgecolor="none"
        )
        for val, y in zip([prob_sat, prob_dis], [0, 1]):
            ax.text(min(val + 0.02, 0.93), y, f"{val:.1%}",
                    va="center", fontsize=11, fontweight="bold", color="white")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability", fontsize=10, color="white")
        ax.tick_params(colors="white")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        with st.expander("📋 View input summary"):
            st.json({
                "customer_remark":       customer_remark or "(none)",
                "response_time_minutes": response_time,
                "issue_hour":            issue_hour,
                "issue_day":             issue_day
            })
    else:
        st.info("Fill in the interaction details on the left and click **Predict CSAT**.")

# ── Business Insights ──────────────────────────────────────────────────────
st.divider()
st.markdown("### 💡 Key Business Insights")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**Dissatisfaction Rate by Channel**")
    fig, ax = plt.subplots(figsize=(4, 2.5))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.bar(
        ["Email", "Inbound", "Outcall"],
        [26.5, 17.3, 16.8],
        color=["#e74c3c", "#f39c12", "#27ae60"],
        edgecolor="none", width=0.5
    )
    ax.set_ylabel("Dissatisfied %", fontsize=9, color="white")
    ax.set_ylim(0, 35)
    ax.tick_params(colors="white", labelsize=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("white")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with c2:
    st.markdown("**Avg CSAT by Agent Shift**")
    fig, ax = plt.subplots(figsize=(4, 2.5))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    shifts = ["Split", "Afternoon", "Night", "Evening", "Morning"]
    scores = [4.43, 4.29, 4.29, 4.28, 4.19]
    colors_sh = [
        "#27ae60" if s == "Split" else "#e74c3c" if s == "Morning" else "#3498db"
        for s in shifts
    ]
    ax.barh(shifts, scores, color=colors_sh, edgecolor="none", height=0.45)
    ax.set_xlim(4.1, 4.55)
    ax.set_xlabel("Avg CSAT", fontsize=9, color="white")
    ax.tick_params(colors="white", labelsize=8)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_color("white")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with c3:
    st.markdown("**Worst Sub-categories**")
    fig, ax = plt.subplots(figsize=(4, 2.5))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    subs   = ["Commission\nrelated", "Unable\nto Login", "Service\nDenial",
              "Call\nDisconn.", "Technician\nVisit"]
    scores2 = [2.33, 2.43, 3.22, 3.23, 3.49]
    ax.bar(subs, scores2, color="#e74c3c", edgecolor="none", width=0.5)
    ax.axhline(y=4.24, color="gray", linestyle="--", linewidth=1, label="Overall avg 4.24")
    ax.set_ylim(0, 5)
    ax.set_ylabel("Avg CSAT", fontsize=9, color="white")
    ax.tick_params(colors="white", labelsize=7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("white")
    legend = ax.legend(fontsize=8)
    for text in legend.get_texts():
        text.set_color("white")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Model metrics ──────────────────────────────────────────────────────────
st.divider()
st.markdown("### 🏆 Final Model Performance")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Test Accuracy",       "72.90%", "+43.6pp vs baseline")
m2.metric("F1-Score (Macro)",    "0.637",  "+0.444 vs baseline")
m3.metric("ROC-AUC",             "0.768",  "Binary classifier")
m4.metric("Dissatisfied Recall", "64%",    "Catches unhappy customers")

# ── Footer ─────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "DeepCSAT | 4-layer ANN | Binary Classification | "
    "TF-IDF (200 features) + 135 structural features | "
    "compute_class_weight balancing | Dataset: Shopzilla 85,907 records"
)
