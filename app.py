# app.py — Churn Predictor Pro
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import csv as csv_mod
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

import auth

# ═══════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ChurnIQ",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
.login-box {
    max-width: 420px; margin: 40px auto;
    background: #1e293b; border: 1px solid #334155;
    border-radius: 16px; padding: 36px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
}
.login-title { font-size: 1.8rem; font-weight: 700; color: #38bdf8; margin-bottom: 4px; }
.login-sub   { color: #64748b; font-size: .9rem; margin-bottom: 24px; }
.badge-high   { color: #f87171; font-size: 1.6rem; font-weight: 800; }
.badge-medium { color: #fbbf24; font-size: 1.6rem; font-weight: 800; }
.badge-low    { color: #34d399; font-size: 1.6rem; font-weight: 800; }
.sec-hdr { font-size: .72rem; font-weight: 700; color: #475569;
    letter-spacing: 2px; text-transform: uppercase; margin-bottom: 6px; }
.role-admin { background:#7c3aed22; color:#a78bfa;
    border:1px solid #7c3aed; border-radius:20px; padding:2px 10px;
    font-size:.7rem; font-weight:700; letter-spacing:1px; }
.role-user  { background:#0ea5e922; color:#38bdf8;
    border:1px solid #0ea5e9; border-radius:20px; padding:2px 10px;
    font-size:.7rem; font-weight:700; letter-spacing:1px; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════
LOG_FILE = "prediction_log.csv"

CAT_OPTIONS = {
    "gender":           ["Female", "Male"],
    "SeniorCitizen":    ["No", "Yes"],
    "Partner":          ["No", "Yes"],
    "Dependents":       ["No", "Yes"],
    "PhoneService":     ["No", "Yes"],
    "MultipleLines":    ["No", "No phone service", "Yes"],
    "InternetService":  ["DSL", "Fiber optic", "No"],
    "OnlineSecurity":   ["No", "No internet service", "Yes"],
    "OnlineBackup":     ["No", "No internet service", "Yes"],
    "DeviceProtection": ["No", "No internet service", "Yes"],
    "TechSupport":      ["No", "No internet service", "Yes"],
    "StreamingTV":      ["No", "No internet service", "Yes"],
    "StreamingMovies":  ["No", "No internet service", "Yes"],
    "Contract":         ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod":    ["Bank transfer (automatic)", "Credit card (automatic)",
                         "Electronic check", "Mailed check"],
}

# ═══════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════
for key, val in [("logged_in", False), ("username", ""),
                 ("role", ""), ("display_name", "")]:
    if key not in st.session_state:
        st.session_state[key] = val

# ═══════════════════════════════════════════════════════════
# LOAD ARTIFACTS
# ═══════════════════════════════════════════════════════════
@st.cache_resource
def load_artifacts():
    if not os.path.exists("model.pkl"):
        return None, None, None, None, None, None
    model     = joblib.load("model.pkl")
    features  = joblib.load("features.pkl")
    encoders  = joblib.load("encoders.pkl")
    imp       = joblib.load("feature_importance.pkl") if os.path.exists("feature_importance.pkl") else None
    meta      = json.load(open("model_metadata.json")) if os.path.exists("model_metadata.json") else {}
    try:
        lime_exp = joblib.load("lime_explainer.pkl") if os.path.exists("lime_explainer.pkl") else None
    except Exception:
        lime_exp = None
    return model, features, encoders, imp, meta, lime_exp

model, features, encoders, importance, meta, lime_explainer = load_artifacts()

# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════
def preprocess(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    if "tenure" in df.columns and "TotalCharges" in df.columns:
        df["ChargePerTenure"] = np.where(
            df["tenure"] > 0,
            df["TotalCharges"] / df["tenure"],
            df["MonthlyCharges"]
        )
    if "Contract" in df.columns:
        df["LongTermContract"] = df["Contract"].apply(
            lambda x: 1 if x in ["One year", "Two year"] else 0
        )
    for col in df.columns:
        if col in encoders:
            try:
                df[col] = encoders[col].transform(df[col].astype(str))
            except Exception:
                df[col] = 0
    for f in features:
        if f not in df.columns:
            df[f] = 0
    return df[features]


def risk_label(prob: float):
    if prob >= 0.60:
        return "🔴 HIGH RISK", "badge-high"
    elif prob >= 0.30:
        return "🟡 MEDIUM RISK", "badge-medium"
    return "🟢 LOW RISK", "badge-low"


def make_gauge(prob: float):
    color = "#f87171" if prob >= 0.60 else ("#fbbf24" if prob >= 0.30 else "#34d399")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 34, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "bgcolor": "#1e293b",
            "bordercolor": "#334155",
            "steps": [
                {"range": [0,  30], "color": "#052e16"},
                {"range": [30, 60], "color": "#451a03"},
                {"range": [60, 100], "color": "#3b0764"},
            ],
        },
        title={"text": "Churn Probability", "font": {"size": 14, "color": "#94a3b8"}},
    ))
    fig.update_layout(
        height=240, margin=dict(t=50, b=0, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
    )
    return fig


def log_prediction(username, input_dict, prob, pred):
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user": username,
        "prediction": pred,
        "probability": round(prob, 4),
        "contract": input_dict.get("Contract", ""),
        "tenure": input_dict.get("tenure", ""),
        "monthly_charges": input_dict.get("MonthlyCharges", ""),
        "internet": input_dict.get("InternetService", ""),
    }
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_log():
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()
    try:
        return pd.read_csv(LOG_FILE)
    except Exception:
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════
# LOGIN PAGE
# ═══════════════════════════════════════════════════════════
def show_login():
    _, center, _ = st.columns([1, 1.2, 1])
    with center:
        st.markdown("""
        <div class="login-box">
            <div class="login-title">📡 ChurnIQ</div>
            <div class="login-sub">Customer Intelligence Platform</div>
        </div>
        """, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("### 🔐 Sign In")
            username = st.text_input("Username", placeholder="e.g. admin")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            login_btn = st.button("Sign In", use_container_width=True, type="primary")

            if login_btn:
                if not username or not password:
                    st.error("Enter both username and password.")
                else:
                    user = auth.verify_login(username, password)
                    if user:
                        st.session_state.logged_in    = True
                        st.session_state.username     = username
                        st.session_state.role         = user["role"]
                        st.session_state.display_name = user["name"]
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")

        st.caption("Default → Admin: `admin` / `admin123` | User: `user1` / `user123`")


# ═══════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("## 📡 ChurnIQ")
        role_badge = "role-admin" if st.session_state.role == "admin" else "role-user"
        st.markdown(f"""
        **{st.session_state.display_name}**  
        @{st.session_state.username}  
        <span class="{role_badge}">{st.session_state.role.upper()}</span>
        """, unsafe_allow_html=True)

        st.markdown("---")
        if model and meta:
            st.markdown(f"**Model:** {meta.get('model_name','—')}")
            st.markdown(f"**AUC:** `{meta.get('test_auc',0):.4f}`")
            st.markdown(f"**F1:** `{meta.get('test_f1',0):.4f}`")
            st.markdown(f"**Accuracy:** `{meta.get('test_accuracy',0):.1%}`")

        st.markdown("---")
        st.markdown("🟢 Low Risk → below 30%")
        st.markdown("🟡 Medium Risk → 30–60%")
        st.markdown("🔴 High Risk → above 60%")

        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            for k in ["logged_in", "username", "role", "display_name"]:
                st.session_state[k] = False if k == "logged_in" else ""
            st.rerun()


# ═══════════════════════════════════════════════════════════
# PREDICT TAB
# ═══════════════════════════════════════════════════════════
def render_predict():
    if model is None:
        st.error("Run `python train.py` first!")
        return

    st.subheader("🔍 Single Customer Prediction")

    with st.form("pred_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown('<p class="sec-hdr">Demographics</p>', unsafe_allow_html=True)
            gender     = st.selectbox("Gender",         CAT_OPTIONS["gender"])
            senior     = st.selectbox("Senior Citizen", CAT_OPTIONS["SeniorCitizen"])
            partner    = st.selectbox("Partner",        CAT_OPTIONS["Partner"])
            dependents = st.selectbox("Dependents",     CAT_OPTIONS["Dependents"])
            tenure     = st.slider("Tenure (months)", 0, 72, 12)

        with c2:
            st.markdown('<p class="sec-hdr">Services</p>', unsafe_allow_html=True)
            phone    = st.selectbox("Phone Service",     CAT_OPTIONS["PhoneService"])
            lines    = st.selectbox("Multiple Lines",    CAT_OPTIONS["MultipleLines"])
            internet = st.selectbox("Internet Service",  CAT_OPTIONS["InternetService"])
            sec_     = st.selectbox("Online Security",   CAT_OPTIONS["OnlineSecurity"])
            backup   = st.selectbox("Online Backup",     CAT_OPTIONS["OnlineBackup"])
            device   = st.selectbox("Device Protection", CAT_OPTIONS["DeviceProtection"])
            tech     = st.selectbox("Tech Support",      CAT_OPTIONS["TechSupport"])
            tv       = st.selectbox("Streaming TV",      CAT_OPTIONS["StreamingTV"])
            movies   = st.selectbox("Streaming Movies",  CAT_OPTIONS["StreamingMovies"])

        with c3:
            st.markdown('<p class="sec-hdr">Billing</p>', unsafe_allow_html=True)
            contract  = st.selectbox("Contract",          CAT_OPTIONS["Contract"])
            paperless = st.selectbox("Paperless Billing", CAT_OPTIONS["PaperlessBilling"])
            payment   = st.selectbox("Payment Method",    CAT_OPTIONS["PaymentMethod"])
            monthly   = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5)
            total     = st.number_input("Total Charges ($)", 0.0, 10000.0,
                                        float(monthly * max(tenure, 1)), step=1.0)

        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True, type="primary")

    if submitted:
        raw = {
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": tenure,
            "PhoneService": phone, "MultipleLines": lines,
            "InternetService": internet, "OnlineSecurity": sec_,
            "OnlineBackup": backup, "DeviceProtection": device,
            "TechSupport": tech, "StreamingTV": tv, "StreamingMovies": movies,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total,
        }

        processed = preprocess(pd.DataFrame([raw]))
        prob = model.predict_proba(processed)[0][1]
        pred = "Churn" if prob >= 0.5 else "Stay"
        label, badge_cls = risk_label(prob)
        log_prediction(st.session_state.username, raw, prob, pred)

        left, right = st.columns(2)
        with left:
            st.plotly_chart(make_gauge(prob), use_container_width=True)
        with right:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown(f'<p class="{badge_cls}">{label}</p>', unsafe_allow_html=True)
            st.markdown(f"**Probability:** `{prob:.1%}`")
            if prob >= 0.60:
                st.error("⚠️ High risk — take immediate retention action.")
            elif prob >= 0.30:
                st.warning("💡 Moderate risk — consider proactive outreach.")
            else:
                st.success("✅ Customer likely to stay.")

            st.markdown("**Suggested Actions**")
            if contract == "Month-to-month":
                st.markdown("- Offer discounted 1 or 2-year contract")
            if internet == "Fiber optic" and sec_ == "No":
                st.markdown("- Bundle Online Security add-on")
            if tenure < 12:
                st.markdown("- Assign dedicated account manager")
            if payment == "Electronic check":
                st.markdown("- Encourage switch to automatic payment")

        # ── LIME Explanation ──────────────────────────────
        if lime_explainer is not None:
            st.markdown("---")
            st.subheader("🧠 Why This Prediction? (LIME Explanation)")
            st.caption("Red = pushes toward Churn | Green = pushes toward Stay")
            try:
                import lime.lime_tabular
                exp = lime_explainer.explain_instance(
                    processed.values[0],
                    model.predict_proba,
                    num_features=10
                )
                lime_df = pd.DataFrame(exp.as_list(), columns=["Feature Rule", "Impact"])
                lime_df["Color"] = lime_df["Impact"].apply(
                    lambda v: "#f87171" if v > 0 else "#34d399"
                )
                fig_lime = go.Figure(go.Bar(
                    x=lime_df["Impact"],
                    y=lime_df["Feature Rule"],
                    orientation="h",
                    marker_color=lime_df["Color"],
                ))
                fig_lime.update_layout(
                    title="Feature Contributions to This Prediction",
                    yaxis=dict(autorange="reversed"),
                    height=400,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0",
                )
                st.plotly_chart(fig_lime, use_container_width=True)
            except Exception as e:
                st.info(f"LIME explanation unavailable: {e}")
        else:
            st.info("Run `python -m pip install lime` then retrain to see explanations.")


# ═══════════════════════════════════════════════════════════
# BATCH TAB
# ═══════════════════════════════════════════════════════════
def render_batch():
    st.subheader("📂 Batch Churn Prediction")
    st.info("Upload a CSV with the same columns as the training data.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if not uploaded:
        return

    raw_batch = pd.read_csv(uploaded)
    st.caption(f"Loaded {len(raw_batch):,} rows")

    id_col = None
    if "customerID" in raw_batch.columns:
        id_col = raw_batch["customerID"].reset_index(drop=True)

    proc = preprocess(raw_batch.drop(columns=["customerID", "Churn"], errors="ignore"))
    probs = model.predict_proba(proc)[:, 1]
    preds = ["Churn" if p >= 0.5 else "Stay" for p in probs]
    risk  = ["High" if p >= 0.6 else ("Medium" if p >= 0.3 else "Low") for p in probs]

    out = pd.DataFrame({
        "Churn Probability (%)": (probs * 100).round(2),
        "Prediction": preds,
        "Risk Level": risk,
    })
    if id_col is not None:
        out.insert(0, "customerID", id_col)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total", f"{len(out):,}")
    k2.metric("Predicted Churn", f"{(out['Prediction']=='Churn').sum():,}")
    k3.metric("Churn Rate", f"{(out['Prediction']=='Churn').mean():.1%}")
    k4.metric("Avg Probability", f"{probs.mean():.1%}")

    risk_counts = out["Risk Level"].value_counts().reset_index()
    risk_counts.columns = ["Risk", "Count"]
    fig = px.pie(
        risk_counts, names="Risk", values="Count", hole=0.45,
        color="Risk",
        color_discrete_map={"High":"#f87171","Medium":"#fbbf24","Low":"#34d399"},
        title="Risk Distribution"
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(out, use_container_width=True)
    st.download_button(
        "⬇️ Download Results as CSV",
        out.to_csv(index=False).encode(),
        file_name=f"churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ═══════════════════════════════════════════════════════════
# INSIGHTS TAB
# ═══════════════════════════════════════════════════════════
def render_insights():
    st.subheader("📈 Model Insights")

    if importance is not None:
        top_n = st.slider("Show top N features", 5, len(importance), 10)
        fig = px.bar(
            importance.head(top_n), x="Importance", y="Feature",
            orientation="h", color="Importance",
            color_continuous_scale="Blues", title="Feature Importance",
        )
        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 💡 Key Findings")
    findings = [
        ("📋 Contract Type",    "Month-to-month customers churn **2–3× more** than annual contracts."),
        ("💳 Payment Method",   "Electronic check users have the highest churn rate (~45%)."),
        ("🌐 Fiber Optic",      "Fiber optic customers churn more due to higher monthly costs."),
        ("👴 Senior Citizens",  "Senior citizens are ~40% more likely to churn."),
        ("📅 Tenure",           "First 12 months = highest risk. Focus retention efforts here."),
        ("🔒 Add-On Services",  "Customers with security & backup add-ons churn significantly less."),
    ]
    for title, body in findings:
        with st.expander(title):
            st.markdown(body)

    st.markdown("---")
    st.markdown("### 📌 Retention Strategy")
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown("**🎯 Target These Customers**")
        st.markdown("- Month-to-month + Fiber optic\n- Electronic check payers\n- Tenure under 6 months")
    with r2:
        st.markdown("**🛡️ Reduce Churn**")
        st.markdown("- Offer loyalty discounts at 6 & 12 months\n- Bundle security add-ons\n- Account managers for new customers")
    with r3:
        st.markdown("**💬 Engagement**")
        st.markdown("- Personalized upgrade offers\n- Satisfaction surveys\n- Annual plan switch incentives")


# ═══════════════════════════════════════════════════════════
# HISTORY TAB
# ═══════════════════════════════════════════════════════════
def render_history(show_all=False):
    st.subheader("📋 Prediction History")
    log_df = load_log()

    if log_df.empty:
        st.info("No predictions logged yet.")
        return

    if not show_all:
        log_df = log_df[log_df["user"] == st.session_state.username]

    if log_df.empty:
        st.info("No predictions found for your account.")
        return

    st.dataframe(log_df.sort_values("timestamp", ascending=False), use_container_width=True)

    if show_all:
        st.download_button(
            "⬇️ Download Full Log",
            log_df.to_csv(index=False).encode(),
            file_name="full_prediction_log.csv",
            mime="text/csv",
        )


# ═══════════════════════════════════════════════════════════
# ADMIN PANEL
# ═══════════════════════════════════════════════════════════
def render_admin():
    st.subheader("⚙️ Admin Panel")
    a1, a2, a3, a4 = st.tabs(
        ["👥 Users", "📊 Model Comparison", "📋 All Predictions", "🔑 Change Password"]
    )

    with a1:
        st.markdown("#### Current Users")
        users = auth.get_all_users()
        rows = [{"Username": u, "Name": d["name"], "Role": d["role"],
                 "Created": d.get("created_at","")[:10]}
                for u, d in users.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.markdown("---")
        st.markdown("#### Add New User")
        with st.form("add_user"):
            col1, col2 = st.columns(2)
            with col1:
                nu = st.text_input("Username")
                nn = st.text_input("Display Name")
            with col2:
                np_ = st.text_input("Password", type="password")
                nr  = st.selectbox("Role", ["user", "admin"])
            if st.form_submit_button("➕ Add User", type="primary"):
                if nu and np_ and nn:
                    if auth.add_user(nu, np_, nn, nr):
                        st.success(f"User '{nu}' added!")
                        st.rerun()
                    else:
                        st.error("Username already exists.")
                else:
                    st.warning("Fill all fields.")

        st.markdown("---")
        st.markdown("#### Delete User")
        del_options = [u for u in users.keys() if u != "admin"]
        if del_options:
            del_user = st.selectbox("Select user to delete", del_options)
            if st.button("🗑️ Delete User"):
                if auth.delete_user(del_user):
                    st.success(f"Deleted '{del_user}'.")
                    st.rerun()
        else:
            st.info("No deletable users.")

    with a2:
        if meta and "all_model_results" in meta:
            rows = [{"Model": k, "Accuracy": v["test_acc"], "ROC-AUC": v["test_auc"],
                     "F1 Score": v["test_f1"], "CV AUC": v["cv_auc"]}
                    for k, v in meta["all_model_results"].items()]
            comp_df = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
            st.dataframe(
                comp_df.style
                    .highlight_max(subset=["Accuracy","ROC-AUC","F1 Score","CV AUC"], color="#14532d")
                    .format({c: "{:.4f}" for c in ["Accuracy","ROC-AUC","F1 Score","CV AUC"]}),
                use_container_width=True,
            )
            fig = px.bar(
                comp_df.melt(id_vars="Model", value_vars=["Accuracy","ROC-AUC","F1 Score"]),
                x="Model", y="value", color="variable", barmode="group",
                title="Model Comparison",
                color_discrete_sequence=["#38bdf8","#fbbf24","#34d399"],
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", yaxis_range=[0,1]
            )
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"Best model: **{meta.get('model_name','—')}**")
        else:
            st.info("Run `python train.py` first.")

    with a3:
        render_history(show_all=True)

    with a4:
        st.markdown("#### Change Password")
        users = auth.get_all_users()
        with st.form("change_pwd"):
            target = st.selectbox("User", list(users.keys()))
            new_pw  = st.text_input("New Password", type="password")
            confirm = st.text_input("Confirm Password", type="password")
            if st.form_submit_button("Update Password", type="primary"):
                if new_pw != confirm:
                    st.error("Passwords don't match.")
                elif len(new_pw) < 6:
                    st.warning("Password must be at least 6 characters.")
                else:
                    auth.change_password(target, new_pw)
                    st.success(f"Password updated for '{target}'.")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    show_login()
else:
    render_sidebar()
    st.title("📡 ChurnIQ — Customer Intelligence Platform")

    if st.session_state.role == "admin":
        tabs = st.tabs(["🔍 Predict", "📂 Batch", "📈 Insights", "📋 My History", "⚙️ Admin"])
        with tabs[0]: render_predict()
        with tabs[1]: render_batch()
        with tabs[2]: render_insights()
        with tabs[3]: render_history(show_all=False)
        with tabs[4]: render_admin()
    else:
        tabs = st.tabs(["🔍 Predict", "📂 Batch", "📈 Insights", "📋 My History"])
        with tabs[0]: render_predict()
        with tabs[1]: render_batch()
        with tabs[2]: render_insights()
        with tabs[3]: render_history(show_all=False)
