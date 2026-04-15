"""
FDS — Fraud Detection System
Streamlit Dashboard · Berbasis layout PADI Dashboard

Instalasi:
    pip install streamlit plotly pandas scikit-learn xgboost shap

Jalankan:
    streamlit run fds_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ─────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="FDS — Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
)

# ─────────────────────────────────────────
#  GLOBAL CSS  (meniru warna PADI dashboard)
# ─────────────────────────────────────────
st.markdown("""
<style>
/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #1a3c5e !important;
}
[data-testid="stSidebar"] .stRadio > label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: rgba(255,255,255,0.75) !important;
    font-size: 13px;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.1);
}

/* ── Hide default streamlit elements ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── Cards ── */
.card {
    background: #ffffff;
    border-radius: 10px;
    border: 1px solid #e2e8f0;
    padding: 20px 24px;
    margin-bottom: 8px;
}

/* ── Metric tiles ── */
.metric-tile {
    background: #ffffff;
    border-radius: 10px;
    border: 1px solid #e2e8f0;
    padding: 16px 20px;
    text-align: left;
}
.metric-label {
    font-size: 11px;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 28px;
    font-weight: 700;
    margin: 0;
}

/* ── Badges ── */
.badge-green  { background:#d1fae5; color:#065f46; padding:4px 14px; border-radius:20px; font-weight:700; font-size:13px; display:inline-block; }
.badge-red    { background:#fee2e2; color:#991b1b; padding:4px 14px; border-radius:20px; font-weight:700; font-size:13px; display:inline-block; }
.badge-yellow { background:#fef9c3; color:#854d0e; padding:4px 14px; border-radius:20px; font-weight:700; font-size:13px; display:inline-block; }
.badge-orange { background:#ffedd5; color:#9a3412; padding:4px 14px; border-radius:20px; font-weight:700; font-size:13px; display:inline-block; }

/* ── Alert boxes ── */
.alert-red    { background:#fef2f2; border:1px solid #fecaca; border-left:4px solid #ef4444; border-radius:8px; padding:14px 16px; margin-bottom:10px; }
.alert-yellow { background:#fffbeb; border:1px solid #fde68a; border-left:4px solid #f59e0b; border-radius:8px; padding:14px 16px; margin-bottom:10px; }
.alert-green  { background:#f0fdf4; border:1px solid #bbf7d0; border-left:4px solid #10b981; border-radius:8px; padding:14px 16px; margin-bottom:10px; }

/* ── Info box ── */
.info-box {
    background:#eff6ff;
    border-left:3px solid #0ea5e9;
    border-radius:6px;
    padding:10px 14px;
    font-size:12px;
    color:#1e40af;
    margin-top:12px;
}

/* ── Section header ── */
.section-title { font-size:15px; font-weight:700; color:#1e293b; margin-bottom:4px; }
.section-sub   { font-size:12px; color:#64748b; margin-bottom:16px; }

/* ── Table styling ── */
.styled-table { width:100%; border-collapse:collapse; font-size:12px; }
.styled-table th { background:#f0f4f8; padding:8px 10px; text-align:left; color:#64748b; font-weight:600; font-size:11px; }
.styled-table td { padding:9px 10px; border-top:1px solid #e2e8f0; }
.styled-table tr.anomaly { background:#fff5f5; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  KONSTANTA FITUR
# ─────────────────────────────────────────
NUMERICAL_FEATURES = [
    "hour_of_day", "transaction_amount", "credit_limit", "credit_utilization",
    "amount_to_avg_ratio", "avg_txn_amount_30d", "customer_age", "account_age_days",
    "txn_count_30d", "txn_count_24h", "unique_merchant_30d", "failed_auth_24h",
    "device_change_7d", "time_since_last_txn_hr", "distance_from_home_km",
    "velocity_score", "chargeback_count_12m", "is_new_device", "is_foreign_ip",
    "is_high_risk_merchant", "is_weekend", "is_odd_hour", "flag_high_amount",
    "flag_high_velocity", "behavioral_signal_exp",
]

NOMINAL_FEATURES = {
    "payment_channel":       ["EDC / POS", "Mobile App", "Web", "ATM", "Transfer Bank"],
    "merchant_category":     ["Retail", "E-Commerce", "F&B", "Jewelry", "Gambling", "Travel", "Other"],
    "region":                ["Jawa", "Kalimantan", "Sumatera", "Sulawesi", "Bali & Nusa Tenggara", "Other"],
    "customer_segment":      ["Mass", "Private", "Mass Affluent", "Priority", "Affluent"],
    "card_type":             ["Kredit Gold", "Kredit Platinum", "Debit", "Virtual", "Prepaid"],
    "device_type":           ["iOS", "Android", "Web Browser", "Unknown"],
    "authentication_method": ["PIN", "Biometric", "OTP", "3DS", "No Auth"],
    "ip_country":            ["ID", "SG", "CN", "MY", "US", "Unknown"],
    "transaction_type":      ["Withdrawal", "Top Up", "Transfer", "Bill Payment", "Purchase", "Other"],
}

FEATURE_BOUNDS = {
    "hour_of_day":           (0,    23),
    "transaction_amount":    (10_000,  50_000_000),
    "credit_limit":          (5_000_000, 100_000_000),
    "credit_utilization":    (0.01, 0.95),
    "amount_to_avg_ratio":   (0.01, 318.83),
    "avg_txn_amount_30d":    (12_963, 5_344_274),
    "customer_age":          (18,   69),
    "account_age_days":      (1,    3649),
    "txn_count_30d":         (1,    79),
    "txn_count_24h":         (0,    9),
    "unique_merchant_30d":   (1,    29),
    "failed_auth_24h":       (0,    5),
    "device_change_7d":      (0,    3),
    "time_since_last_txn_hr":(0.04, 281.52),
    "distance_from_home_km": (0,    205.6),
    "velocity_score":        (0,    99.6),
    "chargeback_count_12m":  (0,    3),
    "behavioral_signal_exp": (0.004312, 0.998232),
}

BINARY_FEATURES = ["is_new_device","is_foreign_ip","is_high_risk_merchant",
                   "is_weekend","is_odd_hour","flag_high_amount","flag_high_velocity"]

# ─────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        with open("fraud_detection_pipeline.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

pipeline = load_model()

# ─────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────
def risk_label(prob: float):
    if prob < 0.20:
        return "Rendah", "#10b981", "badge-green"
    elif prob < 0.50:
        return "Sedang", "#f59e0b", "badge-yellow"
    elif prob < 0.75:
        return "Tinggi", "#ef4444", "badge-red"
    else:
        return "Sangat Tinggi", "#7f1d1d", "badge-orange"

def score_from_prob(prob: float) -> int:
    """Konversi fraud probability ke credit-style score (300–850, makin tinggi makin aman)."""
    return int(850 - prob * 550)

def gauge_chart(prob: float):
    score = score_from_prob(prob)
    label, color, _ = risk_label(prob)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 34, "color": color}},
        title={"text": "Fraud Probability", "font": {"size": 13, "color": "#64748b"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#e2e8f0"},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  20], "color": "#d1fae5"},
                {"range": [20, 50], "color": "#fef9c3"},
                {"range": [50, 75], "color": "#fee2e2"},
                {"range": [75,100], "color": "#fecaca"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75,
                "value": prob * 100,
            },
        },
    ))
    fig.update_layout(height=230, margin=dict(t=30, b=0, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig

def shap_chart(shap_values, feature_names, n=12):
    vals = list(zip(feature_names, shap_values))
    vals.sort(key=lambda x: abs(x[1]), reverse=True)
    vals = vals[:n]
    vals.sort(key=lambda x: x[1])
    feats = [v[0] for v in vals]
    svs   = [v[1] for v in vals]
    colors = ["#ef4444" if v < 0 else "#10b981" for v in svs]

    fig = go.Figure(go.Bar(
        x=svs, y=feats,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in svs],
        textposition="outside",
    ))
    fig.add_vline(x=0, line_color="#e2e8f0")
    fig.update_layout(
        height=380,
        margin=dict(t=10, b=10, l=10, r=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#f0f4f8"),
        yaxis=dict(tickfont=dict(size=11)),
        showlegend=False,
    )
    return fig

def build_input_df(inputs: dict) -> pd.DataFrame:
    return pd.DataFrame([inputs])

# ─────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ FDS")
    st.markdown("<p style='font-size:11px;color:rgba(255,255,255,0.4);margin-top:-10px;'>Fraud Detection System</p>", unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigasi",
        ["📋  Input Transaksi", "📊  Hasil Deteksi", "🔔  Early Warning"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<p style='font-size:10px;color:rgba(255,255,255,0.3);'>v1.0.0 · FDS Dashboard</p>",
        unsafe_allow_html=True,
    )

    if pipeline is None:
        st.warning("⚠️ Model tidak ditemukan.\nLetakkan `fraud_detection_pipeline.pkl` di direktori yang sama.")

now_str = datetime.now().strftime("%d %B %Y · %H:%M WIB")

# ═══════════════════════════════════════════
#  PAGE 1 — INPUT TRANSAKSI
# ═══════════════════════════════════════════
if "📋  Input Transaksi" in page:
    st.markdown(f"""
    <div style='display:flex;justify-content:space-between;align-items:flex-end;
                border-bottom:1px solid #e2e8f0;padding-bottom:16px;margin-bottom:24px;'>
        <div>
            <div style='font-size:20px;font-weight:700;color:#1e293b;'>Form Input Transaksi</div>
            <div style='font-size:13px;color:#64748b;margin-top:3px;'>Masukkan data transaksi untuk deteksi fraud real-time</div>
        </div>
        <div style='font-size:12px;color:#64748b;background:#f0f4f8;padding:6px 12px;border-radius:6px;border:1px solid #e2e8f0;'>{now_str}</div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("input_form"):
        # ── Row 1 ──
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='section-title'>💳 Detail Transaksi</div>", unsafe_allow_html=True)
            transaction_amount     = st.number_input("Jumlah Transaksi (Rp)", min_value=10_000, max_value=50_000_000, value=500_000, step=10_000)
            hour_of_day            = st.slider("Jam Transaksi", 0, 23, 14)
            payment_channel        = st.selectbox("Channel Pembayaran", NOMINAL_FEATURES["payment_channel"])
            transaction_type       = st.selectbox("Jenis Transaksi", NOMINAL_FEATURES["transaction_type"])
            merchant_category      = st.selectbox("Kategori Merchant", NOMINAL_FEATURES["merchant_category"])
            region                 = st.selectbox("Region", NOMINAL_FEATURES["region"])

        with col2:
            st.markdown("<div class='section-title'>👤 Profil Nasabah</div>", unsafe_allow_html=True)
            customer_age           = st.number_input("Usia Nasabah", min_value=18, max_value=69, value=35)
            customer_segment       = st.selectbox("Segmen Nasabah", NOMINAL_FEATURES["customer_segment"])
            card_type              = st.selectbox("Tipe Kartu", NOMINAL_FEATURES["card_type"])
            credit_limit           = st.number_input("Limit Kredit (Rp)", min_value=5_000_000, max_value=100_000_000, value=20_000_000, step=1_000_000)
            account_age_days       = st.number_input("Usia Akun (hari)", min_value=1, max_value=3649, value=365)
            chargeback_count_12m   = st.number_input("Jumlah Chargeback 12 Bulan", min_value=0, max_value=3, value=0)

        # ── Row 2 ──
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("<div class='section-title'>📈 Perilaku Transaksi (30 Hari)</div>", unsafe_allow_html=True)
            avg_txn_amount_30d     = st.number_input("Rata-rata Nominal Transaksi (Rp)", min_value=12_963, max_value=5_344_274, value=450_000, step=10_000)
            txn_count_30d          = st.number_input("Jumlah Transaksi 30 Hari", min_value=1, max_value=79, value=15)
            txn_count_24h          = st.number_input("Jumlah Transaksi 24 Jam Terakhir", min_value=0, max_value=9, value=2)
            unique_merchant_30d    = st.number_input("Merchant Unik 30 Hari", min_value=1, max_value=29, value=8)
            time_since_last_txn_hr = st.number_input("Waktu Sejak Transaksi Terakhir (jam)", min_value=0.04, max_value=281.52, value=2.5)
            velocity_score         = st.number_input("Velocity Score (0–99.6)", min_value=0.0, max_value=99.6, value=10.0)

        with col4:
            st.markdown("<div class='section-title'>🔐 Keamanan & Perangkat</div>", unsafe_allow_html=True)
            device_type            = st.selectbox("Tipe Perangkat", NOMINAL_FEATURES["device_type"])
            authentication_method  = st.selectbox("Metode Autentikasi", NOMINAL_FEATURES["authentication_method"])
            ip_country             = st.selectbox("Negara IP", NOMINAL_FEATURES["ip_country"])
            failed_auth_24h        = st.number_input("Gagal Auth 24 Jam", min_value=0, max_value=5, value=0)
            device_change_7d       = st.number_input("Ganti Perangkat 7 Hari", min_value=0, max_value=3, value=0)
            distance_from_home_km  = st.number_input("Jarak dari Rumah (km)", min_value=0.0, max_value=205.6, value=5.0)

        # ── Row 3 — derived / flags ──
        st.markdown("---")
        col5, col6 = st.columns(2)

        with col5:
            st.markdown("<div class='section-title'>🧮 Fitur Turunan & Sinyal</div>", unsafe_allow_html=True)
            credit_utilization     = st.slider("Credit Utilization (0.01–0.95)", 0.01, 0.95, 0.25)
            amount_to_avg_ratio    = st.number_input("Amount to Avg Ratio", min_value=0.01, max_value=318.83, value=1.0)
            behavioral_signal_exp  = st.slider("Behavioral Signal (0–1)", 0.004, 0.998, 0.5)

        with col6:
            st.markdown("<div class='section-title'>🚩 Flag Indikator</div>", unsafe_allow_html=True)
            is_new_device          = st.checkbox("Perangkat Baru (is_new_device)")
            is_foreign_ip          = st.checkbox("IP Asing (is_foreign_ip)")
            is_high_risk_merchant  = st.checkbox("Merchant Berisiko Tinggi (is_high_risk_merchant)")
            is_weekend             = st.checkbox("Transaksi Weekend (is_weekend)")
            is_odd_hour            = st.checkbox("Jam Ganjil / Dini Hari (is_odd_hour)")
            flag_high_amount       = st.checkbox("Nominal Sangat Besar (flag_high_amount)")
            flag_high_velocity     = st.checkbox("Velocity Sangat Tinggi (flag_high_velocity)")

        submitted = st.form_submit_button("🔍 Deteksi Fraud →", use_container_width=True, type="primary")

    if submitted:
        inputs = {
            # numerical
            "hour_of_day":           hour_of_day,
            "transaction_amount":    transaction_amount,
            "credit_limit":          credit_limit,
            "credit_utilization":    credit_utilization,
            "amount_to_avg_ratio":   amount_to_avg_ratio,
            "avg_txn_amount_30d":    avg_txn_amount_30d,
            "customer_age":          customer_age,
            "account_age_days":      account_age_days,
            "txn_count_30d":         txn_count_30d,
            "txn_count_24h":         txn_count_24h,
            "unique_merchant_30d":   unique_merchant_30d,
            "failed_auth_24h":       failed_auth_24h,
            "device_change_7d":      device_change_7d,
            "time_since_last_txn_hr":time_since_last_txn_hr,
            "distance_from_home_km": distance_from_home_km,
            "velocity_score":        velocity_score,
            "chargeback_count_12m":  chargeback_count_12m,
            "is_new_device":         int(is_new_device),
            "is_foreign_ip":         int(is_foreign_ip),
            "is_high_risk_merchant": int(is_high_risk_merchant),
            "is_weekend":            int(is_weekend),
            "is_odd_hour":           int(is_odd_hour),
            "flag_high_amount":      int(flag_high_amount),
            "flag_high_velocity":    int(flag_high_velocity),
            "behavioral_signal_exp": behavioral_signal_exp,
            # nominal
            "payment_channel":       payment_channel,
            "merchant_category":     merchant_category,
            "region":                region,
            "customer_segment":      customer_segment,
            "card_type":             card_type,
            "device_type":           device_type,
            "authentication_method": authentication_method,
            "ip_country":            ip_country,
            "transaction_type":      transaction_type,
        }
        st.session_state["inputs"] = inputs
        st.session_state["page"]   = "result"
        st.success("✅ Data berhasil disubmit. Buka menu **📊 Hasil Deteksi** di sidebar.")

# ═══════════════════════════════════════════
#  PAGE 2 — HASIL DETEKSI
# ═══════════════════════════════════════════
elif "📊  Hasil Deteksi" in page:
    st.markdown(f"""
    <div style='display:flex;justify-content:space-between;align-items:flex-end;
                border-bottom:1px solid #e2e8f0;padding-bottom:16px;margin-bottom:24px;'>
        <div>
            <div style='font-size:20px;font-weight:700;color:#1e293b;'>Hasil Deteksi Fraud</div>
            <div style='font-size:13px;color:#64748b;margin-top:3px;'>Output model XGBoost · Pipeline fraud_detection_pipeline.pkl</div>
        </div>
        <div style='font-size:12px;color:#64748b;background:#f0f4f8;padding:6px 12px;border-radius:6px;border:1px solid #e2e8f0;'>{now_str}</div>
    </div>
    """, unsafe_allow_html=True)

    if "inputs" not in st.session_state:
        st.info("ℹ️ Belum ada data. Silakan isi form di **📋 Input Transaksi** terlebih dahulu.")
        st.stop()

    inputs  = st.session_state["inputs"]
    input_df = build_input_df(inputs)

    if pipeline is None:
        # ── Demo mode (model tidak tersedia) ──
        st.warning("⚠️ Model tidak ditemukan — menampilkan **mode demo** dengan nilai acak.")
        fraud_prob   = np.random.uniform(0.05, 0.95)
        shap_vals    = None
        feature_names = NUMERICAL_FEATURES
    else:
        fraud_prob   = pipeline.predict_proba(input_df)[0][1]
        # SHAP
        try:
            preprocessor = pipeline[:-1]       # semua step kecuali classifier
            classifier   = pipeline[-1]
            X_transformed = preprocessor.transform(input_df)
            explainer    = shap.TreeExplainer(classifier)
            shap_values  = explainer.shap_values(X_transformed)
            if isinstance(shap_values, list):
                shap_vals = shap_values[1][0]
            else:
                shap_vals = shap_values[0]
            # nama fitur setelah transformasi
            try:
                ohe = preprocessor.named_transformers_["cat"]["encoder"]
                ohe_names = list(ohe.get_feature_names_out(list(NOMINAL_FEATURES.keys())))
            except Exception:
                ohe_names = []
            feature_names = NUMERICAL_FEATURES + ohe_names
        except Exception as e:
            shap_vals     = None
            feature_names = NUMERICAL_FEATURES
            st.warning(f"SHAP tidak dapat dihitung: {e}")

    label, color, badge_cls = risk_label(fraud_prob)
    score = score_from_prob(fraud_prob)
    verdict = "🚨 FRAUD" if fraud_prob >= 0.5 else "✅ LEGITIMATE"
    verdict_color = "#ef4444" if fraud_prob >= 0.5 else "#10b981"

    # ── Gauge + Keputusan ──
    col_gauge, col_detail = st.columns([1, 2])

    with col_gauge:
        with st.container(border=True):
            st.plotly_chart(gauge_chart(fraud_prob), use_container_width=True, config={"displayModeBar": False})
            st.markdown(f"""
            <div style='text-align:center;margin-top:-8px;'>
                <div style='font-size:26px;font-weight:800;color:{verdict_color};'>{verdict}</div>
                <div style='margin-top:8px;'><span class='{badge_cls}'>RISIKO {label.upper()}</span></div>
                <div style='margin-top:14px;font-size:13px;color:#64748b;'>Risk Score Model</div>
                <div style='font-size:30px;font-weight:700;color:{color};'>{score}</div>
                <div style='font-size:11px;color:#94a3b8;'>dari 850</div>
            </div>
            """, unsafe_allow_html=True)

    with col_detail:
        with st.container(border=True):
            st.markdown("<div class='section-title'>Detail Keputusan</div>", unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Fraud Probability", f"{fraud_prob*100:.1f}%")
            m2.metric("Risk Score", f"{score}")
            m3.metric("Threshold", "50%")
            m4.metric("Keputusan", "Blokir" if fraud_prob >= 0.5 else "Approve")

            st.markdown("---")
            # Ringkasan fitur input penting
            st.markdown("**Ringkasan Input Kunci**")
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Amount:** Rp {inputs['transaction_amount']:,}")
            c1.markdown(f"**Channel:** {inputs['payment_channel']}")
            c1.markdown(f"**Merchant:** {inputs['merchant_category']}")
            c2.markdown(f"**Velocity Score:** {inputs['velocity_score']}")
            c2.markdown(f"**IP Country:** {inputs['ip_country']}")
            c2.markdown(f"**Auth:** {inputs['authentication_method']}")
            c3.markdown(f"**Failed Auth 24h:** {inputs['failed_auth_24h']}")
            c3.markdown(f"**Device Change:** {inputs['device_change_7d']}")
            c3.markdown(f"**Distance:** {inputs['distance_from_home_km']} km")

            # Info box rekomendasi
            if fraud_prob >= 0.75:
                st.markdown('<div class="alert-red">🚨 <strong>Rekomendasi:</strong> Blokir transaksi segera & eskalasi ke tim investigasi fraud.</div>', unsafe_allow_html=True)
            elif fraud_prob >= 0.50:
                st.markdown('<div class="alert-yellow">⚠️ <strong>Rekomendasi:</strong> Tahan transaksi sementara & minta verifikasi tambahan dari nasabah.</div>', unsafe_allow_html=True)
            elif fraud_prob >= 0.20:
                st.markdown('<div class="alert-green">🔍 <strong>Rekomendasi:</strong> Approve dengan monitoring. Pantau transaksi berikutnya dari nasabah ini.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-green">✅ <strong>Rekomendasi:</strong> Approve. Profil transaksi normal sesuai histori nasabah.</div>', unsafe_allow_html=True)

    # ── SHAP Chart ──
    st.markdown("---")
    with st.container(border=True):
        st.markdown("<div class='section-title'>Explainability — Kontribusi Fitur (SHAP Values)</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-sub'>Fitur positif (merah) meningkatkan risiko fraud; negatif (hijau) menurunkan risiko. Dapat diaudit sesuai regulasi OJK.</div>", unsafe_allow_html=True)

        if shap_vals is not None:
            st.plotly_chart(shap_chart(shap_vals, feature_names), use_container_width=True, config={"displayModeBar": False})
        else:
            # Demo SHAP
            demo_shap = {
                "failed_auth_24h":       0.18,
                "velocity_score":        0.15,
                "distance_from_home_km": 0.12,
                "is_foreign_ip":         0.10,
                "amount_to_avg_ratio":   0.09,
                "is_high_risk_merchant": 0.08,
                "device_change_7d":      0.07,
                "behavioral_signal_exp": -0.11,
                "account_age_days":      -0.09,
                "txn_count_30d":         -0.06,
                "authentication_method_Biometric": -0.05,
                "credit_utilization":    0.04,
            }
            vals  = list(demo_shap.values())
            feats = list(demo_shap.keys())
            st.plotly_chart(shap_chart(np.array(vals), feats), use_container_width=True, config={"displayModeBar": False})
            st.caption("*Mode demo — nilai SHAP ilustratif karena model belum dimuat.*")

# ═══════════════════════════════════════════
#  PAGE 3 — EARLY WARNING
# ═══════════════════════════════════════════
elif "🔔  Early Warning" in page:
    st.markdown(f"""
    <div style='display:flex;justify-content:space-between;align-items:flex-end;
                border-bottom:1px solid #e2e8f0;padding-bottom:16px;margin-bottom:24px;'>
        <div>
            <div style='font-size:20px;font-weight:700;color:#1e293b;'>Panel Early Warning</div>
            <div style='font-size:13px;color:#64748b;margin-top:3px;'>Pemantauan portofolio transaksi real-time · Demo data</div>
        </div>
        <div style='font-size:12px;color:#64748b;background:#f0f4f8;padding:6px 12px;border-radius:6px;border:1px solid #e2e8f0;'>{now_str}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Summary metrics ──
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Transaksi Hari Ini", "1,284")
    m2.metric("Legitimate", "1,251", delta="97.4%")
    m3.metric("Flagged (>50%)", "33", delta="-3 vs kemarin", delta_color="inverse")
    m4.metric("Avg Fraud Prob", "8.3%")

    st.markdown("---")

    # ── Trend fraud probability (simulasi) ──
    col_trend, col_dist = st.columns([3, 2])

    with col_trend:
        with st.container(border=True):
            st.markdown("<div class='section-title'>Tren Fraud Probability Harian</div>", unsafe_allow_html=True)
            hours = list(range(0, 24))
            np.random.seed(42)
            fraud_counts = np.clip(np.random.normal(8, 4, 24).astype(int), 0, 30)
            fraud_counts[2]  = 18   # dini hari spike
            fraud_counts[3]  = 22
            fraud_counts[14] = 5
            df_trend = pd.DataFrame({"Jam": hours, "Jumlah Flagged": fraud_counts, "Baseline": [6]*24})
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=df_trend["Jam"], y=df_trend["Baseline"],
                mode="lines", name="Baseline", line=dict(dash="dash", color="#94a3b8")))
            fig_trend.add_trace(go.Scatter(x=df_trend["Jam"], y=df_trend["Jumlah Flagged"],
                mode="lines+markers", name="Flagged Txn", line=dict(color="#ef4444", width=2),
                fill="tozeroy", fillcolor="rgba(239,68,68,0.08)"))
            fig_trend.update_layout(
                height=220, margin=dict(t=10, b=10, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Jam ke-", gridcolor="#f0f4f8"),
                yaxis=dict(title="Jumlah", gridcolor="#f0f4f8"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})

    with col_dist:
        with st.container(border=True):
            st.markdown("<div class='section-title'>Distribusi Risiko</div>", unsafe_allow_html=True)
            fig_pie = go.Figure(go.Pie(
                labels=["Rendah (<20%)", "Sedang (20–50%)", "Tinggi (50–75%)", "Sangat Tinggi (>75%)"],
                values=[1031, 220, 24, 9],
                marker_colors=["#10b981", "#f59e0b", "#ef4444", "#7f1d1d"],
                hole=0.5,
            ))
            fig_pie.update_layout(
                height=220, margin=dict(t=10, b=10, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(font=dict(size=10)),
                showlegend=True,
            )
            st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

    # ── Tabel transaksi flagged ──
    st.markdown("---")
    with st.container(border=True):
        st.markdown("<div class='section-title'>⚠️ Transaksi Flagged — Perlu Tindakan</div>", unsafe_allow_html=True)

        flagged_data = [
            {"ID": "TXN-9921", "Nasabah": "Budi Santoso",   "Amount": "Rp 45.000.000", "Channel": "Mobile App",   "Merchant": "Jewelry",   "Fraud Prob": "87.3%", "Risiko": "Sangat Tinggi", "Status": "🚨 Blokir"},
            {"ID": "TXN-8843", "Nasabah": "Siti Rahayu",    "Amount": "Rp 12.500.000", "Channel": "Web",          "Merchant": "Gambling",  "Fraud Prob": "76.1%", "Risiko": "Sangat Tinggi", "Status": "🚨 Blokir"},
            {"ID": "TXN-7712", "Nasabah": "Ahmad Fauzi",    "Amount": "Rp 8.750.000",  "Channel": "Transfer Bank","Merchant": "Travel",    "Fraud Prob": "63.4%", "Risiko": "Tinggi",        "Status": "⚠️ Tahan"},
            {"ID": "TXN-6608", "Nasabah": "Dewi Lestari",   "Amount": "Rp 3.200.000",  "Channel": "EDC / POS",    "Merchant": "Retail",    "Fraud Prob": "54.9%", "Risiko": "Tinggi",        "Status": "⚠️ Tahan"},
            {"ID": "TXN-5501", "Nasabah": "Rudi Hermawan",  "Amount": "Rp 1.800.000",  "Channel": "Mobile App",   "Merchant": "E-Commerce","Fraud Prob": "41.2%", "Risiko": "Sedang",        "Status": "🔍 Monitor"},
        ]

        df_flagged = pd.DataFrame(flagged_data)
        st.dataframe(
            df_flagged,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Fraud Prob": st.column_config.TextColumn("Fraud Prob"),
                "Status": st.column_config.TextColumn("Status"),
            }
        )

    # ── Alert cards ──
    st.markdown("---")
    st.markdown("**🚨 Alert Aktif — Memerlukan Tindakan Segera**")

    alerts = [
        {
            "id": "TXN-9921", "nama": "Budi Santoso", "prob": "87.3%",
            "detail": "Transaksi Rp 45.000.000 ke merchant Jewelry via Mobile App dari IP Singapura. Velocity score 94.2 — jauh di atas baseline historis nasabah. Device baru terdeteksi 2 jam sebelum transaksi.",
            "rekomendasi": "Blokir transaksi & hubungi nasabah untuk verifikasi identitas. Eskalasi ke tim investigasi jika nasabah tidak dapat dihubungi dalam 2 jam.",
            "cls": "alert-red",
        },
        {
            "id": "TXN-8843", "nama": "Siti Rahayu", "prob": "76.1%",
            "detail": "Transaksi Rp 12.500.000 ke kategori Gambling via Web dengan IP asing (CN). Failed authentication 3x dalam 24 jam terakhir. Jarak dari domisili 187 km.",
            "rekomendasi": "Tahan transaksi & kirim notifikasi OTP ke nomor terdaftar. Jika OTP gagal, blokir sementara kartu nasabah.",
            "cls": "alert-yellow",
        },
    ]
    for a in alerts:
        st.markdown(f"""
        <div class='{a["cls"]}'>
            <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;'>
                <div>
                    <span style='font-size:14px;font-weight:700;color:#1e293b;'>{a["nama"]}</span>
                    <span style='font-size:11px;color:#64748b;margin-left:8px;'>{a["id"]}</span>
                </div>
                <span style='font-size:13px;font-weight:700;color:#ef4444;'>Fraud Prob: {a["prob"]}</span>
            </div>
            <div style='font-size:12px;color:#1e293b;margin-bottom:10px;line-height:1.6;'>{a["detail"]}</div>
            <div style='font-size:12px;background:rgba(255,255,255,0.8);padding:8px 12px;border-radius:6px;'>
                <strong>Rekomendasi tindakan:</strong> {a["rekomendasi"]}
            </div>
        </div>
        """, unsafe_allow_html=True)
