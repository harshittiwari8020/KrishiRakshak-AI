from pathlib import Path
import sqlite3
import datetime

import pandas as pd
import streamlit as st
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

# =========================================================
# KRISHIRAKSHAK AI - CROP DISEASE DETECTION
# =========================================================
st.set_page_config(
    page_title="KrishiRakshak AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "crop_disease_model.pth"
DB_PATH = BASE_DIR / "outputs" / "predictions.db"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUPPORT_EMAIL = "krishikarshak8020@gmail.com"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ------------------------- CSS -------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
    :root { --forest:#103b2b; --green:#1f8050; --mint:#eaf6ee; --ink:#173d2c; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp { background:linear-gradient(135deg,#f4f8f5 0%,#fbfdfb 55%,#eef6f0 100%); }
    [data-testid="stHeader"] { background:rgba(255,255,255,.82); }
    [data-testid="stSidebar"] { background:linear-gradient(180deg,#0d3023 0%,#154a34 100%); border-right:1px solid rgba(255,255,255,.08); }
    [data-testid="stSidebar"] > div:first-child { padding-top:1.2rem; }
    [data-testid="stSidebar"] * { color:#f2fff5; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#c7dfd0; }
    [data-testid="stSidebar"] [data-testid="stRadio"] > label { color:#a9cbbb; font-size:.72rem; font-weight:800; letter-spacing:1.6px; margin:8px 0 12px; }
    [data-testid="stSidebar"] [data-testid="stRadio"] > div { gap:7px; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        display:flex; align-items:center; width:100%; box-sizing:border-box;
        padding:11px 13px; border-radius:11px; border:1px solid transparent;
        background:rgba(255,255,255,.035); transition:all .18s ease; cursor:pointer;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background:rgba(255,255,255,.12); border-color:rgba(255,255,255,.12); transform:translateX(2px);
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background:linear-gradient(100deg,#278657,#1b7047) !important;
        border:1px solid rgba(194,244,208,.32) !important;
        box-shadow:0 5px 16px rgba(0,0,0,.18), inset 3px 0 0 #b8f2c8;
        color:#fff !important; font-weight:800;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) * { color:#fff !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child { transform:scale(.88); }
    .brand-panel { padding:8px 4px 20px; }
    .brand-mark { width:48px;height:48px;display:flex;align-items:center;justify-content:center;border-radius:15px;background:linear-gradient(145deg,#a9e76e,#37a76b);font-size:27px;box-shadow:0 7px 18px #071f1680;margin-bottom:12px; }
    .brand-name { font-size:1.35rem;font-weight:800;color:white;letter-spacing:-.5px; }
    .brand-sub { font-size:.68rem;letter-spacing:1.4px;color:#b9d9c5;margin-top:3px; }
    .hero-box { background:linear-gradient(115deg,#103b2b 0%,#176b46 58%,#36a36a 100%); padding:34px 36px; border-radius:22px; color:white; margin:8px 0 26px; box-shadow:0 16px 36px rgba(15,70,40,.16); position:relative; overflow:hidden; }
    .hero-box:after { content:'🌿';position:absolute;right:38px;top:22px;font-size:74px;opacity:.16; }
    .hero-box h1 { color:white !important;font-size:2.25rem !important;font-weight:800 !important;margin:7px 0 10px !important;letter-spacing:-1px; }
    .hero-box p { color:#e5f5e9 !important;margin:0 !important;font-size:1rem;max-width:720px;line-height:1.65; }
    .eyebrow { color:#bde9ca;font-weight:800;letter-spacing:2px;font-size:.72rem; }
    .section-title { color:var(--ink);font-size:1.35rem;font-weight:800;margin:14px 0;letter-spacing:-.3px; }
    .stat-card,.feature-card { background:rgba(255,255,255,.94);border:1px solid #dfebe2;border-radius:17px;padding:21px;box-shadow:0 7px 22px rgba(25,70,40,.055);transition:transform .18s ease,box-shadow .18s ease; }
    .stat-card:hover,.feature-card:hover { transform:translateY(-3px);box-shadow:0 12px 28px rgba(25,70,40,.10); }
    .stat-card { min-height:116px; }
    .stat-label { color:#718276;font-size:.75rem;font-weight:800;letter-spacing:.8px; }
    .stat-value { color:var(--ink);font-size:1.9rem;font-weight:800;margin-top:8px; }
    .stat-hint { color:#87948b;font-size:.76rem;margin-top:3px; }
    .feature-card { min-height:190px;margin-bottom:10px;position:relative;overflow:hidden; }
    .feature-card:before { content:"";position:absolute;left:0;top:0;bottom:0;width:4px;background:linear-gradient(180deg,#35a66b,#9bdc9e); }
    .feature-tag { display:inline-block;background:#eaf6ee;color:#207449;border:1px solid #d4eadb;border-radius:30px;padding:4px 9px;font-size:.67rem;font-weight:800;letter-spacing:.6px;margin:0 0 8px 0; }
    .feature-card p { min-height:48px; }
    .feature-meta { color:#268052;font-size:.76rem;font-weight:700;margin-top:11px; }
    div.stButton > button[kind="primary"] { background:linear-gradient(100deg,#176b43,#29965c);border:1px solid #248451;border-radius:11px;min-height:45px;font-weight:800;box-shadow:0 5px 14px rgba(32,120,73,.16);color:white; }
    div.stButton > button[kind="primary"]:hover { background:linear-gradient(100deg,#125735,#21824f);color:white;transform:translateY(-2px);box-shadow:0 8px 18px rgba(32,120,73,.24); }
    .feature-icon { width:48px;height:48px;display:flex;align-items:center;justify-content:center;border-radius:14px;background:#eaf6ee;font-size:1.55rem;margin-bottom:13px; }
    .feature-card h3 { color:var(--ink);margin:5px 0;font-size:1.08rem;font-weight:800; }
    .feature-card p { color:#6d7e72;font-size:.88rem;line-height:1.55;margin-bottom:0; }
    .result-card { background:#eff8f1;border:1px solid #d6eadb;border-radius:15px;padding:20px; }
    div.stButton > button[kind="primary"] { background:linear-gradient(100deg,#176b43,#29965c);border:0;border-radius:11px;min-height:45px;font-weight:800;box-shadow:0 5px 14px rgba(32,120,73,.16); }
    div.stButton > button[kind="primary"]:hover { background:#155b36;color:white;transform:translateY(-1px); }
    div.stButton > button { border-radius:10px; }
    div[data-testid="stFileUploader"] { background:white;border:1px dashed #9fc9aa;border-radius:14px;padding:11px; }
    .small-muted { color:#718176;font-size:.88rem; }
    [data-testid="stDataFrame"] { border:1px solid #e1ebe3;border-radius:12px;overflow:hidden; }
    [data-testid="stMetric"] { background:white;border:1px solid #e1ebe3;border-radius:14px;padding:13px 16px; }
    .footer-note { color:#87948b;font-size:.78rem;text-align:center;padding:14px 0 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------- Database -------------------------
def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                filename TEXT,
                prediction TEXT,
                confidence REAL
            )"""
        )


def read_history(limit=1000):
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT created_at, filename, prediction, confidence FROM predictions ORDER BY id DESC LIMIT ?",
            conn,
            params=(limit,),
        )


def save_prediction(filename, prediction, confidence):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO predictions (created_at, filename, prediction, confidence) VALUES (?, ?, ?, ?)",
            (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), filename, str(prediction), float(confidence)),
        )


def pretty_name(name):
    return str(name).replace("___", " — ").replace("_", " ").title()


# ------------------------- Model -------------------------
@st.cache_resource(show_spinner="Loading crop disease model...")
def load_model():
    if not MODEL_PATH.exists():
        return None, None, 224, f"Model file not found: {MODEL_PATH}"
    try:
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        except TypeError:
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

        if not isinstance(checkpoint, dict):
            return None, None, 224, "The model checkpoint format is not a dictionary."

        classes = checkpoint.get("classes") or checkpoint.get("class_names")
        state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict")
        image_size = int(checkpoint.get("image_size", 224))

        if classes is None or state_dict is None:
            return None, None, image_size, "Checkpoint must contain 'classes' and 'state_dict'."

        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        return model, list(classes), image_size, None
    except Exception as exc:
        return None, None, 224, f"Could not load model: {exc}"


def predict_image(image, model, classes, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    values, indices = torch.topk(probabilities, min(3, len(classes)))
    return [(classes[int(idx)], float(val)) for val, idx in zip(values.cpu(), indices.cpu())]


init_db()
model, classes, image_size, model_error = load_model()
history = read_history()

# ------------------------- Sidebar / Navigation -------------------------
if "navigation_target" in st.session_state:
    st.session_state["page_nav"] = st.session_state.pop("navigation_target")

with st.sidebar:
    st.markdown(
        "<div class='brand-panel'><div class='brand-mark'>🌱</div>"
        "<div class='brand-name'>KrishiRakshak</div>"
        "<div class='brand-sub'>AI CROP HEALTH PLATFORM</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    page = st.radio(
        "NAVIGATION",
        ["Home", "Disease Detection", "Overview", "Scan History", "Customer Support"],
        label_visibility="visible",
        key="page_nav",
    )
    st.markdown("---")
    st.caption("AI-assisted crop screening")

# ------------------------- Header -------------------------
# Use native Streamlit text elements here so HTML tags are never shown as code.
st.markdown('<div class="eyebrow">SMART FARMING  •  AI-POWERED INSIGHTS</div>', unsafe_allow_html=True)
st.markdown("# KrishiRakshak AI")
st.markdown('<div class="small-muted">An intelligent crop health screening platform powered by deep learning.</div>', unsafe_allow_html=True)

if model_error:
    st.error(model_error)
    st.info("Check that your trained EfficientNet-B0 checkpoint is at models/crop_disease_model.pth and contains the expected class labels and state dictionary.")
    st.stop()

# ------------------------- Home -------------------------
if page == "Home":
    st.markdown(
        '<div class="hero-box"><div class="eyebrow">SMART AGRICULTURE  •  INTELLIGENT CROP CARE</div>'
        '<h1>Grow healthier crops with AI.</h1>'
        '<p>Screen crop leaf images with deep learning, review prediction confidence, explore scan insights, and keep your crop health records organised in one platform.</p></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Platform features</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-muted">Choose a workspace to get started. All platform modules use the same active green action style.</div>', unsafe_allow_html=True)
    st.write("")

    feature_data = [
        ("🔬", "Disease Detection", "AI-POWERED SCREENING", "Upload a crop leaf image and view the top predictions, confidence scores, and practical next-step guidance.", "Image analysis  •  Top-3 predictions", "Disease Detection", "home_detect"),
        ("📊", "Crop Health Overview", "ANALYTICS DASHBOARD", "Understand scan activity with healthy and possible-disease counts, average model score, recent scans, and prediction distribution.", "Statistics  •  Visual insights", "Overview", "home_overview"),
        ("🕘", "Scan History", "RECORD MANAGEMENT", "Review earlier crop scans with timestamps, image filenames, predicted classes, and confidence scores. Export records to CSV.", "Historical records  •  CSV export", "Scan History", "home_history"),
        ("✉️", "Customer Support", "HELP & ASSISTANCE", "Reach the support team for application questions, technical issues, or feedback about the KrishiRakshak platform.", "Email support  •  User assistance", "Customer Support", "home_support"),
    ]

    col1, col2 = st.columns(2, gap="large")
    for idx, item in enumerate(feature_data):
        icon, title, tag, description, meta, target, key = item
        col = col1 if idx % 2 == 0 else col2
        with col:
            st.markdown(
                f'<div class="feature-card"><div class="feature-icon">{icon}</div>'
                f'<div class="feature-tag">{tag}</div><h3>{title}</h3>'
                f'<p>{description}</p><div class="feature-meta">✓ {meta}</div></div>',
                unsafe_allow_html=True,
            )
            if st.button(f"Open {title}", key=key, type="primary", use_container_width=True):
                st.session_state["navigation_target"] = target
                st.rerun()

    st.write("")
    st.markdown(
        '<div class="result-card"><b style="color:#173d2c;">Platform note</b>'
        '<div class="small-muted" style="margin-top:6px;">KrishiRakshak AI provides decision-support screening. Model predictions should be confirmed by a qualified agriculture expert before treatment decisions.</div></div>',
        unsafe_allow_html=True,
    )

# ------------------------- Disease Detection -------------------------
elif page == "Disease Detection":
    st.markdown('<div class="section-title">AI-powered leaf screening</div>', unsafe_allow_html=True)
    st.markdown('<p class="small-muted">Upload a clear, well-lit crop leaf image. Supported formats: JPG, JPEG and PNG.</p>', unsafe_allow_html=True)
    left, right = st.columns([1, 1], gap="large")
    with left:
        uploaded = st.file_uploader("Choose crop leaf image", type=["jpg", "jpeg", "png"], key="leaf_upload")
        if uploaded is not None:
            try:
                image = Image.open(uploaded).convert("RGB")
                st.image(image, caption="Uploaded crop leaf image", use_container_width=True)
                if st.button("🔍 Analyse Crop Leaf", type="primary", use_container_width=True):
                    with st.spinner("Analysing image..."):
                        result = predict_image(image, model, classes, image_size)
                        st.session_state["latest_results"] = result
                        st.session_state["latest_image_name"] = uploaded.name
                        save_prediction(uploaded.name, result[0][0], result[0][1])
                        st.success("Analysis completed and scan saved.")
            except Exception as exc:
                st.error(f"Unable to open image: {exc}")
        else:
            st.info("Your uploaded leaf image preview will appear here.")
    with right:
        st.markdown('<div class="section-title">Analysis result</div>', unsafe_allow_html=True)
        results = st.session_state.get("latest_results")
        if results:
            label, score = results[0]
            healthy = "healthy" in str(label).lower()
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.caption("TOP PREDICTION")
            st.subheader(pretty_name(label))
            st.metric("Model confidence score", f"{score * 100:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
            if score < 0.60:
                st.warning("The model confidence is low. Try a clearer image and seek confirmation from an agriculture expert.")
            elif healthy:
                st.success("The model did not identify a disease pattern in this image.")
            else:
                st.warning("Possible disease pattern detected. This is an AI screening result, not a confirmed diagnosis.")
            st.write("**Top predictions**")
            result_df = pd.DataFrame([{"Predicted class": pretty_name(name), "Model score": f"{value * 100:.2f}%"} for name, value in results])
            st.dataframe(result_df, use_container_width=True, hide_index=True)
            st.write("**Suggested next steps**")
            if healthy:
                st.markdown("- Continue regular field scouting.\n- Maintain balanced irrigation and plant nutrition.\n- Monitor new leaves regularly.")
            else:
                st.markdown("- Inspect nearby plants for similar symptoms.\n- Avoid moving affected plant debris between fields.\n- Ask a local agriculture officer or KVK to confirm the disease.\n- Do not apply pesticides based only on this AI prediction.")
            st.caption(f"Image: {st.session_state.get('latest_image_name', 'Uploaded image')}")
            st.caption("A model score is not the same as the probability that a diagnosis is correct.")
        else:
            st.info("Upload an image and click Analyse Crop Leaf to see the prediction here.")

# ------------------------- Overview -------------------------
elif page == "Overview":
    history = read_history()
    total = len(history)
    if total:
        healthy_count = int(history["prediction"].astype(str).str.contains("healthy", case=False, na=False).sum())
        disease_count = total - healthy_count
        avg_score = float(history["confidence"].mean() * 100)
    else:
        healthy_count = disease_count = 0
        avg_score = 0.0
    st.markdown('<div class="section-title">Your crop health overview</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    stats = [
        ("TOTAL SCANS", total, "All saved predictions"),
        ("HEALTHY RESULTS", healthy_count, "Predicted healthy"),
        ("POSSIBLE DISEASES", disease_count, "Require expert confirmation"),
        ("AVERAGE MODEL SCORE", f"{avg_score:.1f}%", "Across saved scans"),
    ]
    for col, (label, value, hint) in zip(cols, stats):
        with col:
            st.markdown(f'<div class="stat-card"><div class="stat-label">{label}</div><div class="stat-value">{value}</div><div class="stat-hint">{hint}</div></div>', unsafe_allow_html=True)
    st.write("")
    lcol, rcol = st.columns([1.3, 1])
    with lcol:
        st.markdown('<div class="section-title">Recent scans</div>', unsafe_allow_html=True)
        if total:
            recent = history.head(8).copy()
            recent["prediction"] = recent["prediction"].map(pretty_name)
            recent["confidence"] = recent["confidence"].map(lambda x: f"{x * 100:.1f}%")
            st.dataframe(recent, use_container_width=True, hide_index=True)
        else:
            st.info("No scans recorded yet. Start from Disease Detection.")
    with rcol:
        st.markdown('<div class="section-title">Prediction distribution</div>', unsafe_allow_html=True)
        if total:
            counts = history["prediction"].map(pretty_name).value_counts().head(10)
            st.bar_chart(counts)
        else:
            st.info("The chart will appear after your first scan.")
    st.caption("The dashboard summarizes saved model predictions. Historical results are not independently verified diagnoses.")

# ------------------------- Scan History -------------------------
elif page == "Scan History":
    st.markdown('<div class="section-title">Previous crop scans</div>', unsafe_allow_html=True)
    history = read_history()
    if not history.empty:
        display = history.copy()
        display["prediction"] = display["prediction"].map(pretty_name)
        display["confidence"] = display["confidence"].map(lambda x: f"{x * 100:.2f}%")
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download scan history (CSV)", history.to_csv(index=False).encode("utf-8"), "krishirakshak_scan_history.csv", "text/csv")
    else:
        st.info("No scan history is available yet.")

# ------------------------- Customer Support -------------------------
elif page == "Customer Support":
    st.markdown('<div class="section-title">Customer support</div>', unsafe_allow_html=True)
    st.write("For application assistance or to report an issue, contact our support email:")
    st.success(f"✉️ {SUPPORT_EMAIL}")
    st.markdown(f"[Send an email](mailto:{SUPPORT_EMAIL})")

st.markdown("---")
st.markdown("<div class='footer-note'>KRISHIRAKSHAK AI  •  AI-ASSISTED CROP HEALTH SCREENING  •  CONFIRM SUSPECTED DISEASES WITH A QUALIFIED AGRICULTURE EXPERT</div>", unsafe_allow_html=True)
