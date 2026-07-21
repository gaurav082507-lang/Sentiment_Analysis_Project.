import streamlit as st
import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# CUSTOM CSS — dark themed, gradient hero, real bordered cards
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        .stApp {
            background-color: #08080c;
            background-image:
                linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px);
            background-size: 42px 42px;
        }

        #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }

        .block-container {
            padding-top: 3.5rem;
            padding-bottom: 2rem;
            max-width: 700px;
        }

        /* ---------------- Header ---------------- */
        .eyebrow {
            text-align: center;
            letter-spacing: 4px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            font-weight: 600;
            color: #8b8bd8;
            margin: 0 0 20px 0;
        }
        .hero-title {
            text-align: center;
            font-weight: 800 !important;
            font-size: 46px !important;
            line-height: 1.15 !important;
            color: #f5f5fa;
            margin: 0 !important;
        }
        .hero-gradient {
            font-weight: 800 !important;
            background: linear-gradient(90deg, #8b7bf0 0%, #7aa2f7 55%, #7ad0f7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .hero-sub {
            text-align: center;
            color: #9b9bb0;
            font-size: 15.5px;
            max-width: 520px;
            margin: 22px auto 34px auto !important;
            line-height: 1.65;
        }

        /* ---------------- Real bordered containers (cards) ---------------- */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(18, 18, 26, 0.75) !important;
            border: 1px solid rgba(140, 130, 220, 0.18) !important;
            border-radius: 16px !important;
            backdrop-filter: blur(6px);
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(> div) {
            margin-bottom: 20px;
        }

        .field-label {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11.5px;
            letter-spacing: 2px;
            color: #7d7d95;
            margin: 0 0 12px 0;
            font-weight: 600;
        }

        /* ---------------- Text area ---------------- */
        div[data-testid="stTextArea"] textarea {
            background-color: #101018 !important;
            border: 1px solid rgba(140, 130, 220, 0.25) !important;
            border-radius: 10px !important;
            color: #f0f0f5 !important;
            font-size: 15px !important;
            padding: 14px !important;
        }
        div[data-testid="stTextArea"] textarea:focus {
            border: 1px solid #8b7bf0 !important;
            box-shadow: 0 0 0 1px #8b7bf0 !important;
        }
        div[data-testid="stTextArea"] textarea::placeholder {
            color: #55556b !important;
        }

        /* ---------------- Button ---------------- */
        div[data-testid="stButton"] {
            margin-top: 16px;
        }
        div[data-testid="stButton"] > button {
            width: 100%;
            background: linear-gradient(90deg, #7c6df0 0%, #6f8ef2 100%);
            color: #ffffff !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 13px 0 !important;
            font-weight: 700 !important;
            font-size: 15.5px !important;
            letter-spacing: 0.3px;
            box-shadow: 0 6px 18px rgba(124, 109, 240, 0.35);
            transition: all 0.15s ease-in-out;
        }
        div[data-testid="stButton"] > button:hover {
            filter: brightness(1.08);
            box-shadow: 0 8px 22px rgba(124, 109, 240, 0.5);
            transform: translateY(-1px);
        }
        div[data-testid="stButton"] > button p {
            font-size: 15.5px !important;
            font-weight: 700 !important;
        }

        /* ---------------- Result card ---------------- */
        .result-header {
            display: flex;
            align-items: center;
            gap: 16px;
            margin-bottom: 20px;
        }
        .result-badge {
            width: 58px;
            height: 58px;
            min-width: 58px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 800;
            font-size: 21px;
            border: 1.5px solid;
        }
        .result-title {
            font-size: 25px;
            font-weight: 800;
            color: #f5f5fa;
            margin: 0;
            line-height: 1.2;
        }
        .result-conf {
            font-family: 'JetBrains Mono', monospace;
            font-size: 13.5px;
            font-weight: 600;
            margin-top: 3px;
        }
        .top-pred-label {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            letter-spacing: 2px;
            color: #7d7d95;
            margin: 6px 0 14px 0;
            font-weight: 600;
        }
        .bar-row {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 12px;
        }
        .bar-name {
            font-family: 'JetBrains Mono', monospace;
            font-size: 13.5px;
            color: #c8c8d8;
            width: 88px;
            min-width: 88px;
        }
        .bar-track {
            flex: 1;
            height: 9px;
            background: #1a1a24;
            border-radius: 5px;
            overflow: hidden;
        }
        .bar-fill { height: 100%; border-radius: 5px; }
        .bar-pct {
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            color: #9b9bb0;
            width: 55px;
            text-align: right;
        }

        /* ---------------- Footer ---------------- */
        .footer-wrap {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid rgba(140, 130, 220, 0.14);
        }
        .footer-text { color: #6b6b80; font-size: 13.5px; }
        .footer-text a { color: #9c95f5; text-decoration: none; font-weight: 600; }
        .footer-text a:hover { text-decoration: underline; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# LABEL CONFIG
# ---------------------------------------------------------------------------
LABELS = ["Negative", "Positive", "Neutral", "Irrelevant"]
LABEL_STYLE = {
    "Negative":   {"color": "#f76b7c", "bg": "rgba(247,107,124,0.12)", "short": "NEG"},
    "Positive":   {"color": "#5be08f", "bg": "rgba(91,224,143,0.12)",  "short": "POS"},
    "Neutral":    {"color": "#7aa2f7", "bg": "rgba(122,162,247,0.12)", "short": "NEU"},
    "Irrelevant": {"color": "#c0a8f7", "bg": "rgba(192,168,247,0.12)", "short": "IRR"},
}

# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------
MODEL_PATH = "sentiment_model.keras"  # export via model.save("sentiment_model.keras")


@st.cache_resource(show_spinner=False)
def load_model(path):
    try:
        return tf.keras.models.load_model(path)
    except Exception:
        return None


model = load_model(MODEL_PATH)

# ---------------------------------------------------------------------------
# HEADER (single markdown block — no stray spacing)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="eyebrow">DEEP LEARNING &nbsp;·&nbsp; NLP</div>
    <p class="hero-title">Detect Text<br><span class="hero-gradient">Sentiment</span></p>
    <p class="hero-sub">A Bidirectional LSTM model trained to classify sentiment
    as Positive, Negative, Neutral, or Irrelevant.</p>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# INPUT CARD — real container, so widgets are genuinely nested inside it
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.markdown('<div class="field-label">ENTER TEXT TO ANALYZE</div>', unsafe_allow_html=True)
    text_input = st.text_area(
        label="text_input",
        label_visibility="collapsed",
        placeholder="e.g. I would never recommend this terrible service to anyone.",
        height=120,
    )
    analyze_clicked = st.button("Detect Sentiment")

# ---------------------------------------------------------------------------
# PREDICTION + RESULT CARD
# ---------------------------------------------------------------------------
if analyze_clicked:
    if not text_input.strip():
        st.warning("Please enter some text first.")
    elif model is None:
        st.error(
            f"Couldn't find a trained model at `{MODEL_PATH}`. "
            "Export it from the notebook with `model.save('sentiment_model.keras')` "
            "and place the file next to this app."
        )
    else:
        with st.spinner("Analyzing..."):
            probs = model.predict(tf.constant([text_input]), verbose=0)[0]

        order = np.argsort(probs)[::-1]
        top_idx = order[0]
        top_label = LABELS[top_idx]
        top_conf = probs[top_idx] * 100
        style = LABEL_STYLE[top_label]

        with st.container(border=True):
            st.markdown(
                f"""
                <div class="result-header">
                    <div class="result-badge" style="background:{style['bg']}; color:{style['color']}; border-color:{style['color']};">
                        {style['short']}
                    </div>
                    <div>
                        <p class="result-title">{top_label}</p>
                        <p class="result-conf" style="color:{style['color']};">{top_conf:.2f}% confidence</p>
                    </div>
                </div>
                <div class="top-pred-label">ALL PREDICTIONS</div>
                """,
                unsafe_allow_html=True,
            )

            for idx in order:
                label = LABELS[idx]
                pct = probs[idx] * 100
                s = LABEL_STYLE[label]
                st.markdown(
                    f"""
                    <div class="bar-row">
                        <div class="bar-name">{label}</div>
                        <div class="bar-track">
                            <div class="bar-fill" style="width:{max(pct, 1.5)}%; background:{s['color']};"></div>
                        </div>
                        <div class="bar-pct">{pct:.2f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="footer-wrap">
        <p class="footer-text">
            Built by <strong style="color:#e2e2ee;">Gaurav Gupta</strong> &nbsp;·&nbsp;
            <a href="https://www.linkedin.com/in/gaurav-gupta-79754a377" target="_blank">LinkedIn Profile</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
