import streamlit as st
import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="🐦",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# CUSTOM CSS — dark themed / purple-blue gradient, matches reference design
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background-color: #08080c;
            background-image:
                linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px);
            background-size: 42px 42px;
        }

        /* hide default streamlit chrome */
        #MainMenu, footer, header {visibility: hidden;}

        .block-container {
            padding-top: 3rem;
            max-width: 760px;
        }

        /* ---------------- Eyebrow tag ---------------- */
        .eyebrow {
            text-align: center;
            letter-spacing: 4px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            font-weight: 600;
            color: #8b8bd8;
            margin-bottom: 18px;
        }

        /* ---------------- Hero title ---------------- */
        .hero-title {
            text-align: center;
            font-weight: 800;
            font-size: 56px;
            line-height: 1.08;
            color: #f5f5fa;
            margin: 0;
        }
        .hero-gradient {
            text-align: center;
            font-weight: 800;
            font-size: 56px;
            line-height: 1.15;
            margin: 0 0 22px 0;
            background: linear-gradient(90deg, #8b7bf0 0%, #7aa2f7 60%, #7ad0f7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .hero-sub {
            text-align: center;
            color: #9b9bb0;
            font-size: 16px;
            max-width: 560px;
            margin: 0 auto 40px auto;
            line-height: 1.6;
        }

        /* ---------------- Card container ---------------- */
        .card {
            background: rgba(18, 18, 26, 0.75);
            border: 1px solid rgba(140, 130, 220, 0.18);
            border-radius: 16px;
            padding: 26px 26px 20px 26px;
            margin-bottom: 22px;
            backdrop-filter: blur(6px);
        }

        .field-label {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11.5px;
            letter-spacing: 2px;
            color: #7d7d95;
            margin-bottom: 10px;
            font-weight: 600;
        }

        /* ---------------- Text area override ---------------- */
        .stTextArea textarea {
            background-color: #101018 !important;
            border: 1px solid rgba(140, 130, 220, 0.25) !important;
            border-radius: 10px !important;
            color: #f0f0f5 !important;
            font-size: 15px !important;
            padding: 14px !important;
        }
        .stTextArea textarea:focus {
            border: 1px solid #8b7bf0 !important;
            box-shadow: 0 0 0 1px #8b7bf0 !important;
        }

        /* ---------------- Button override ---------------- */
        div.stButton > button {
            width: 100%;
            background: linear-gradient(90deg, #7c6df0 0%, #6f8ef2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 13px 0;
            font-weight: 700;
            font-size: 15.5px;
            letter-spacing: 0.3px;
            transition: all 0.15s ease-in-out;
            box-shadow: 0 6px 18px rgba(124, 109, 240, 0.35);
        }
        div.stButton > button:hover {
            filter: brightness(1.08);
            box-shadow: 0 8px 22px rgba(124, 109, 240, 0.5);
            transform: translateY(-1px);
        }
        div.stButton > button:active {
            transform: translateY(0px);
        }

        /* ---------------- Result card ---------------- */
        .result-header {
            display: flex;
            align-items: center;
            gap: 16px;
            margin-bottom: 18px;
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
            font-size: 22px;
            border: 1.5px solid;
        }
        .result-title {
            font-size: 26px;
            font-weight: 800;
            color: #f5f5fa;
            margin: 0;
            line-height: 1.2;
        }
        .result-conf {
            font-family: 'JetBrains Mono', monospace;
            font-size: 14px;
            font-weight: 600;
            margin-top: 2px;
        }

        .top-pred-label {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            letter-spacing: 2px;
            color: #7d7d95;
            margin: 18px 0 12px 0;
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
            width: 90px;
            min-width: 90px;
        }
        .bar-track {
            flex: 1;
            height: 9px;
            background: #1a1a24;
            border-radius: 5px;
            overflow: hidden;
        }
        .bar-fill {
            height: 100%;
            border-radius: 5px;
        }
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
            margin-top: 48px;
            padding-top: 22px;
            border-top: 1px solid rgba(140, 130, 220, 0.14);
        }
        .footer-text {
            color: #6b6b80;
            font-size: 13.5px;
            font-family: 'Inter', sans-serif;
        }
        .footer-text a {
            color: #9c95f5;
            text-decoration: none;
            font-weight: 600;
        }
        .footer-text a:hover {
            text-decoration: underline;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# LABEL CONFIG
# ---------------------------------------------------------------------------
LABELS = ["Negative", "Positive", "Neutral", "Irrelevant"]

LABEL_STYLE = {
    "Negative":   {"color": "#f76b7c", "bg": "rgba(247,107,124,0.12)", "border": "#f76b7c", "short": "NEG"},
    "Positive":   {"color": "#5be08f", "bg": "rgba(91,224,143,0.12)",  "border": "#5be08f", "short": "POS"},
    "Neutral":    {"color": "#7aa2f7", "bg": "rgba(122,162,247,0.12)", "border": "#7aa2f7", "short": "NEU"},
    "Irrelevant": {"color": "#c0a8f7", "bg": "rgba(192,168,247,0.12)","border": "#c0a8f7", "short": "IRR"},
}

# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------
MODEL_PATH = "sentiment_model.keras"  # exported from the notebook via model.save("sentiment_model.keras")


@st.cache_resource(show_spinner=False)
def load_model(path):
    try:
        return tf.keras.models.load_model(path)
    except Exception:
        return None


model = load_model(MODEL_PATH)

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown('<div class="eyebrow">DEEP LEARNING &nbsp;·&nbsp; NLP</div>', unsafe_allow_html=True)
st.markdown('<p class="hero-title">Detect Tweet</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-gradient">Sentiment</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">A Bidirectional LSTM model trained on tweets to classify '
    'sentiment as Positive, Negative, Neutral, or Irrelevant.</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# INPUT CARD
# ---------------------------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="field-label">ENTER A TWEET OR ANY TEXT</div>', unsafe_allow_html=True)

text_input = st.text_area(
    label="tweet_input",
    label_visibility="collapsed",
    placeholder="e.g. I would never recommend this terrible service to anyone.",
    height=120,
)

analyze_clicked = st.button("Detect Sentiment")
st.markdown('</div>', unsafe_allow_html=True)

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

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="result-header">
                <div class="result-badge" style="background:{style['bg']}; color:{style['color']}; border-color:{style['border']};">
                    {style['short']}
                </div>
                <div>
                    <p class="result-title">{top_label}</p>
                    <p class="result-conf" style="color:{style['color']};">{top_conf:.2f}% confidence</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="top-pred-label">ALL PREDICTIONS</div>', unsafe_allow_html=True)

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

        st.markdown('</div>', unsafe_allow_html=True)

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