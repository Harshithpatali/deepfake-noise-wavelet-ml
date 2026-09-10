import base64
import textwrap
import os
import streamlit as st
import requests


API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)


def html(block: str) -> None:
    """Render an HTML block, stripped of Python-source indentation.

    Streamlit's markdown renderer treats lines indented 4+ spaces as a
    code block. Since these blocks are written as indented triple-quoted
    strings, dedenting is required or tags render as literal text.
    """
    st.markdown(textwrap.dedent(block).strip(), unsafe_allow_html=True)

st.set_page_config(page_title="DeepFake Forensics", page_icon="🔎", layout="wide")

# ----------------------------------------------------------------------------
# THEME
# Palette: near-black lab background, cyan-teal for "authentic" signal,
# signal-red for "synthetic" anomaly, slate greys for structure.
# Type: Space Grotesk for display, JetBrains Mono for data readouts.
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg: #080B10;
        --panel: #0E141C;
        --panel-line: #1C2733;
        --teal: #23E8C6;
        --red: #FF4462;
        --ink: #DCE4EC;
        --ink-dim: #6E7A87;
    }

    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(35,232,198,0.05), transparent 40%),
            radial-gradient(circle at 85% 90%, rgba(255,68,98,0.04), transparent 40%),
            repeating-linear-gradient(0deg, rgba(255,255,255,0.012) 0px, rgba(255,255,255,0.012) 1px, transparent 1px, transparent 34px),
            var(--bg);
        color: var(--ink);
    }

    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 2.5rem; max-width: 900px; }

    /* ---------- Header ---------- */
    .lab-header { margin-bottom: 0.3rem; }
    .lab-kicker {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        color: var(--teal);
        margin-bottom: 0.4rem;
    }
    .lab-title {
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        line-height: 1.15;
        margin: 0;
    }
    .lab-sub {
        color: var(--ink-dim);
        font-size: 0.98rem;
        max-width: 560px;
        margin-top: 0.5rem;
        line-height: 1.5;
    }

    /* ---------- Pipeline strip ---------- */
    .pipeline {
        display: flex;
        border: 1px solid var(--panel-line);
        border-radius: 10px;
        overflow: hidden;
        margin: 1.8rem 0 2rem 0;
        background: var(--panel);
    }
    .pipeline .stage {
        flex: 1;
        padding: 0.9rem 1.1rem;
        border-right: 1px solid var(--panel-line);
    }
    .pipeline .stage:last-child { border-right: none; }
    .stage .num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: var(--teal);
        opacity: 0.85;
    }
    .stage .name {
        font-size: 0.86rem;
        margin-top: 0.15rem;
        color: var(--ink);
    }

    /* ---------- Upload zone ---------- */
    [data-testid="stFileUploaderDropzone"] {
        background: var(--panel) !important;
        border: 1px dashed var(--panel-line) !important;
        border-radius: 10px !important;
    }
    [data-testid="stFileUploaderDropzone"] * { color: var(--ink-dim) !important; }

    /* ---------- Scan frame around preview image ---------- */
    .scan-frame {
        position: relative;
        border: 1px solid var(--panel-line);
        border-radius: 10px;
        overflow: hidden;
        background: var(--panel);
        margin-top: 1rem;
    }
    .scan-frame img { display: block; width: 100%; }
    .scan-frame::before {
        content: "";
        position: absolute;
        left: 0; right: 0; top: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--teal), transparent);
        box-shadow: 0 0 8px 1px rgba(35,232,198,0.7);
        animation: scan 2.6s linear infinite;
    }
    @keyframes scan { 0% { top: 0%; } 100% { top: 100%; } }
    .scan-caption {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: var(--ink-dim);
        padding: 0.5rem 0.8rem;
        border-top: 1px solid var(--panel-line);
    }

    /* ---------- Analyze button ---------- */
    .stButton>button {
        background: transparent;
        border: 1px solid var(--teal);
        color: var(--teal);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        letter-spacing: 0.04em;
        border-radius: 8px;
        padding: 0.5rem 1.4rem;
        transition: background 0.15s ease, color 0.15s ease;
    }
    .stButton>button:hover {
        background: var(--teal);
        color: #08120F;
    }

    /* ---------- Result panel ---------- */
    .verdict-card {
        border: 1px solid var(--panel-line);
        border-radius: 12px;
        background: var(--panel);
        padding: 1.4rem 1.6rem;
        margin-top: 1.6rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.5rem;
        flex-wrap: wrap;
    }
    .verdict-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: var(--ink-dim);
        letter-spacing: 0.1em;
    }
    .verdict-value {
        font-size: 1.9rem;
        font-weight: 700;
        margin-top: 0.15rem;
    }
    .verdict-value.real { color: var(--teal); }
    .verdict-value.fake { color: var(--red); }

    .gauge-wrap { display: flex; flex-direction: column; align-items: center; }
    .gauge {
        width: 110px; height: 110px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        position: relative;
    }
    .gauge::before {
        content: "";
        position: absolute; inset: 8px;
        border-radius: 50%;
        background: var(--panel);
    }
    .gauge-value {
        position: relative;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--ink);
    }
    .gauge-caption {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: var(--ink-dim);
        margin-top: 0.5rem;
        letter-spacing: 0.06em;
    }

    /* ---------- Threshold scale ---------- */
    .scale-block {
        margin-top: 1.6rem;
        border: 1px solid var(--panel-line);
        border-radius: 12px;
        background: var(--panel);
        padding: 1.2rem 1.5rem 1.4rem 1.5rem;
    }
    .scale-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: var(--ink-dim);
        letter-spacing: 0.08em;
        margin-bottom: 0.8rem;
    }
    .scale-track {
        position: relative;
        height: 6px;
        border-radius: 3px;
        background: linear-gradient(90deg, var(--teal), var(--panel-line) 50%, var(--red));
        margin: 26px 6px 0 6px;
    }
    .scale-tick {
        position: absolute;
        top: -6px;
        width: 2px; height: 18px;
        background: var(--ink-dim);
        opacity: 0.6;
    }
    .scale-marker {
        position: absolute;
        top: -12px;
        width: 2px; height: 30px;
        background: var(--ink);
    }
    .scale-marker-label {
        position: absolute;
        left: 50%;
        transform: translateX(-50%);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        color: var(--ink);
        white-space: nowrap;
    }
    .scale-labels {
        display: flex; justify-content: space-between;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        color: var(--ink-dim);
        margin-top: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
html("""
    <div class="lab-header">
        <div class="lab-kicker">SIGNAL FORENSICS UNIT</div>
        <h1 class="lab-title">DeepFake Detector</h1>
        <p class="lab-sub">
            Analyzes a face image for synthetic-generation traces by isolating the
            camera noise residual, decomposing it across wavelet sub-bands, and
            scoring the result with a trained classifier.
        </p>
    </div>
    <div class="pipeline">
        <div class="stage"><div class="num">01</div><div class="name">Noise residual extraction</div></div>
        <div class="stage"><div class="num">02</div><div class="name">Wavelet decomposition</div></div>
        <div class="stage"><div class="num">03</div><div class="name">ML classification</div></div>
    </div>
""")

# ----------------------------------------------------------------------------
# UPLOAD
# ----------------------------------------------------------------------------
up = st.file_uploader(
    "Upload a face image",
    type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
    label_visibility="collapsed",
)


def radial_gauge(pct: float, color: str, caption: str) -> str:
    pct_clamped = max(0.0, min(1.0, pct))
    deg = pct_clamped * 360
    return (
        '<div class="gauge-wrap">'
        f'<div class="gauge" style="background: conic-gradient({color} {deg}deg, var(--panel-line) {deg}deg);">'
        f'<div class="gauge-value">{pct_clamped * 100:.1f}%</div>'
        "</div>"
        f'<div class="gauge-caption">{caption}</div>'
        "</div>"
    )


if up:
    b64 = base64.b64encode(up.getvalue()).decode()
    mime = up.type or "image/jpeg"
    html(f"""
        <div class="scan-frame">
            <img src="data:{mime};base64,{b64}" />
            <div class="scan-caption">{up.name}</div>
        </div>
    """)

    st.write("")
    analyze = st.button("RUN ANALYSIS")

    if analyze:
        with st.spinner("Extracting noise residual and scoring..."):
            try:
                r = requests.post(
    f"{API_URL}/predict",
    files={
        "file": (
            up.name,
            up.getvalue(),
            up.type
        )
    },
    timeout=120,
)

                if r.status_code != 200:
                    st.error(f"Backend returned HTTP {r.status_code}")
                    st.code(r.text)
                    st.stop()

                x = r.json()
                prediction = str(x["prediction"])
                fake_p = float(x["fake_probability"])
                real_p = float(x["real_probability"])
                threshold = float(x["threshold"])

                is_fake = fake_p >= threshold
                verdict_class = "fake" if is_fake else "real"
                gauge_color = "var(--red)" if is_fake else "var(--teal)"

                html(f"""
                    <div class="verdict-card">
                        <div>
                            <div class="verdict-label">VERDICT</div>
                            <div class="verdict-value {verdict_class}">{prediction.upper()}</div>
                        </div>
                        <div style="display:flex; gap:1.5rem;">
                            {radial_gauge(fake_p, "var(--red)", "FAKE PROB.")}
                            {radial_gauge(real_p, "var(--teal)", "REAL PROB.")}
                        </div>
                    </div>
                """)

                fake_pct = max(0.0, min(1.0, fake_p)) * 100
                thresh_pct = max(0.0, min(1.0, threshold)) * 100

                # When the score and threshold markers sit close together,
                # stack their labels on separate rows so the text can't overlap.
                markers_close = abs(fake_pct - thresh_pct) < 14
                score_label_top = "-20px"
                thresh_label_top = "-38px" if markers_close else "-20px"

                html(f"""
                    <div class="scale-block">
                        <div class="scale-title">SCORE VS. FROZEN THRESHOLD</div>
                        <div class="scale-track">
                            <div class="scale-tick" style="left: 0%;"></div>
                            <div class="scale-tick" style="left: 50%;"></div>
                            <div class="scale-tick" style="left: 100%;"></div>
                            <div class="scale-marker" style="left: {thresh_pct}%; background: var(--ink-dim);">
                                <span class="scale-marker-label" style="top: {thresh_label_top};">THRESHOLD {threshold:.3f}</span>
                            </div>
                            <div class="scale-marker" style="left: {fake_pct}%; background: {gauge_color};">
                                <span class="scale-marker-label" style="top: {score_label_top};">SCORE {fake_p:.3f}</span>
                            </div>
                        </div>
                        <div class="scale-labels">
                            <span>0.0 — AUTHENTIC</span>
                            <span>1.0 — SYNTHETIC</span>
                        </div>
                    </div>
                """)

            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")