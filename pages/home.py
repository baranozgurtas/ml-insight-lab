import streamlit as st

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

    :root {
        --bg: #0a0a0f; --surface: #111118; --border: #2a2a3a;
        --accent: #6c63ff; --accent2: #00d4aa; --accent3: #ff6b6b;
        --text: #e8e8f0; --text-muted: #7a7a9a;
        --font-mono: 'JetBrains Mono', monospace;
        --font-main: 'Space Grotesk', sans-serif;
    }
    html, body, [class*="css"] { font-family: var(--font-main); background-color: var(--bg); color: var(--text); }
    .stApp { background-color: var(--bg); }
    section[data-testid="stSidebar"] { background-color: var(--surface); border-right: 1px solid var(--border); }

    .main-header { text-align: center; padding: 60px 20px 40px; }
    .main-header h1 {
        font-family: var(--font-mono); font-size: 3.5rem; font-weight: 700;
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; letter-spacing: -2px; margin-bottom: 0.3rem;
    }
    .main-header .tagline {
        font-family: var(--font-mono); color: var(--text-muted);
        font-size: 0.9rem; letter-spacing: 3px; text-transform: uppercase;
    }
    .card {
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 12px; padding: 28px; margin: 12px 0;
    }
    .card h3 { font-family: var(--font-mono); color: var(--accent2); font-size: 0.75rem; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 8px; }
    .card h2 { font-size: 1.4rem; font-weight: 600; margin-bottom: 12px; }
    .card p { color: var(--text-muted); line-height: 1.7; font-size: 0.95rem; }
    .badge { display: inline-block; padding: 3px 10px; border-radius: 4px; font-family: var(--font-mono); font-size: 0.7rem; font-weight: 600; letter-spacing: 1px; margin-right: 6px; margin-top: 8px; }
    .badge-purple { background: rgba(108,99,255,0.15); color: var(--accent); border: 1px solid rgba(108,99,255,0.3); }
    .badge-green { background: rgba(0,212,170,0.15); color: var(--accent2); border: 1px solid rgba(0,212,170,0.3); }
    .badge-red { background: rgba(255,107,107,0.15); color: var(--accent3); border: 1px solid rgba(255,107,107,0.3); }
    div[data-testid="stButton"] button {
        background: transparent; border: 1px solid var(--accent); color: var(--accent);
        font-family: var(--font-mono); font-size: 0.8rem; letter-spacing: 1px;
        border-radius: 6px; padding: 10px 20px; transition: all 0.2s; margin-top: 8px;
    }
    div[data-testid="stButton"] button:hover { background: var(--accent); color: white; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🧪 ML Insight Lab</h1>
    <div class="tagline">Interactive Machine Learning Experimentation Platform</div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class="card">
        <h3>Module 01</h3>
        <h2>⚡ Model Behavior Playground</h2>
        <p>Explore how ML models behave under different conditions. Adjust noise, sample size,
        and dataset complexity — watch decision boundaries shift in real time.</p>
        <div>
            <span class="badge badge-purple">DECISION BOUNDARIES</span>
            <span class="badge badge-green">BIAS/VARIANCE</span>
            <span class="badge badge-red">OVERFITTING</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Launch Playground →", key="btn_playground", use_container_width=True):
        st.switch_page("pages/model_playground.py")

with col2:
    st.markdown("""
    <div class="card">
        <h3>Module 02</h3>
        <h2>📊 A/B Testing Simulator</h2>
        <p>Compare two model versions with statistical rigor. Calculate p-values, confidence
        intervals, and required sample sizes. Make data-driven decisions, not gut calls.</p>
        <div>
            <span class="badge badge-purple">P-VALUE</span>
            <span class="badge badge-green">CONFIDENCE INTERVAL</span>
            <span class="badge badge-red">EFFECT SIZE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Launch A/B Tester →", key="btn_ab", use_container_width=True):
        st.switch_page("pages/ab_testing.py")