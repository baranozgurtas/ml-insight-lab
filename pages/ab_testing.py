import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
import math

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    :root{--bg:#0a0a0f;--surface:#111118;--border:#2a2a3a;--accent:#6c63ff;--accent2:#00d4aa;--accent3:#ff6b6b;--text:#e8e8f0;--text-muted:#7a7a9a;--font-mono:'JetBrains Mono',monospace;--font-main:'Space Grotesk',sans-serif;}
    html,body,[class*="css"]{font-family:var(--font-main);background-color:var(--bg);color:var(--text);}
    .stApp{background-color:var(--bg);}
    section[data-testid="stSidebar"]{background-color:var(--surface);border-right:1px solid var(--border);}
    .page-title{font-family:var(--font-mono);font-size:2rem;font-weight:700;color:var(--accent2);margin-bottom:4px;}
    .page-sub{color:var(--text-muted);font-size:0.9rem;margin-bottom:30px;}
    .section-label{font-family:var(--font-mono);font-size:0.65rem;letter-spacing:2px;text-transform:uppercase;color:var(--text-muted);margin-bottom:10px;border-bottom:1px solid var(--border);padding-bottom:6px;}
    .stat-pill{display:inline-block;padding:5px 14px;border-radius:20px;font-family:var(--font-mono);font-size:0.72rem;font-weight:600;margin:3px;}
    .sig{background:rgba(0,212,170,0.15);color:#00d4aa;border:1px solid rgba(0,212,170,0.3);}
    .not-sig{background:rgba(255,107,107,0.15);color:#ff6b6b;border:1px solid rgba(255,107,107,0.3);}
    .neutral-pill{background:rgba(108,99,255,0.15);color:#6c63ff;border:1px solid rgba(108,99,255,0.3);}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">📊 A/B Testing Simulator</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Statistically rigorous model comparison — no gut calls</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🧪 Run A/B Test", "📐 Sample Size Calculator", "📖 Concepts"])

with tab1:
    col_a, col_b, col_params = st.columns([2, 2, 2], gap="large")

    with col_a:
        st.markdown('<div class="section-label">Model A — Baseline</div>', unsafe_allow_html=True)
        model_a_name = st.text_input("Model A Name", value="Logistic Regression", key="name_a")
        n_a = st.number_input("Sample Size", min_value=30, max_value=100000, value=500, step=50, key="n_a")
        acc_a = st.slider("Accuracy (%)", 50.0, 99.9, 72.0, 0.1, key="acc_a")
        st.caption(f"Successes: {int(n_a * acc_a / 100):,} / {n_a:,}")

    with col_b:
        st.markdown('<div class="section-label">Model B — Challenger</div>', unsafe_allow_html=True)
        model_b_name = st.text_input("Model B Name", value="Random Forest", key="name_b")
        n_b = st.number_input("Sample Size", min_value=30, max_value=100000, value=500, step=50, key="n_b")
        acc_b = st.slider("Accuracy (%)", 50.0, 99.9, 75.5, 0.1, key="acc_b")
        st.caption(f"Successes: {int(n_b * acc_b / 100):,} / {n_b:,}")

    with col_params:
        st.markdown('<div class="section-label">Test Parameters</div>', unsafe_allow_html=True)
        alpha = st.selectbox("Significance Level (α)", [0.01, 0.05, 0.10], index=1,
                             format_func=lambda x: f"{x} ({int((1-x)*100)}% confidence)")
        tail = st.radio("Test Type", ["Two-tailed", "One-tailed (B > A)"])
        metric_name = st.text_input("Metric Label", value="Accuracy")

    st.markdown("---")

    p_a = acc_a / 100
    p_b = acc_b / 100
    successes_a = int(n_a * p_a)
    successes_b = int(n_b * p_b)

    p_pooled = (successes_a + successes_b) / (n_a + n_b)
    se = math.sqrt(p_pooled * (1 - p_pooled) * (1/n_a + 1/n_b))
    z_stat = (p_b - p_a) / se if se > 0 else 0

    if tail == "Two-tailed":
        p_value = 2 * (1 - norm.cdf(abs(z_stat)))
        z_critical = norm.ppf(1 - alpha / 2)
    else:
        p_value = 1 - norm.cdf(z_stat)
        z_critical = norm.ppf(1 - alpha)

    is_significant = p_value < alpha
    lift = (p_b - p_a) / p_a * 100 if p_a > 0 else 0

    se_diff = math.sqrt(p_a*(1-p_a)/n_a + p_b*(1-p_b)/n_b)
    ci_z = norm.ppf(1 - alpha/2)
    ci_low = (p_b - p_a) - ci_z * se_diff
    ci_high = (p_b - p_a) + ci_z * se_diff

    cohens_h = abs(2 * math.asin(math.sqrt(max(0, min(1, p_b)))) - 2 * math.asin(math.sqrt(max(0, min(1, p_a)))))
    effect_label = "SMALL" if cohens_h < 0.2 else ("MEDIUM" if cohens_h < 0.5 else "LARGE")
    power = norm.cdf(abs(z_stat) - z_critical) + norm.cdf(-abs(z_stat) - z_critical)

    r1, r2, r3 = st.columns([2, 2, 3], gap="large")

    with r1:
        a_wins = p_a > p_b
        border = "2px solid #00d4aa" if a_wins else "1px solid #ff6b6b"
        bg = "rgba(0,212,170,0.08)" if a_wins else "rgba(255,107,107,0.06)"
        color = "#00d4aa" if a_wins else "#ff6b6b"
        st.markdown(f"""
        <div style="background:{bg};border:{border};border-radius:12px;padding:24px;text-align:center">
            <div style="font-family:'JetBrains Mono';font-size:0.65rem;letter-spacing:2px;text-transform:uppercase;color:#7a7a9a;margin-bottom:6px">Model A</div>
            <div style="font-size:1.1rem;font-weight:600;margin-bottom:12px">{model_a_name}</div>
            <div style="font-family:'JetBrains Mono';font-size:2.8rem;font-weight:700;color:{color}">{acc_a:.1f}%</div>
            <div style="color:#7a7a9a;font-size:0.8rem;margin-top:8px">{metric_name}</div>
            {'<div style="font-family:JetBrains Mono;font-size:0.7rem;color:#00d4aa;margin-top:8px">▲ BASELINE WINS</div>' if a_wins else ''}
        </div>""", unsafe_allow_html=True)

    with r2:
        b_wins = p_b > p_a
        border = "2px solid #00d4aa" if b_wins else "1px solid #ff6b6b"
        bg = "rgba(0,212,170,0.08)" if b_wins else "rgba(255,107,107,0.06)"
        color = "#00d4aa" if b_wins else "#ff6b6b"
        st.markdown(f"""
        <div style="background:{bg};border:{border};border-radius:12px;padding:24px;text-align:center">
            <div style="font-family:'JetBrains Mono';font-size:0.65rem;letter-spacing:2px;text-transform:uppercase;color:#7a7a9a;margin-bottom:6px">Model B</div>
            <div style="font-size:1.1rem;font-weight:600;margin-bottom:12px">{model_b_name}</div>
            <div style="font-family:'JetBrains Mono';font-size:2.8rem;font-weight:700;color:{color}">{acc_b:.1f}%</div>
            <div style="color:#7a7a9a;font-size:0.8rem;margin-top:8px">{metric_name}</div>
            {'<div style="font-family:JetBrains Mono;font-size:0.7rem;color:#00d4aa;margin-top:8px">▲ CHALLENGER WINS</div>' if b_wins else ''}
        </div>""", unsafe_allow_html=True)

    with r3:
        sig_class = "sig" if is_significant else "not-sig"
        sig_text = "SIGNIFICANT" if is_significant else "NOT SIGNIFICANT"
        st.markdown(f"""
        <div style="background:#111118;border:1px solid #2a2a3a;border-radius:12px;padding:24px">
            <div style="font-family:'JetBrains Mono';font-size:0.65rem;letter-spacing:2px;text-transform:uppercase;color:#7a7a9a;margin-bottom:10px">Statistical Results</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px">
                <div><div style="font-family:'JetBrains Mono';font-size:0.6rem;color:#7a7a9a;text-transform:uppercase">p-value</div>
                    <div style="font-family:'JetBrains Mono';font-size:1.4rem;font-weight:700;color:{'#00d4aa' if is_significant else '#ff6b6b'}">{p_value:.4f}</div></div>
                <div><div style="font-family:'JetBrains Mono';font-size:0.6rem;color:#7a7a9a;text-transform:uppercase">z-statistic</div>
                    <div style="font-family:'JetBrains Mono';font-size:1.4rem;font-weight:700">{z_stat:.3f}</div></div>
                <div><div style="font-family:'JetBrains Mono';font-size:0.6rem;color:#7a7a9a;text-transform:uppercase">Lift (B vs A)</div>
                    <div style="font-family:'JetBrains Mono';font-size:1.4rem;font-weight:700;color:{'#00d4aa' if lift>=0 else '#ff6b6b'}">{lift:+.1f}%</div></div>
                <div><div style="font-family:'JetBrains Mono';font-size:0.6rem;color:#7a7a9a;text-transform:uppercase">Effect Size</div>
                    <div style="font-family:'JetBrains Mono';font-size:1.4rem;font-weight:700;color:#6c63ff">{effect_label}</div></div>
            </div>
            <div style="margin-top:14px">
                <span class="stat-pill {sig_class}">{sig_text}</span>
                <span class="stat-pill neutral-pill">α = {alpha}</span>
                <span class="stat-pill neutral-pill">Power ≈ {min(power,1):.0%}</span>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:20px">Confidence Interval for Difference (B - A)</div>', unsafe_allow_html=True)
    fig_ci = go.Figure()
    fig_ci.add_shape(type="line", x0=0, x1=0, y0=-0.3, y1=0.3, line=dict(color="#ff6b6b", width=2, dash="dash"))
    fig_ci.add_trace(go.Scatter(x=[ci_low, ci_high], y=[0, 0], mode='lines',
                                line=dict(color='#6c63ff', width=4), showlegend=False, hoverinfo='skip'))
    fig_ci.add_trace(go.Scatter(x=[p_b - p_a], y=[0], mode='markers',
                                marker=dict(color='#00d4aa', size=14, symbol='diamond'),
                                name='Observed diff', hovertemplate=f"Δ = {p_b-p_a:+.4f}<extra></extra>"))
    fig_ci.add_annotation(x=ci_low, y=0.18, text=f"{ci_low:+.4f}", showarrow=False,
                          font=dict(family='JetBrains Mono', size=10, color='#7a7a9a'))
    fig_ci.add_annotation(x=ci_high, y=0.18, text=f"{ci_high:+.4f}", showarrow=False,
                          font=dict(family='JetBrains Mono', size=10, color='#7a7a9a'))
    fig_ci.update_layout(paper_bgcolor='#0a0a0f', plot_bgcolor='#111118',
                         font=dict(family='JetBrains Mono', color='#e8e8f0'), height=130,
                         margin=dict(l=20, r=20, t=10, b=10),
                         xaxis=dict(gridcolor='#2a2a3a', zeroline=False, title='Difference in Proportions'),
                         yaxis=dict(showticklabels=False, range=[-0.5, 0.5], gridcolor='#1a1a24'))
    st.plotly_chart(fig_ci, use_container_width=True)

    st.markdown('<div class="section-label">Z-Distribution & Critical Region</div>', unsafe_allow_html=True)
    x_range = np.linspace(-4, 4, 400)
    y_range = norm.pdf(x_range)

    fig_z = go.Figure()
    fig_z.add_trace(go.Scatter(x=x_range, y=y_range, fill='tozeroy',
                               fillcolor='rgba(108,99,255,0.08)', line=dict(color='#6c63ff', width=1.5),
                               showlegend=False, hoverinfo='skip'))
    if tail == "Two-tailed":
        for mask in [x_range <= -z_critical, x_range >= z_critical]:
            fig_z.add_trace(go.Scatter(x=x_range[mask], y=y_range[mask], fill='tozeroy',
                                       fillcolor='rgba(255,107,107,0.25)', line=dict(color='rgba(0,0,0,0)'),
                                       showlegend=False, hoverinfo='skip'))
    else:
        mask = x_range >= z_critical
        fig_z.add_trace(go.Scatter(x=x_range[mask], y=y_range[mask], fill='tozeroy',
                                   fillcolor='rgba(255,107,107,0.25)', line=dict(color='rgba(0,0,0,0)'),
                                   showlegend=False, hoverinfo='skip'))
    fig_z.add_shape(type="line", x0=z_stat, x1=z_stat, y0=0, y1=norm.pdf(z_stat),
                    line=dict(color='#00d4aa', width=2.5))
    fig_z.add_annotation(x=z_stat, y=norm.pdf(z_stat) + 0.03, text=f"z={z_stat:.2f}",
                         showarrow=False, font=dict(family='JetBrains Mono', size=10, color='#00d4aa'))
    fig_z.update_layout(paper_bgcolor='#0a0a0f', plot_bgcolor='#111118',
                        font=dict(family='JetBrains Mono', color='#e8e8f0'), height=200,
                        margin=dict(l=20, r=20, t=10, b=30),
                        xaxis=dict(gridcolor='#2a2a3a', zeroline=True, zerolinecolor='#2a2a3a', title='Z-score'),
                        yaxis=dict(gridcolor='#2a2a3a', title=''))
    st.plotly_chart(fig_z, use_container_width=True)

    if is_significant and b_wins:
        verdict = f"✅ <b>Deploy Model B ({model_b_name})</b>. Challenger outperforms baseline by {lift:+.1f}% (p={p_value:.4f} < α={alpha}). {int((1-alpha)*100)}% CI: [{ci_low:+.4f}, {ci_high:+.4f}]. Effect size: {effect_label.lower()}."
        vc = "#00d4aa"
    elif is_significant and a_wins:
        verdict = f"⚠️ <b>Keep Model A ({model_a_name})</b>. Baseline is significantly better (p={p_value:.4f}). Model B performs worse by {abs(lift):.1f}%."
        vc = "#f0c040"
    else:
        verdict = f"⏳ <b>Inconclusive.</b> Not statistically significant (p={p_value:.4f} ≥ α={alpha}). Collect more data — current power ≈ {min(power,1):.0%}."
        vc = "#7a7a9a"

    st.markdown(f"""
    <div style="background:#111118;border:1px solid #2a2a3a;border-left:3px solid {vc};border-radius:12px;padding:24px;margin:16px 0">
        <div style="font-family:'JetBrains Mono';font-size:0.65rem;letter-spacing:2px;text-transform:uppercase;color:#7a7a9a;margin-bottom:10px">Verdict</div>
        <div style="font-size:1.05rem;line-height:1.7">{verdict}</div>
    </div>""", unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-label">Required Sample Size</div>', unsafe_allow_html=True)
    sc1, sc2 = st.columns(2, gap="large")
    with sc1:
        baseline_rate = st.slider("Baseline Metric (%)", 50.0, 99.0, 72.0, 0.5)
        mde = st.slider("Minimum Detectable Effect (%)", 0.5, 20.0, 3.0, 0.5)
        alpha_ss = st.selectbox("Significance Level", [0.01, 0.05, 0.10], index=1, key="alpha_ss")
        power_target = st.selectbox("Statistical Power", [0.80, 0.85, 0.90, 0.95], index=0)
    with sc2:
        p1 = baseline_rate / 100
        p2 = min(p1 + mde / 100, 0.9999)
        z_a = norm.ppf(1 - alpha_ss / 2)
        z_b = norm.ppf(power_target)
        p_avg = (p1 + p2) / 2
        n_req = math.ceil((z_a*math.sqrt(2*p_avg*(1-p_avg)) + z_b*math.sqrt(p1*(1-p1)+p2*(1-p2)))**2 / (p2-p1)**2)
        st.markdown(f"""
        <div style="background:#111118;border:1px solid #2a2a3a;border-radius:12px;padding:28px;text-align:center">
            <div style="font-family:'JetBrains Mono';font-size:0.65rem;letter-spacing:2px;text-transform:uppercase;color:#7a7a9a;margin-bottom:10px">Required per group</div>
            <div style="font-family:'JetBrains Mono';font-size:3.5rem;font-weight:700;color:#6c63ff;line-height:1">{n_req:,}</div>
            <div style="font-family:'JetBrains Mono';font-size:0.8rem;color:#7a7a9a;margin-top:6px">samples</div>
            <div style="margin-top:16px;font-family:'JetBrains Mono';font-size:0.75rem;color:#7a7a9a">Total: <span style="color:#e8e8f0;font-weight:600">{n_req*2:,}</span> across both groups</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:20px">Sample Size vs MDE Tradeoff</div>', unsafe_allow_html=True)
    mde_range = np.arange(0.5, 15.5, 0.5)
    n_range = []
    for m in mde_range:
        p2_ = min(p1 + m/100, 0.9999)
        p_avg_ = (p1 + p2_) / 2
        n_ = math.ceil((z_a*math.sqrt(2*p_avg_*(1-p_avg_)) + z_b*math.sqrt(p1*(1-p1)+p2_*(1-p2_)))**2 / (p2_-p1)**2)
        n_range.append(n_)
    fig_mde = go.Figure()
    fig_mde.add_trace(go.Scatter(x=mde_range, y=n_range, mode='lines',
                                 line=dict(color='#6c63ff', width=2.5),
                                 fill='tozeroy', fillcolor='rgba(108,99,255,0.06)',
                                 hovertemplate='MDE: %{x:.1f}%<br>n: %{y:,}<extra></extra>'))
    fig_mde.add_shape(type="line", x0=mde, x1=mde, y0=0, y1=n_req,
                      line=dict(color='#00d4aa', width=1.5, dash='dash'))
    fig_mde.add_trace(go.Scatter(x=[mde], y=[n_req], mode='markers',
                                 marker=dict(color='#00d4aa', size=10, symbol='diamond'),
                                 showlegend=False, hoverinfo='skip'))
    fig_mde.update_layout(paper_bgcolor='#0a0a0f', plot_bgcolor='#111118',
                          font=dict(family='JetBrains Mono', color='#e8e8f0'), height=250,
                          margin=dict(l=20,r=20,t=10,b=30),
                          xaxis=dict(title='MDE (%)', gridcolor='#2a2a3a'),
                          yaxis=dict(title='n per group', gridcolor='#2a2a3a'))
    st.plotly_chart(fig_mde, use_container_width=True)

with tab3:
    concepts = [
        ("p-value", "#6c63ff", "Probability of observing your results (or more extreme) if the null hypothesis were true. Small p-value = evidence against null. Does NOT mean probability your model is better."),
        ("Significance Level (α)", "#00d4aa", "Threshold for rejecting H₀. α=0.05 means you accept 5% false positive risk (Type I error). Lower α = more stringent test."),
        ("Statistical Power (1-β)", "#ff6b6b", "Probability of correctly detecting a real effect. Power=0.80 means you detect a true improvement 80% of the time."),
        ("Confidence Interval", "#6c63ff", "Range containing the true parameter with (1-α)% probability. If CI for (B-A) excludes 0, the difference is significant."),
        ("Effect Size (Cohen's h)", "#00d4aa", "Practical magnitude of difference, independent of sample size. Small: h<0.2, Medium: 0.2–0.5, Large: >0.5."),
        ("Type I vs Type II Error", "#ff6b6b", "Type I (false positive): reject H₀ when true (prob=α). Type II (false negative): fail to reject H₀ when false (prob=β)."),
        ("Lift", "#6c63ff", "Relative improvement of B over A. Lift=(B-A)/A×100%. Always pair with significance test — 10% lift on 30 samples is meaningless."),
        ("MDE", "#00d4aa", "Minimum Detectable Effect: smallest improvement your test is designed to detect. Smaller MDE = more samples needed."),
    ]
    cols = st.columns(2, gap="large")
    for i, (title, color, desc) in enumerate(concepts):
        with cols[i % 2]:
            st.markdown(f"""
            <div style="background:#111118;border:1px solid #2a2a3a;border-left:3px solid {color};border-radius:8px;padding:18px;margin-bottom:12px">
                <div style="font-family:'JetBrains Mono';font-size:0.8rem;font-weight:700;color:{color};margin-bottom:6px">{title}</div>
                <div style="color:#7a7a9a;font-size:0.85rem;line-height:1.7">{desc}</div>
            </div>""", unsafe_allow_html=True)