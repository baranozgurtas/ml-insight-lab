import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import make_moons, make_circles, make_blobs, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    :root { --bg:#0a0a0f;--surface:#111118;--border:#2a2a3a;--accent:#6c63ff;--accent2:#00d4aa;--accent3:#ff6b6b;--text:#e8e8f0;--text-muted:#7a7a9a;--font-mono:'JetBrains Mono',monospace;--font-main:'Space Grotesk',sans-serif; }
    html,body,[class*="css"]{font-family:var(--font-main);background-color:var(--bg);color:var(--text);}
    .stApp{background-color:var(--bg);}
    section[data-testid="stSidebar"]{background-color:var(--surface);border-right:1px solid var(--border);}
    h1,h2,h3{font-family:var(--font-mono);}
    .page-title{font-family:var(--font-mono);font-size:2rem;font-weight:700;color:var(--accent);margin-bottom:4px;}
    .page-sub{color:var(--text-muted);font-size:0.9rem;margin-bottom:30px;}
    .metric-box{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:18px 22px;text-align:center;}
    .metric-box .label{font-family:var(--font-mono);font-size:0.65rem;letter-spacing:2px;text-transform:uppercase;color:var(--text-muted);margin-bottom:6px;}
    .metric-box .value{font-family:var(--font-mono);font-size:1.8rem;font-weight:700;color:var(--accent2);}
    .metric-box .value.warn{color:var(--accent3);}
    .section-label{font-family:var(--font-mono);font-size:0.65rem;letter-spacing:2px;text-transform:uppercase;color:var(--text-muted);margin-bottom:10px;border-bottom:1px solid var(--border);padding-bottom:6px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">⚡ Model Behavior Playground</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Manipulate data & models — watch everything change in real time</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="section-label">Dataset</div>', unsafe_allow_html=True)
    dataset_type = st.selectbox("Shape", ["Moons", "Circles", "Blobs", "Linear"])
    n_samples = st.slider("Sample Size", 100, 1000, 300, 50)
    noise = st.slider("Noise Level", 0.0, 0.5, 0.15, 0.01)
    test_size = st.slider("Test Split", 0.1, 0.5, 0.2, 0.05)

    st.markdown('<div class="section-label" style="margin-top:20px">Model</div>', unsafe_allow_html=True)
    model_name = st.selectbox("Algorithm", [
        "Logistic Regression", "Decision Tree", "SVM (RBF)", "Random Forest", "Gradient Boosting"
    ])

    max_depth = 5
    n_estimators = 100
    max_depth_rf = 5
    n_estimators_gb = 100
    lr_gb = 0.1
    C_svm = 1.0

    if model_name == "Decision Tree":
        max_depth = st.slider("Max Depth", 1, 20, 5)
    elif model_name == "Random Forest":
        n_estimators = st.slider("N Estimators", 10, 200, 100, 10)
        max_depth_rf = st.slider("Max Depth", 1, 20, 5)
    elif model_name == "Gradient Boosting":
        n_estimators_gb = st.slider("N Estimators", 10, 200, 100, 10)
        lr_gb = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
    elif model_name == "SVM (RBF)":
        C_svm = st.slider("C (Regularization)", 0.01, 10.0, 1.0, 0.1)

@st.cache_data
def generate_data(dataset_type, n_samples, noise, random_state=42):
    if dataset_type == "Moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif dataset_type == "Circles":
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)
    elif dataset_type == "Blobs":
        X, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=noise*5+0.5, random_state=random_state)
        y = (y > 0).astype(int)
    else:
        X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0,
                                   n_informative=2, n_clusters_per_class=1,
                                   flip_y=noise, random_state=random_state)
    return X, y

X, y = generate_data(dataset_type, n_samples, noise)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

def get_model():
    if model_name == "Logistic Regression":
        return LogisticRegression(random_state=42)
    elif model_name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    elif model_name == "SVM (RBF)":
        return SVC(kernel='rbf', C=C_svm, probability=True, random_state=42)
    elif model_name == "Random Forest":
        return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth_rf, random_state=42)
    elif model_name == "Gradient Boosting":
        return GradientBoostingClassifier(n_estimators=n_estimators_gb, learning_rate=lr_gb, random_state=42)

model = get_model()
model.fit(X_train_sc, y_train)

train_acc = accuracy_score(y_train, model.predict(X_train_sc))
test_acc = accuracy_score(y_test, model.predict(X_test_sc))
gap = train_acc - test_acc

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f'<div class="metric-box"><div class="label">Train Accuracy</div><div class="value">{train_acc:.1%}</div></div>', unsafe_allow_html=True)
with m2:
    wc = "warn" if test_acc < 0.75 else ""
    st.markdown(f'<div class="metric-box"><div class="label">Test Accuracy</div><div class="value {wc}">{test_acc:.1%}</div></div>', unsafe_allow_html=True)
with m3:
    gc = "warn" if gap > 0.1 else ""
    st.markdown(f'<div class="metric-box"><div class="label">Overfit Gap</div><div class="value {gc}">{gap:.1%}</div></div>', unsafe_allow_html=True)
with m4:
    diagnosis = "OVERFIT 🔴" if gap > 0.15 else ("GOOD FIT 🟢" if gap < 0.05 else "SLIGHT OVERFIT 🟡")
    st.markdown(f'<div class="metric-box"><div class="label">Diagnosis</div><div class="value" style="font-size:1rem;padding-top:4px">{diagnosis}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns([3, 2], gap="large")

with col1:
    st.markdown('<div class="section-label">Decision Boundary</div>', unsafe_allow_html=True)

    h = 0.05
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = scaler.transform(np.c_[xx.ravel(), yy.ravel()])

    if hasattr(model, "predict_proba"):
        Z = model.predict_proba(grid)[:, 1]
    else:
        Z = model.predict(grid).astype(float)
    Z = Z.reshape(xx.shape)

    fig_boundary = go.Figure()
    # Use named colorscale — most reliable across plotly versions
    fig_boundary.add_trace(go.Contour(
        x=np.arange(x_min, x_max, h),
        y=np.arange(y_min, y_max, h),
        z=Z,
        colorscale='RdBu',
        showscale=False,
        opacity=0.3,
        contours=dict(showlines=False),
        hoverinfo='skip'
    ))

    colors = ['#ff6b6b', '#6c63ff']
    for cls in [0, 1]:
        mask_tr = y_train == cls
        mask_te = y_test == cls
        fig_boundary.add_trace(go.Scatter(
            x=X_train[mask_tr, 0], y=X_train[mask_tr, 1],
            mode='markers', name=f'Train {cls}',
            marker=dict(color=colors[cls], size=6, opacity=0.8)
        ))
        fig_boundary.add_trace(go.Scatter(
            x=X_test[mask_te, 0], y=X_test[mask_te, 1],
            mode='markers', name=f'Test {cls}',
            marker=dict(color=colors[cls], size=8, opacity=1,
                        symbol='diamond', line=dict(width=1.5, color='white'))
        ))

    fig_boundary.update_layout(
        paper_bgcolor='#0a0a0f', plot_bgcolor='#111118',
        font=dict(family='JetBrains Mono', color='#e8e8f0'),
        xaxis=dict(gridcolor='#2a2a3a', showgrid=True, zeroline=False),
        yaxis=dict(gridcolor='#2a2a3a', showgrid=True, zeroline=False),
        legend=dict(bgcolor='#111118', bordercolor='#2a2a3a', borderwidth=1, font=dict(size=10)),
        margin=dict(l=20, r=20, t=20, b=20),
        height=420
    )
    st.plotly_chart(fig_boundary, use_container_width=True)

with col2:
    st.markdown('<div class="section-label">Learning Curve</div>', unsafe_allow_html=True)

    train_sizes, train_scores, test_scores = learning_curve(
        get_model(), X_train_sc, y_train,
        cv=5, train_sizes=np.linspace(0.1, 1.0, 8),
        scoring='accuracy', n_jobs=-1
    )
    train_mean = train_scores.mean(axis=1)
    test_mean = test_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_std = test_scores.std(axis=1)

    fig_lc = go.Figure()
    fig_lc.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
        fill='toself', fillcolor='rgba(255,107,107,0.1)',
        line=dict(color='rgba(0,0,0,0)'), showlegend=False, hoverinfo='skip'
    ))
    fig_lc.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([test_mean + test_std, (test_mean - test_std)[::-1]]),
        fill='toself', fillcolor='rgba(108,99,255,0.1)',
        line=dict(color='rgba(0,0,0,0)'), showlegend=False, hoverinfo='skip'
    ))
    fig_lc.add_trace(go.Scatter(x=train_sizes, y=train_mean, mode='lines+markers',
        name='Train', line=dict(color='#ff6b6b', width=2), marker=dict(size=5)))
    fig_lc.add_trace(go.Scatter(x=train_sizes, y=test_mean, mode='lines+markers',
        name='CV Test', line=dict(color='#6c63ff', width=2), marker=dict(size=5)))

    fig_lc.update_layout(
        paper_bgcolor='#0a0a0f', plot_bgcolor='#111118',
        font=dict(family='JetBrains Mono', color='#e8e8f0', size=10),
        xaxis=dict(title='Training Samples', gridcolor='#2a2a3a', zeroline=False),
        yaxis=dict(title='Accuracy', gridcolor='#2a2a3a', zeroline=False, range=[0, 1.05]),
        legend=dict(bgcolor='#111118', bordercolor='#2a2a3a', borderwidth=1, font=dict(size=10)),
        margin=dict(l=20, r=20, t=20, b=20), height=200
    )
    st.plotly_chart(fig_lc, use_container_width=True)

    st.markdown('<div class="section-label" style="margin-top:10px">Bias / Variance Analysis</div>', unsafe_allow_html=True)

    final_train = train_mean[-1]
    final_test = test_mean[-1]
    bv_gap = final_train - final_test

    if final_train < 0.75:
        bv_label, bv_color = "HIGH BIAS", "#ff6b6b"
        bv_desc = "Model is too simple — underfitting. Try a more complex model or add features."
    elif bv_gap > 0.15:
        bv_label, bv_color = "HIGH VARIANCE", "#ff6b6b"
        bv_desc = "Model memorizes training data. Add regularization or more samples."
    elif bv_gap > 0.07:
        bv_label, bv_color = "MODERATE VARIANCE", "#f0c040"
        bv_desc = "Slight overfitting. Consider regularization or pruning."
    else:
        bv_label, bv_color = "WELL BALANCED", "#00d4aa"
        bv_desc = "Good bias-variance tradeoff. Model generalizes well."

    st.markdown(f"""
    <div style="background:#111118;border:1px solid #2a2a3a;border-left:3px solid {bv_color};border-radius:8px;padding:16px;margin-top:8px">
        <div style="font-family:'JetBrains Mono';font-size:0.8rem;font-weight:700;color:{bv_color};margin-bottom:6px">{bv_label}</div>
        <div style="color:#7a7a9a;font-size:0.82rem;line-height:1.6">{bv_desc}</div>
        <div style="margin-top:10px;font-family:'JetBrains Mono';font-size:0.7rem;color:#7a7a9a">
            Train: <span style="color:#ff6b6b">{final_train:.1%}</span> &nbsp;|&nbsp;
            CV: <span style="color:#6c63ff">{final_test:.1%}</span> &nbsp;|&nbsp;
            Gap: <span style="color:{bv_color}">{bv_gap:.1%}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)