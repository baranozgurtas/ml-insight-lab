import streamlit as st

st.set_page_config(
    page_title="ML Insight Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

home = st.Page("pages/home.py", title="Home", icon="🏠", default=True)
playground = st.Page("pages/model_playground.py", title="Model Playground", icon="⚡")
ab_testing = st.Page("pages/ab_testing.py", title="A/B Testing", icon="📊")

pg = st.navigation([home, playground, ab_testing])
pg.run()