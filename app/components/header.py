import streamlit as st

def render_header() -> None:
    st.markdown("<div style='display:flex; flex-direction:column; align-items:center;'>", unsafe_allow_html=True)
    st.image("app/assets/logo.png", width=500)
    st.markdown("<div style='color:#AAA; font-size:1.1rem; margin-top:8px;'>Modular Neuroimaging Analysis Platform</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
