import streamlit as st

from streamlit_ui.api_client import API_BASE_URL
from streamlit_ui.auth import require_authentication, sign_out
from streamlit_ui.styles import page_header


require_authentication()
page_header("Settings", "Review your current session and application connection.", "Account")

st.subheader("Session")
st.write(f"Signed in as user **{st.session_state.user_id}**")

st.subheader("API connection")
st.code(API_BASE_URL, language=None)
st.caption("Set DOCCHAT_API_URL before launching Streamlit to use a different FastAPI service.")

st.divider()
if st.button("Log out", type="primary"):
    sign_out()
    st.rerun()
