import streamlit as st

from streamlit_ui import api_client
from streamlit_ui.auth import sign_in
from streamlit_ui.styles import page_header


page_header("Welcome back", "Sign in to your DocChat account.", "Account")

left, form_column, right = st.columns([1, 1.4, 1])
with form_column:
    with st.form("login_form"):
        email = st.text_input("Email address", placeholder="you@example.com")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)

    if submitted:
        try:
            result = api_client.login(email.strip(), password)
            sign_in(result["access_token"], result["user_id"])
            st.session_state["user_id"] = result["user_id"]
            st.session_state["user_email"] = email
            st.success("Login successful!")
            st.rerun()
        except api_client.ApiError as exc:
            st.error(f"Login failed: {exc}")
        except api_client.requests.RequestException:
            st.error("Could not reach the API. Make sure FastAPI is running.")

    st.divider()
    
    st.markdown("Don't have an account?")
    if st.button("Create an Account", use_container_width=True):
        st.switch_page("pages/signup.py")
