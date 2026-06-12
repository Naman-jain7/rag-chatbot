import streamlit as st

from streamlit_ui import api_client
from streamlit_ui.styles import page_header


page_header("Create your account", "Start chatting with your documents today.", "Account")

left, form_column, right = st.columns([1, 1.5, 1])
with form_column:
    with st.form("signup_form"):
        full_name = st.text_input("Full name", placeholder="John Doe")
        email = st.text_input("Email address", placeholder="you@example.com")
        password = st.text_input("Password", type="password", help="Use at least 8 characters.")
        confirm_password = st.text_input("Confirm password", type="password")
        age = st.number_input("Age (optional)", min_value=0, max_value=150, value=None, step=1)
        submitted = st.form_submit_button("Create account", type="primary", use_container_width=True)

    if submitted:
        if not full_name.strip() or not email.strip():
            st.error("Full name and email are required.")
        elif len(password) < 8:
            st.error("Password must be at least 8 characters.")
        elif password != confirm_password:
            st.error("Passwords do not match.")
        else:
            try:
                api_client.signup(full_name.strip(), email.strip(), password, age)
                st.success("Account created successfully! Please login.")
                st.switch_page("pages/login.py")
            except Exception as e:
                st.error(f"Could not connect to server: {e}")

    st.divider()
    st.markdown("Already have an account?")
    if st.button("Back to Login", use_container_width=True):
        st.switch_page("pages/login.py")
