import streamlit as st

from streamlit_ui.auth import init_session, is_authenticated
from streamlit_ui.styles import apply_styles

st.set_page_config(
page_title="DocChat",
    page_icon="DC",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session()
apply_styles()

if is_authenticated():
    pages = {
        "Workspace": [
            st.Page("pages/documents.py", title="Documents", icon=":material/description:", default=True),
            st.Page("pages/chat.py", title="Chat", icon=":material/forum:"),
        ],
        "Account": [
            st.Page("pages/settings.py", title="Settings", icon=":material/settings:"),
        ],
    }
else:
    pages = {
        "DocChat": [
            st.Page("pages/home.py", title="Home", icon=":material/home:", default=True),
            st.Page("pages/login.py", title="Login", icon=":material/login:"),
            st.Page("pages/signup.py", title="Sign Up", icon=":material/person_add:"),
        ]
    }

st.navigation(pages).run()
