import streamlit as st


def init_session() -> None:
    defaults = {
        "access_token": None,
        "user_id": None,
        "chat_messages": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def is_authenticated() -> bool:
    return bool(st.session_state.get("access_token") and st.session_state.get("user_id"))


def sign_in(access_token: str, user_id: int) -> None:
    st.session_state.access_token = access_token
    st.session_state.user_id = user_id
    st.session_state.chat_messages = []


def sign_out() -> None:
    st.session_state.access_token = None
    st.session_state.user_id = None
    st.session_state.chat_messages = []


def require_authentication() -> None:
    if not is_authenticated():
        st.warning("Please sign in to continue.")
        st.stop()
