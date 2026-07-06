import streamlit as st

from streamlit_ui import api_client
from streamlit_ui.auth import require_authentication
from streamlit_ui.styles import page_header


require_authentication()
page_header("My Memories", "Long-term facts the chatbot remembers about you.", "Account")

user_id = st.session_state["user_id"]
token = st.session_state["access_token"]

try:
    memories = api_client.list_memories(user_id, token)
except api_client.ApiError:
    st.error("Could not load memories.")
    st.stop()
except api_client.requests.RequestException:
    st.error("Could not reach the API. Make sure FastAPI is running.")
    st.stop()

st.subheader("Add New Memory")
with st.form("add_memory_form", clear_on_submit=True):
    new_memory_text = st.text_input("Memory text", placeholder="e.g. My favorite color is blue.")
    submitted = st.form_submit_button("Add Memory", type="primary")
    if submitted and new_memory_text.strip():
        try:
            api_client.add_memory(user_id, token, new_memory_text.strip())
            st.toast("Memory added successfully.")
            st.rerun()
        except api_client.ApiError as exc:
            st.error(f"Could not add memory: {exc}")

st.divider()

col1, col2 = st.columns([3, 1])
with col1:
    st.subheader(f"Memories ({len(memories)})")
with col2:
    if memories and st.button("Delete All", type="primary", use_container_width=True):
        try:
            api_client.delete_all_memories(user_id, token)
            st.toast("All memories deleted.")
            st.rerun()
        except api_client.ApiError as exc:
            st.error(f"Could not delete memories: {exc}")

if not memories:
    st.info("No memories stored yet. Chat with the assistant to build memories.")
    st.stop()

for mem in memories:
    cols = st.columns([5, 1])
    with cols[0]:
        st.markdown(f"- {mem['memory_text']}")
        if mem.get("created_at"):
            st.caption(f"Stored: {mem['created_at'][:10]}")
    with cols[1]:
        if st.button("Delete", key=f"del-{mem['id']}", use_container_width=True):
            try:
                api_client.delete_memory(user_id, token, mem["id"])
                st.toast("Memory deleted.")
                st.rerun()
            except api_client.ApiError as exc:
                st.error(f"Could not delete: {exc}")
    st.divider()
