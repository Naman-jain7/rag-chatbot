import uuid

import streamlit as st

from streamlit_ui import api_client
from streamlit_ui.auth import require_authentication
from streamlit_ui.styles import page_header


require_authentication()

META_PREFIX = "\n__META__"

# ── Initialise session state ──────────────────────────────────────────────────
if "conversations" not in st.session_state:
    st.session_state["conversations"] = []
if "chat_id" not in st.session_state:
    st.session_state["chat_id"] = str(uuid.uuid4())
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []
if "token_usage" not in st.session_state:
    st.session_state["token_usage"] = {}

def _new_chat():
    st.session_state["chat_messages"] = []
    st.session_state["token_usage"] = {}
    st.session_state["chat_id"] = str(uuid.uuid4())

def _load_conversation(chat_id: str):
    try:
        history = api_client.get_chat_history(
            token=st.session_state["access_token"],
            user_id=st.session_state["user_id"],
            chat_id=chat_id,
        )

        st.session_state["chat_messages"] = history.get("messages", [])
        st.session_state["token_usage"] = history.get("token_usage", {})
        st.session_state["chat_id"] = chat_id
    except Exception as e:
        st.error(f"Failed to load chat history: {e}")


with st.sidebar:
    st.markdown("### Chat History")
    if st.button("+ New Chat", use_container_width=True, type="primary"):
        _new_chat()
        st.rerun()

    st.divider()

    # (Optional) Request active threads list for this user from backend on startup
    conv_list = st.session_state.conversations
    if not conv_list:
        st.caption("No conversations yet.")

    for conv in reversed(conv_list):
        label = conv["title"]
        if st.button(label, key=f"conv-{conv['id']}", use_container_width=True):
            _load_conversation(conv["id"])
            st.rerun()

# ── Main chat area ─────────────────────────────────────────────────────────────
header, action = st.columns([5, 1])
with header:
    page_header("Chat", "Ask questions grounded in your uploaded documents.", "Workspace")
with action:
    st.write("")
    st.write("")
    if st.button("New chat", use_container_width=True):
        _new_chat()
        st.rerun()

st.caption(f"Chat ID: {st.session_state.chat_id}")

for message in st.session_state["chat_messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Token usage display ────────────────────────────────────────────────────────
usage = st.session_state["token_usage"]
if usage:
    st.caption(
        f"Token usage — Prompt: {usage.get('input_tokens', 0)} | "
        f"Output: {usage.get('output_tokens', 0)} | "
        f"Total: {usage.get('total_tokens', 0)}"
    )

# ── Chat input ─────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your documents..."):
    # Append user message locally immediately for UI fluid responsiveness
    st.session_state["chat_messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        try:
            # Stream directly from FastAPI backend which holds the Postgres checkpointer
            for chunk in api_client.stream_chat(
                st.session_state.user_id,
                st.session_state.access_token,
                prompt,
                chat_id=st.session_state['chat_id'],
            ):
                if chunk.startswith(META_PREFIX):
                    import json
                    meta = json.loads(chunk[len(META_PREFIX):])
                    st.session_state['token_usage'] = meta
                    continue
                full_response += chunk
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
            
            # Save assistant response locally to keep UI state accurate
            st.session_state['chat_messages'].append({"role": "assistant", "content": full_response})
            
            # (Optional) Track metadata sidebar updates dynamically
            # If it's a brand new chat, add it to sidebar track list
            if not any(c["id"] == st.session_state['chat_id'] for c in st.session_state['conversations']):
                st.session_state['conversations'].append({
                    "id": st.session_state.chat_id,
                    "title": prompt[:30] + "..."
                })

        except api_client.ApiError as exc:
            placeholder.error(f"Sorry, the request failed: {exc}")
        except api_client.requests.RequestException:
            placeholder.error("Could not reach the API. Make sure FastAPI is running.")