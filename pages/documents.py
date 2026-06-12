from datetime import datetime

import streamlit as st

from streamlit_ui import api_client
from streamlit_ui.auth import require_authentication
from streamlit_ui.styles import page_header


require_authentication()
page_header("My Documents", "Manage your uploaded knowledge base.", "Workspace")

token = st.session_state["access_token"]
user_id = st.session_state["user_id"]

with st.expander("Upload documents", expanded=True):
    uploaded_files = st.file_uploader(
        "Drag and drop files here, or click to browse",
        type=["pdf", "txt", "md", "html", "py", "js", "ts", "java", "cpp", "go", "rb", "php", "rs"],
        accept_multiple_files=True,
    )
    if st.button("Upload selected files", type="primary", disabled=not uploaded_files):
        progress = st.progress(0, text="Uploading documents...")
        for index, uploaded_file in enumerate(uploaded_files):
            try:
                api_client.upload_document(user_id, token, uploaded_file)
                st.toast(f"Uploaded {uploaded_file.name}")
            except api_client.ApiError as exc:
                if "upload_errors" not in st.session_state:
                    st.session_state.upload_errors = []
                st.session_state["upload_errors"].append(f"{uploaded_file.name}: {exc}")
            progress.progress((index + 1) / len(uploaded_files), text="Uploading documents...")
        progress.empty()
        st.rerun()

if "upload_errors" in st.session_state and st.session_state["upload_errors"]:
    for err in st.session_state.upload_errors:
        st.error(err)
    st.session_state.upload_errors = []

filter_text = st.text_input("Filter documents", placeholder="Search by filename", label_visibility="collapsed")

try:
    documents = api_client.list_documents(user_id, token)
except api_client.ApiError as exc:
    st.error(f"Could not load documents: {exc}")
    documents = []
except api_client.requests.RequestException:
    st.error("Could not reach the API. Make sure FastAPI is running.")
    documents = []

filtered_documents = [
    document for document in documents if filter_text.lower() in document["filename"].lower()
]

st.subheader(f"Documents ({len(filtered_documents)})")
if not filtered_documents:
    st.info("No matching documents yet.")

for document in filtered_documents:
    filename = document["filename"]
    created_at = document.get("created_at", "")
    try:
        created_label = datetime.fromisoformat(created_at).strftime("%b %d, %Y")
    except (TypeError, ValueError):
        created_label = "Uploaded"

    name_column, date_column, action_column = st.columns([5, 2, 1])
    name_column.markdown(f"**{filename}**")
    date_column.caption(created_label)
    if action_column.button("Delete", key=f"delete-{document['doc_id']}", use_container_width=True):
        try:
            api_client.delete_document(user_id, token, filename)
            st.toast(f"Deleted {filename}")
            st.rerun()
        except api_client.ApiError as exc:
            st.error(f"Could not delete {filename}: {exc}")
    st.divider()
