import json
from datetime import datetime
from typing import Any

import streamlit as st

from streamlit_ui import api_client
from streamlit_ui.auth import require_authentication
from streamlit_ui.styles import page_header


DOCUMENT_TYPES = ["pdf", "txt", "md", "html", "py", "js", "ts", "java", "cpp", "go", "rb", "php", "rs"]
AUDIO_TYPES = ["mp3", "wav", "m4a", "ogg", "flac"]


def _record_error(filename: str, exc: Exception) -> None:
    if "upload_errors" not in st.session_state:
        st.session_state.upload_errors = []
    st.session_state["upload_errors"].append(f"{filename}: {exc}")


def _parse_metadata(document: dict[str, Any]) -> dict[str, Any]:
    metadata = document.get("extra_metadata") or {}
    if isinstance(metadata, str):
        try:
            return json.loads(metadata)
        except json.JSONDecodeError:
            return {}
    if isinstance(metadata, dict):
        return metadata
    return {}


def _format_duration(seconds: Any) -> str | None:
    if seconds is None:
        return None
    try:
        total_seconds = int(round(float(seconds)))
    except (TypeError, ValueError):
        return None

    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _format_created_at(value: Any) -> str:
    try:
        return datetime.fromisoformat(str(value)).strftime("%b %d, %Y")
    except (TypeError, ValueError):
        return "Uploaded"


require_authentication()
page_header("My Documents", "Manage your uploaded knowledge base.", "Workspace")

token = st.session_state["access_token"]
user_id = st.session_state["user_id"]

upload_tab, audio_tab = st.tabs(["Documents", "Audio"])

with upload_tab:
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=DOCUMENT_TYPES,
        accept_multiple_files=True,
        key="document_uploader",
    )
    if st.button("Upload selected documents", type="primary", disabled=not uploaded_files):
        progress = st.progress(0, text="Uploading documents...")
        for index, uploaded_file in enumerate(uploaded_files):
            try:
                api_client.upload_document(user_id, token, uploaded_file)
                st.toast(f"Uploaded {uploaded_file.name}")
            except api_client.ApiError as exc:
                _record_error(uploaded_file.name, exc)
            progress.progress((index + 1) / len(uploaded_files), text="Uploading documents...")
        progress.empty()
        st.rerun()

with audio_tab:
    audio_files = st.file_uploader(
        "Upload audio",
        type=AUDIO_TYPES,
        accept_multiple_files=True,
        key="audio_uploader",
    )
    if st.button("Upload selected audio", type="primary", disabled=not audio_files):
        results = []
        progress = st.progress(0, text="Transcribing audio...")
        for index, uploaded_file in enumerate(audio_files):
            try:
                with st.spinner(f"Transcribing {uploaded_file.name}..."):
                    result = api_client.upload_audio(user_id, token, uploaded_file)
                result["filename"] = uploaded_file.name
                results.append(result)
                st.toast(f"Uploaded {uploaded_file.name}")
            except api_client.ApiError as exc:
                _record_error(uploaded_file.name, exc)
            progress.progress((index + 1) / len(audio_files), text="Transcribing audio...")
        progress.empty()
        st.session_state["audio_upload_results"] = results
        st.rerun()

if "upload_errors" in st.session_state and st.session_state["upload_errors"]:
    for err in st.session_state.upload_errors:
        st.error(err)
    st.session_state.upload_errors = []

if st.session_state.get("audio_upload_results"):
    for result in st.session_state["audio_upload_results"]:
        duration = _format_duration(result.get("duration"))
        details = f"{result.get('chunks', 0)} chunks"
        if duration:
            details = f"{duration} audio, {details}"
        st.success(f"{result.get('filename', 'Audio')} indexed: {details}")

filter_column, type_column = st.columns([3, 1])
filter_text = filter_column.text_input(
    "Filter files",
    placeholder="Search by filename",
    label_visibility="collapsed",
)
type_label = type_column.selectbox("Type", ["All", "Documents", "Audio"], label_visibility="collapsed")
source_type = {"Documents": "document", "Audio": "audio"}.get(type_label)

try:
    documents = api_client.list_documents(user_id, token, source_type=source_type)
except api_client.ApiError as exc:
    st.error(f"Could not load files: {exc}")
    documents = []
except api_client.requests.RequestException:
    st.error("Could not reach the API. Make sure FastAPI is running.")
    documents = []

filtered_documents = [
    document for document in documents if filter_text.lower() in document["filename"].lower()
]

st.subheader(f"Files ({len(filtered_documents)})")
if not filtered_documents:
    st.info("No matching files yet.")

for document in filtered_documents:
    filename = document["filename"]
    source_label = "Audio" if document.get("source_type") == "audio" else "Document"
    created_label = _format_created_at(document.get("created_at"))
    metadata = _parse_metadata(document)
    duration = _format_duration(metadata.get("duration"))

    name_column, type_column, date_column, action_column = st.columns([5, 1.25, 1.75, 1])
    name_column.markdown(f"**{filename}**")
    if duration:
        name_column.caption(f"Duration {duration}")
    type_column.caption(source_label)
    date_column.caption(created_label)
    if action_column.button("Delete", key=f"delete-{document['doc_id']}", use_container_width=True):
        try:
            api_client.delete_document(user_id, token, filename)
            st.toast(f"Deleted {filename}")
            st.rerun()
        except api_client.ApiError as exc:
            st.error(f"Could not delete {filename}: {exc}")
    st.divider()
