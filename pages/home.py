import streamlit as st

from streamlit_ui.styles import feature_card


st.markdown('<div class="dc-eyebrow">Professional document intelligence</div>', unsafe_allow_html=True)
st.title("Chat with your documents, intelligently")
st.markdown(
    '<div class="dc-subtitle">Upload your knowledge, ask natural-language questions, '
    "and get grounded answers with source citations.</div>",
    unsafe_allow_html=True,
)

primary, secondary, spacer = st.columns([1, 1, 3])
with primary:
    if st.button("Get started", type="primary", use_container_width=True):
        st.switch_page("pages/signup.py")
with secondary:
    if st.button("Sign in", use_container_width=True):
        st.switch_page("pages/login.py")

st.divider()
st.subheader("Streamlined document intelligence")
st.caption("Transform static files into an interactive knowledge base in three steps.")

upload, ask, answer = st.columns(3)
with upload:
    feature_card("Upload documents", "Add PDFs, text, markdown, HTML, and source-code files securely.")
with ask:
    feature_card("Ask questions", "Query your documents naturally, without special commands or syntax.")
with answer:
    feature_card("Get answers", "Receive context-aware responses grounded in your uploaded sources.")
