import streamlit as st


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --surface: #ffffff;
            --surface-dim: #f7f8fa;
            --surface-container: #f0f2f5;
            --primary: #2d3748;
            --text: #1a202c;
            --muted: #4a5568;
            --outline: #e2e8f0;
        }
        .stApp {
            background: var(--surface);
            color: var(--text);
        }
        [data-testid="stSidebar"] {
            background: var(--surface-dim);
            border-right: 1px solid var(--outline);
        }
        [data-testid="stSidebarNav"]::before {
            content: "DocChat";
            display: block;
            color: var(--primary);
            font-size: 1.25rem;
            font-weight: 700;
            padding: 0.5rem 1rem 1rem;
        }
        .block-container {
            max-width: 1180px;
            padding-top: 2.5rem;
        }
        h1, h2, h3 {
            color: var(--primary);
            letter-spacing: -0.025em;
        }
        .dc-eyebrow {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }
        .dc-subtitle {
            color: var(--muted);
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }
        .dc-card {
            background: var(--surface);
            border: 1px solid var(--outline);
            border-radius: 0.5rem;
            min-height: 155px;
            padding: 1.5rem;
        }
        .dc-card p {
            color: var(--muted);
        }
        .stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
            background: var(--primary);
            border-color: var(--primary);
        }
        [data-testid="stFileUploaderDropzone"] {
            background: var(--surface-dim);
            border-color: #cbd5e0;
        }
        [data-testid="stChatMessage"] {
            border: 1px solid var(--outline);
            border-radius: 0.5rem;
            background: var(--surface);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str, eyebrow: str = "DocChat") -> None:
    st.markdown(f'<div class="dc-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="dc-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def feature_card(title: str, body: str) -> None:
    st.markdown(
        f'<div class="dc-card"><h3>{title}</h3><p>{body}</p></div>',
        unsafe_allow_html=True,
    )
