import streamlit as st


def section_title(title: str, subtitle: str | None = None):
    st.markdown(f"## {title}")

    if subtitle:
        st.markdown(
            f"""
            <p style="color: #CBD5E1; margin-top: -0.5rem;">
                {subtitle}
            </p>
            """,
            unsafe_allow_html=True,
        )


def card(content: str):
    st.markdown(
        f"""
        <div style="
            background-color: #1E293B;
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 1.25rem;
            margin-bottom: 1rem;
        ">
            {content}
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str = ""):
    note_html = ""

    if note:
        note_html = f"""
        <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 0.35rem;">
            {note}
        </div>
        """

    st.markdown(
        f"""
        <div style="
            background-color: #1E293B;
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 1.1rem 1.2rem;
            min-height: 118px;
        ">
            <div style="color: #94A3B8; font-size: 0.9rem; margin-bottom: 0.4rem;">
                {label}
            </div>
            <div style="font-size: 2rem; font-weight: 700; color: #F8FAFC;">
                {value}
            </div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def confidence_card(confidence: str, inside_domain: bool):
    if inside_domain:
        color = "#22C55E"
        bg = "rgba(34, 197, 94, 0.12)"
        border = "rgba(34, 197, 94, 0.35)"
        message = "Inputs are inside the recommended training domain."
    elif confidence == "Moderate":
        color = "#F59E0B"
        bg = "rgba(245, 158, 11, 0.12)"
        border = "rgba(245, 158, 11, 0.35)"
        message = "Some inputs are outside the recommended training domain."
    else:
        color = "#EF4444"
        bg = "rgba(239, 68, 68, 0.12)"
        border = "rgba(239, 68, 68, 0.35)"
        message = "Multiple inputs are outside the recommended training domain."

    st.markdown(
        f"""
        <div style="
            background-color: {bg};
            border: 1px solid {border};
            border-radius: 14px;
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
        ">
            <div style="font-size: 0.85rem; color: #CBD5E1; margin-bottom: 0.25rem;">
                Model Confidence
            </div>
            <div style="font-size: 1.4rem; font-weight: 700; color: {color};">
                {confidence}
            </div>
            <div style="color: #CBD5E1; margin-top: 0.35rem;">
                {message}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )