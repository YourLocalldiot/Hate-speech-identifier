"""Streamlit application scaffold for the hate-speech classifier.

This module provides a placeholder entry point for a future web interface.
It is intentionally minimal and does not implement inference logic yet.
"""

from __future__ import annotations

import streamlit as st


def main() -> None:
    """Render the Streamlit application UI.

    The interface will be expanded later to accept text input and display
    predictions from the trained model.
    """
    st.title("Hate Speech Identifier")
    st.write("Model interface placeholder")


if __name__ == "__main__":
    main()
