"""Streamlit chat interface for trainees. Talks to the FastAPI service."""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Course Assistant | مساعد المقرر", page_icon="🎓")
st.title("🎓 Course Assistant | مساعد المقرر")
st.caption("Ask about the course material in Arabic or English — اسأل عن محتوى المقرر بالعربية أو الإنجليزية")

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(f'<div dir="auto">{turn["content"]}</div>', unsafe_allow_html=True)

question = st.chat_input("What is overfitting? / ما هو فرط التخصيص؟")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(f'<div dir="auto">{question}</div>', unsafe_allow_html=True)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=120)
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as exc:
                st.error(f"Could not reach the API: {exc}")
                st.stop()

        st.markdown(f'<div dir="auto">{data["answer"]}</div>', unsafe_allow_html=True)
        if data["sources"]:
            with st.expander("Sources | المصادر"):
                for src in data["sources"]:
                    page = f", page {src['page']}" if src.get("page") else ""
                    st.markdown(f"**[{src['id']}] {src['source']}{page}**")
                    st.markdown(f'<div dir="auto" style="color:gray">{src["snippet"]}…</div>', unsafe_allow_html=True)

    st.session_state.history.append({"role": "assistant", "content": data["answer"]})
