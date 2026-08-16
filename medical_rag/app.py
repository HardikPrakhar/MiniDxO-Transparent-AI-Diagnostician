import os
import uuid

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("MiniDxO — AI Medical Assistant")
st.caption("This is an AI assistant. Do not use for real medical decisions.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Describe your symptoms..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consulting medical sources..."):
            try:
                res = requests.post(
                    f"{API_URL}/chat",
                    json={"message": prompt},
                    headers={"X-Session-Id": st.session_state.session_id},
                    timeout=60,
                )
                res.raise_for_status()
                reply = res.json()["response"]
            except requests.RequestException as e:
                reply = f"Sorry, I couldn't reach the backend ({e})."
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

if st.sidebar.button("Clear conversation"):
    requests.post(
        f"{API_URL}/reset",
        headers={"X-Session-Id": st.session_state.session_id},
    )
    st.session_state.messages = []
    st.rerun()
