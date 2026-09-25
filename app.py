import uuid
import streamlit as st
import requests

st.set_page_config(page_title="Research Assistant", page_icon="🔎")

BACKEND_URL = "http://localhost:8000"

st.title("🔎 Research Assistant")
st.caption("Planning → Tool Use → Reflection, with conversation memory · FastAPI + LangGraph")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.thread_id = str(uuid.uuid4())

with st.sidebar:
    st.header("Settings")

    if st.button("🆕 New chat"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())

    try:
        res = requests.get(f"{BACKEND_URL}/", timeout=2)
        if res.status_code == 200:
            st.success("🟢 Backend Connected")
        else:
            st.error("🔴 Backend Error")
    except requests.exceptions.RequestException:
        st.error("🔴 Backend Offline (Run main.py first!)")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("What do you want to research?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Planning, researching & reflecting..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"query": prompt, "thread_id": st.session_state.thread_id},
                    timeout=120
                )

                if response.status_code == 200:
                    data = response.json()

                    with st.expander("🧭 Plan"):
                        for i, step in enumerate(data["plan"], 1):
                            st.write(f"{i}. {step}")

                    with st.expander("📝 Draft (before reflection)"):
                        st.write(data["draft"])

                    with st.expander("🪞 Reflection report", expanded=True):
                        st.markdown(data["critique"])

                    st.markdown("### ✅ Revised answer")
                    st.write(data["answer"])

                    st.session_state.messages.append({"role": "assistant", "content": data["answer"]})
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown Error')}")

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to Backend. Is it running?")
            except Exception as e:
                st.error(f"An error occurred: {e}")
