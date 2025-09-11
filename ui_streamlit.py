# ui_streamlit.py
import json
import streamlit as st
from agent_runner import run_agent

st.set_page_config(page_title="Agentic Ticket Intake", page_icon="🎫", layout="centered")
st.title("🎫 Agentic Ticket Intake")
st.caption("Clean → Predict → ClickUp (custom ReAct via local Ollama)")

with st.form("ticket_form", clear_on_submit=False):
    subj = st.text_input("Subject", value="SSO login fails intermittently")
    body = st.text_area("Body", height=200, value="Users redirected back to login after update.\n\nThanks,\nBob")
    submitted = st.form_submit_button("Submit")

if submitted:
    with st.spinner("Running agent..."):
        res = run_agent(subj, body)

    st.subheader("Result")
    st.code(json.dumps(res, indent=2, ensure_ascii=False), language="json")

    if res.get("ok") and res.get("task_url"):
        st.success("Task created!")
        st.markdown(f"[Open task in ClickUp]({res['task_url']})")
