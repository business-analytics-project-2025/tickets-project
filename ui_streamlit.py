# ui_streamlit.py
import json
import streamlit as st
from agent_runner import run_agent

st.set_page_config(page_title="Agentic Ticket Intake", page_icon="🎫", layout="centered")

st.title("🎫 Agentic Ticket Intake")

with st.form("ticket_form", clear_on_submit=False):
    subj = st.text_input("Subject", value="")
    body = st.text_area("Body", height=200, value="")
    submitted = st.form_submit_button("Submit")

if submitted:
    with st.spinner("Running agent..."):
        res = run_agent(subj, body)

    st.subheader("Result")
    st.code(json.dumps(res, indent=2, ensure_ascii=False), language="json")

    if res.get("ok") and res.get("task_url"):
        st.success("Task created!")
        st.markdown(f"[Open task in ClickUp]({res['task_url']})")
