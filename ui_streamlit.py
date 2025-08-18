# ui_streamlit.py
import streamlit as st
import requests

SERVER_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Ticket Intake", page_icon="✅", layout="centered")
st.title("Submit a Support Ticket")

with st.form("ticket_form", clear_on_submit=False):
    subject = st.text_input("Subject", "")
    body = st.text_area("Body", "", height=200)
    submitted = st.form_submit_button("Submit")

if submitted:
    if not subject.strip() and not body.strip():
        st.error("Please enter a subject or body.")
    else:
        try:
            resp = requests.post(f"{SERVER_BASE}/tool/submit_ticket", json={"subject": subject, "body": body}, timeout=60)
            data = resp.json()
            if data.get("ok"):
                st.success("We have received your ticket.")
                if data.get("task_url"):
                    st.markdown(f"[View in ClickUp]({data['task_url']})")
            else:
                reason = data.get("reason", "unknown_error")
                if reason == "duplicate":
                    st.warning("We already received this ticket.")
                else:
                    st.error(f"Error: {reason}")
        except Exception as e:
            st.error(f"Server error: {e}")
