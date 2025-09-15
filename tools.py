# tools.py
import json
import time
from langchain.agents import tool
from tickets.text_clean import clean_subject_body
from tickets.orchestrator import predict_all as pipeline_predict_all
from tickets.clickup_client import (
    create_task, add_tags, set_dropdown_value,
    append_tags_note, append_field_note
)
from tickets.config import PRIORITY_TO_CLICKUP, TYPE_FIELD_ID, DEPT_FIELD_ID

@tool
def clean_text(json_input_str: str) -> str:
    """Cleans the subject and body of a ticket."""
    try:
        data = json.loads(json_input_str)
        s, b = clean_subject_body(data.get('subject', ''), data.get('body', ''))
        return json.dumps({"cleaned_subject": s, "cleaned_body": b})
    except Exception as e:
        return json.dumps({"ok": False, "reason": f"Error in clean_text: {e}"})

@tool
def predict_ticket_attributes(json_input_str: str) -> str:
    """Runs ML models to predict attributes for a ticket."""
    try:
        data = json.loads(json_input_str)
        predictions = pipeline_predict_all(data.get('subject', ''), data.get('body', ''))
        if 'confidences' in predictions:
            del predictions['confidences']
        return json.dumps(predictions)
    except Exception as e:
        return json.dumps({"ok": False, "reason": f"Error in predict_ticket_attributes: {e}"})

@tool
def create_clickup_task(json_input_str: str) -> str:
    """Creates a task in ClickUp."""
    try:
        data = json.loads(json_input_str)
        subject = data.get('subject', '(no subject)')
        body = data.get('body', '')
        department = data.get('department', '')
        ticket_type = data.get('type', '')
        priority = data.get('priority', 'Medium')
        tags = data.get('tags', [])

        priority_num = PRIORITY_TO_CLICKUP.get(priority, 3)
        task = create_task(name=subject, description=body, priority_num=priority_num)
        
        task_id = task.get("id")
        if not task_id:
            return json.dumps({"ok": False, "reason": "ClickUp API did not return a task ID", "response": task})

        # Add a short delay to prevent API race conditions
        time.sleep(1) 

        if ticket_type:
            _, exact, chosen, _ = set_dropdown_value(task_id, TYPE_FIELD_ID, ticket_type)
            if not exact:
                append_field_note(task_id, "Type", ticket_type, chosen)
        if department:
            _, exact, chosen, _ = set_dropdown_value(task_id, DEPT_FIELD_ID, department)
            if not exact:
                append_field_note(task_id, "Department", department, chosen)
        if tags:
            failed = add_tags(task_id, tags)
            if failed:
                append_tags_note(task_id, failed)

        return json.dumps({"ok": True, "task_id": task_id, "task_url": task.get("url", "")})
    except Exception as e:
        return json.dumps({"ok": False, "reason": f"clickup_failed: {type(e).__name__} - {e}"})