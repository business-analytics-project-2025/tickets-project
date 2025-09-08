# test_fields.py
from tickets.clickup_client import create_task, set_dropdown_value, get_task
from tickets.service_submit import TYPE_FIELD_ID, DEPT_FIELD_ID
import time, json

t = create_task("Field ID sanity check", "desc", 3)
tid = t["id"]
print("Task:", tid, t.get("url"))

resp, exact, chosen, optid = set_dropdown_value(tid, TYPE_FIELD_ID, "Problem")
print("Type set -> exact:", exact, "chosen:", chosen)
resp, exact, chosen, optid = set_dropdown_value(tid, DEPT_FIELD_ID, "General Inquiry")
print("Dept set -> exact:", exact, "chosen:", chosen)

for i in range(6):
    time.sleep(1.0)
    cf = get_task(tid).get("custom_fields", [])
    print(f"[{i+1}s] cf:", json.dumps(cf, ensure_ascii=False))
