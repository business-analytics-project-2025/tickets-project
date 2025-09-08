# print_task_fields.py
from tickets.clickup_client import get_task
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python print_task_fields.py <task_id>")
        sys.exit(1)
    tid = sys.argv[1]
    t = get_task(tid)
    print("Task:", tid, t.get("url"))
    print("Custom fields snapshot:")
    for cf in t.get("custom_fields", []) or []:
        print("-", cf.get("name"), "| id:", cf.get("id"), "| value:", cf.get("value"))
