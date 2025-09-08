# debug_clickup_fields.py
from tickets.clickup_client import list_fields
from pprint import pprint

if __name__ == "__main__":
    data = list_fields()
    fields = data.get("fields", [])
    print(f"\nFound {len(fields)} custom fields attached to this List:\n")
    for f in fields:
        fid = f.get("id")
        name = f.get("name")
        ftype = f.get("type")
        print(f"- {name!r}  id={fid}  type={ftype}")
        if ftype == "drop_down":
            opts = (f.get("type_config", {}) or {}).get("options", []) or []
            print(f"  options ({len(opts)}): " + ", ".join([o.get("name","") for o in opts]))
    print("\nRaw:\n")
    pprint(data)
