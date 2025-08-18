# mcp_clickup_server.py
import os
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Business logic service (already built)
from tickets.service_submit import submit_ticket as service_submit
from tickets.clickup_client import list_fields
from tickets.registry import ensure_loaded

# ---- Config from env ----
REQUIRED_ENVS = ["CLICKUP_TOKEN", "CLICKUP_LIST_ID", "CLICKUP_TEAM_ID"]
for k in REQUIRED_ENVS:
    if not os.getenv(k):
        print(f"[WARN] Missing env var: {k}. Set it before using server tools.")

app = FastAPI(title="Tickets MCP Server", version="1.0")

class SubmitReq(BaseModel):
    subject: str
    body: str

@app.get("/tool/health")
def health():
    # check token + models
    tok_ok = bool(os.getenv("CLICKUP_TOKEN"))
    try:
        ensure_loaded()
        models_ok = True
    except Exception as e:
        models_ok = False
        return {"ok": False, "models_ok": False, "token_ok": tok_ok, "reason": str(e)}
    return {"ok": tok_ok and models_ok, "models_ok": models_ok, "token_ok": tok_ok}

@app.post("/tool/submit_ticket")
def submit_ticket(req: SubmitReq):
    result = service_submit(req.subject, req.body)
    return result

@app.get("/tool/list_fields")
def fields():
    try:
        return {"ok": True, "data": list_fields()}
    except Exception as e:
        return {"ok": False, "reason": str(e)}

if __name__ == "__main__":
    # Default to localhost:8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
