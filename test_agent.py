import json
from agent_runner import run_agent

print("Running agent...")
res = run_agent(
    "SSO login fails intermitently",
    "Users are redirected back to the login page after we updated the server.\n--\nBest,\nAlice"
)
print("\n--- Final Result ---")
print(json.dumps(res, indent=2))