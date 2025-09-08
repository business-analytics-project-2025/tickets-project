from agent_runner import run_agent

res = run_agent(
    "SSO login fails intermitently",
    "Users redirected back to login after update.\n--\nBest,\nAlice"
)
print(res)
