# debug_agents.py
from tickets.agents import IntakeAgent, PreprocessAgent, TagsAgent, DepartmentAgent, TypeAgent, PriorityAgent

if __name__ == "__main__":
    subject = "SSO login fails intermittently"
    body = "Users get redirected back to login. Started after yesterday's update."

    intake = IntakeAgent()
    pre = PreprocessAgent()
    tix = pre.handle(intake.handle(subject, body))

    for agent in [TagsAgent(), DepartmentAgent(), TypeAgent(), PriorityAgent()]:
        name = agent.task
        try:
            out = agent.handle(tix)
            print(f"{name}: OK  -> preds={out.preds[:3]} ...")
        except Exception as e:
            print(f"{name}: FAIL -> {e}")
