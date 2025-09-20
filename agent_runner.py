from dotenv import load_dotenv
load_dotenv()

import json
from typing import Dict, Any

from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

from tools import clean_text, predict_ticket_attributes, create_clickup_task

TOOLS = [clean_text, predict_ticket_attributes, create_clickup_task]

PROMPT_TEMPLATE = """
You are a support ticket processing agent. Your goal is to take a raw ticket, clean it,
predict its attributes, and create a task in ClickUp. You must follow this sequence of tool calls:
1. `clean_text`: To normalize the ticket content.
2. `predict_ticket_attributes`: To get ML predictions for tags, department, etc.
3. `create_clickup_task`: To create the task in the external system.

You must use the output from the previous step as input for the next.
Respond with your final answer ONLY after the `create_clickup_task` tool has been used successfully.
If any tool returns an error, stop immediately and report the failure.

TOOLS:
------
You have access to the following tools:
{tools}

To use a tool, you MUST use the following format:
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: A valid JSON string with the tool's arguments. Example: {{"subject": "example subject", "body": "example body"}}
Observation: the result of the action


When you have the final result from `create_clickup_task`, you MUST respond with the exact JSON from the observation.
Do not add any other text. The format should be:
Thought: I now have the final answer.
Final Answer: {{"ok": true, "task_id": "...", "task_url": "..."}}


Begin!

Ticket Details:
Subject: {subject}
Body: {body}

Agent Scratchpad:
{agent_scratchpad}
"""

def run_agent(subject: str, body: str) -> Dict[str, Any]:
    try:
        llm = ChatOllama(model="llama3", temperature=0.0)
        prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
        agent = create_react_agent(llm=llm, tools=TOOLS, prompt=prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=TOOLS,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
        response = agent_executor.invoke({
            "subject": subject,
            "body": body
        })
        final_output_str = response.get("output", "{}")
        return json.loads(final_output_str)

    except json.JSONDecodeError:
        return {"ok": False, "reason": "agent_did_not_return_valid_json", "raw_output": response.get("output")}
    except Exception as e:
        return {"ok": False, "reason": f"agent_execution_error: {str(e)}"}