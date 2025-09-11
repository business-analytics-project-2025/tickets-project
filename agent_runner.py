# agent_runner.py
import json
from typing import Dict, Any

from langchain_ollama import ChatOllama # <-- UPDATED IMPORT
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException

# Our new, typed tools
from tools import clean_text, predict_ticket_attributes, create_clickup_task

# The list of tools the agent can use
TOOLS = [clean_text, predict_ticket_attributes, create_clickup_task]

# --- V2 PROMPT: More explicit about JSON format ---
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
Action Input: A valid JSON object with the tool's arguments. Example: {{"subject": "example subject", "body": "example body"}}
Observation: the result of the action
When you have the final result from `create_click_task`, you MUST respond with the exact JSON from the observation.
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
    """
    Initializes and runs the ReAct agent to process a single ticket.
    """
    try:
        # 1. Initialize the LLM
        # ADD the 'stop' argument here to make the agent more disciplined
        llm = ChatOllama(model="llama3", temperature=0.0)

        # 2. Create the prompt
        prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

        # 3. Create the agent
        agent = create_react_agent(llm=llm, tools=TOOLS, prompt=prompt)

        # 4. Create and run the agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=TOOLS,
            verbose=True, # Set to False in production
            handle_parsing_errors=True, # Robustness for LLM output
            max_iterations=5 # Safety break
        )

        # 5. Invoke the agent
        response = agent_executor.invoke({
            "subject": subject,
            "body": body
        })

        # The final output is expected to be a JSON string
        final_output_str = response.get("output", "{}")
        return json.loads(final_output_str)

    except json.JSONDecodeError:
        # If the LLM fails to return valid JSON in the final step
        return {"ok": False, "reason": "agent_did_not_return_valid_json", "raw_output": response.get("output")}
    except Exception as e:
        # Catch any other unexpected errors during agent execution
        return {"ok": False, "reason": f"agent_execution_error: {str(e)}"}