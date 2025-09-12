# Agentic AI Ticket Processing System

This project demonstrates a fully autonomous, agentic workflow for processing support tickets. The system takes a ticket's subject and body, uses an AI agent powered by a local LLM to understand and process it, enriches it with predictions from a multi-model ML pipeline, and automatically creates a fully-detailed task in ClickUp.

---
## Features

* **Agentic Workflow**: Uses the LangChain ReAct (Reasoning and Acting) framework to create a goal-oriented agent.
* **Local LLM**: Powered by a local Large Language Model (e.g., Llama 3) via Ollama for privacy and cost-effectiveness.
* **Multi-Model ML Pipeline**: Enriches tickets with predictions for four different attributes:
    * **Tags** (Multi-label classification with DeBERTa-v3)
    * **Department** (Single-label classification with BERT)
    * **Type** (Single-label classification with DistilBERT)
    * **Priority** (Single-label classification with RoBERTa)
* **ClickUp Integration**: Seamlessly creates and updates tasks in a ClickUp list, including setting priority, tags, and custom fields.
* **Interactive Demo**: Includes a web-based UI built with Streamlit for easy local demonstration.

---
## Setup and Installation

### Prerequisites
* Python 3.11+
* [Ollama](https://ollama.com/) installed and running.
* A local LLM pulled via Ollama. This project is optimized for `llama3`.
    ```bash
    ollama run llama3
    ```

### 1. Clone the Repository
```bash
git clone https://github.com/business-analytics-project-2025/tickets-project
cd tickets-project
```

### 2. Create a Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Model Assets
This project requires pre-trained model weights (`.pt` files) and label mappings (`.json` files).
* Create a folder named `models` in the root of the project.
* Place all your `.pt` and `.json` asset files inside this `models` folder.

### 5. Configure Environment
You need to provide your ClickUp API credentials and IDs.
1.  Create a file named `.env` in the root of the project.
2.  Add the following variables to the `.env` file:
    ```
    CLICKUP_TOKEN="pk_paste_your_token_here"
    CLICKUP_LIST_ID="paste_your_list_id_here"
    CLICKUP_TEAM_ID="paste_your_team_id_here"
    ```
3.  Update the custom field IDs in `tickets/config.py` to match the IDs from your ClickUp list.
    ```python
    # tickets/config.py
    TYPE_FIELD_ID = "paste_your_type_field_id_here"
    DEPT_FIELD_ID = "paste_your_department_field_id_here"
    ```

---
## Usage

### Run the Interactive Web Demo
This is the best way to demonstrate the project.
```powershell
streamlit run ui_streamlit.py
```
Open the provided URL in your browser to use the app.

### Run a Single Test
To run a quick test from the command line:
```powershell
python test_agent.py
```