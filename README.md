
# 🤖 Self-Healing Data Agent (LangGraph & DuckDB)

An intelligent, stateful AI data analytics agent that converts natural language questions into executable SQL queries. Built on a cyclic, multi-node LangGraph architecture, the application features an automated **Self-Healing Error Correction Loop** that catches SQL compilation and runtime execution errors, interprets traceback logs, and dynamically self-corrects query logic without crashing.

## 🚀 Key Features

### State-Driven Agentic Workflow

* Built using **LangGraph StateGraph** to preserve data state across non-linear, cyclic computational passes.
* Enables reliable multi-step reasoning and iterative query refinement.

### Self-Healing Runtime Loop

* Intercepts database syntax errors, schema binder exceptions, and missing `GROUP BY` clauses.
* Feeds execution diagnostics back into the LLM context for automated SQL repair and regeneration.

### High-Performance OLAP Layer

* Uses an in-memory **DuckDB** engine for fast columnar analytics.
* Supports efficient type coercion, aggregation, and dataset normalization.

### Interactive Analytics Frontend

* Built with **Streamlit** for dataset uploads and real-time interaction.
* Displays agent execution rounds, query generation progress, and analytics outputs.

---

## 🏗️ Architecture Design

The workflow replaces a fragile one-shot pipeline with a resilient state machine composed of two primary nodes and a conditional routing mechanism.

### 1. Code Generator Node (`sql_generator`)

* Reads the current workflow state.
* Analyzes available table schemas and previous execution errors.
* Generates optimized DuckDB SQL queries using an LLM.

### 2. Database Executor Node (`database_executor`)

* Executes generated SQL queries against the in-memory DuckDB database.
* On success, stores the query results in the shared state.
* On failure, captures the complete error trace and returns it to the workflow.

### 3. Conditional Routing Edge

Routes execution based on query outcomes:

* **Success** → End workflow.
* **Database Error + Retry Count < 3** → Return to `sql_generator` for correction.
* **Maximum Retries Reached** → Terminate execution safely.

---

## 🛠️ Tech Stack

**Programming & Agent Frameworks**

* Python
* LangGraph
* LangChain

**LLM & AI**

* Meta-Llama-3-8B-Instruct (via Hugging Face API Router)

**Data Processing**

* DuckDB
* Pandas
* NumPy

**Frontend**

* Streamlit
