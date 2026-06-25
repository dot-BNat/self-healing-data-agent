import csv
import os
import tempfile
import duckdb
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# --- LangGraph Imports ---
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


# 1. Define the Graph State
# This object keeps track of our data as it moves through different nodes
class AgentState(TypedDict):
    user_query: str
    columns: list
    generated_sql: str
    error_message: str
    query_results: pd.DataFrame
    retry_count: int


def preprocess_and_save(file):
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(
                file,
                encoding="utf-8",
                na_values=["NA", "N/A", "missing", " ", ""],
            )
        elif file.name.endswith(".xlsx"):
            df = pd.read_excel(file, na_values=["NA", "N/A", "missing", " ", ""])
        else:
            st.error("Unsupported file format.")
            return None, None, None

        for col in df.select_dtypes(include=["object"]):
            df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)

        for col in df.columns:
            if "date" in col.lower():
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif df[col].dtype == "object":
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".csv"
        ) as temp_file:
            temp_path = temp_file.name
            df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)

        return temp_path, df.columns.tolist(), df
    except Exception as e:
        st.error(f"Error parsing dataset: {e}")
        return None, None, None


# --- STREAMLIT UI SETUP ---
st.set_page_config(
    page_title="Self-Fixing Data Analyst", page_icon="🤖", layout="wide"
)
st.title("🤖 Self-Fixing Data Agent (LangGraph)")

with st.sidebar:
    st.header("Configuration")
    if HF_TOKEN:
        st.success("🔒 API Token loaded")
    else:
        st.error("❌ Missing HF_TOKEN in your .env file!")

uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file", type=["csv", "xlsx"]
)

if uploaded_file is not None:
    temp_path, columns, df = preprocess_and_save(uploaded_file)

    if temp_path and columns and df is not None:
        st.write("### Uploaded Data Preview:")
        st.dataframe(df.head(5))

        safe_path = temp_path.replace("\\", "/")
        duckdb.execute(
            f"CREATE OR REPLACE TABLE uploaded_data AS SELECT * FROM read_csv_auto('{safe_path}')"
        )

        if HF_TOKEN:
            client = OpenAI(
                base_url="https://router.huggingface.co/v1", api_key=HF_TOKEN
            )

            # --- LANGGRAPH NODE DEFINITIONS ---

            # Node 1: The Code Generator (Writes or Rewrites SQL)
            def generate_sql_node(state: AgentState):
                user_query = state["user_query"]
                cols = state["columns"]
                prev_error = state.get("error_message", "")
                prev_sql = state.get("generated_sql", "")
                retries = state.get("retry_count", 0)

                # Custom system prompt injection if it's a self-healing retry
                system_content = (
                    "You are an expert SQL generator. Generate a single, clean, valid DuckDB SQL query. "
                    "The table is named 'uploaded_data'.\n"
                    f"Available columns: {', '.join(cols)}\n\n"
                    "RULES:\n"
                    "- Respond with ONLY the executable SQL query string.\n"
                    "- Do NOT wrap the query in markdown block code tags.\n"
                    "- Lower values in 'SprintFinish' and 'MainRaceFinish' mean better results.\n"
                    "- Use LIKE operations for matching strings (e.g., RiderName LIKE '%Ogura%').\n"
                    "- CRITICAL FOR CONSECUTIVE/STREAKS: To find consecutive finishes, use window functions "
                    "like LAG(MainRaceFinish) OVER (PARTITION BY RiderName ORDER BY Date/RaceNumber) to compare a race with the previous one."
                )

                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": f"Question: {user_query}"},
                ]

                # If we are retrying, feed the error back into the LLM context!
                if prev_error:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": f"My previous query was: {prev_sql}",
                        }
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"That query failed with this database error: {prev_error}\n"
                                "Please fix the query. Remember, if selecting a regular column "
                                "alongside an aggregate function (MIN, MAX, AVG), you MUST use a GROUP BY clause or subquery!"
                            ),
                        }
                    )

                completion = client.chat.completions.create(
                    model="meta-llama/Meta-Llama-3-8B-Instruct",
                    messages=messages,
                    temperature=0.1,
                    max_tokens=256,
                )

                raw_sql = completion.choices[0].message.content
                clean_sql = (
                    raw_sql.strip().replace("```sql", "").replace("```", "")
                )

                return {
                    "generated_sql": clean_sql,
                    "retry_count": retries + 1,
                }

            # Node 2: The Database Executor (Tests the SQL against DuckDB)
            def execute_sql_node(state: AgentState):
                sql = state["generated_sql"]
                try:
                    # Run query
                    res_df = duckdb.query(sql).to_df()
                    # Clear previous errors if successful
                    return {"query_results": res_df, "error_message": ""}
                except Exception as e:
                    # Catch the exact database binder or syntax error
                    return {"error_message": str(e)}

            # Conditional Edge Router: Evaluates where the graph steps next
            def route_post_execution(state: AgentState):
                if state["error_message"] == "":
                    return "success"  # Exit the graph
                elif state["retry_count"] >= 3:
                    return "max_retries"  # Give up after 3 attempts
                else:
                    return "retry"  # Send back to generator node

            # --- BUILD THE LANGGRAPH STATE MACHINE ---
            workflow = StateGraph(AgentState)

            # Add nodes to the blueprint
            workflow.add_node("sql_generator", generate_sql_node)
            workflow.add_node("database_executor", execute_sql_node)

            # Set linear starting pipeline entrypoint
            workflow.set_entry_point("sql_generator")
            workflow.add_edge("sql_generator", "database_executor")

            # Add the cyclic feedback edge loop
            workflow.add_conditional_edges(
                "database_executor",
                route_post_execution,
                {
                    "success": END,
                    "retry": "sql_generator",
                    "max_retries": END,
                },
            )

            # Compile the graph
            agent_graph = workflow.compile()

            # --- USER INTERFACE RUNTIME ---
            user_query = st.text_area(
                "Ask your self-healing data agent a question:"
            )

            if st.button("Submit Query"):
                if user_query.strip() == "":
                    st.warning("Please type a question.")
                else:
                    # UI placeholders to show the background agent working live
                    status_log = st.empty()
                    code_log = st.empty()

                    initial_state = {
                        "user_query": user_query,
                        "columns": columns,
                        "generated_sql": "",
                        "error_message": "",
                        "retry_count": 0,
                    }

                    # Execute the graph state loop
                    with st.spinner("Agent running graph operations..."):
                        final_output = agent_graph.invoke(initial_state)

                    # Display what happened behind the scenes
                    st.write(
                        f"🔄 **Total Agent Rounds:** {final_output['retry_count']}"
                    )
                    st.code(final_output["generated_sql"], language="sql")

                    if final_output["error_message"] == "":
                        st.markdown("### Output Results View:")
                        st.dataframe(final_output["query_results"])
                    else:
                        st.error(
                            f"Agent failed to fix query after 3 loops. Final Error: {final_output['error_message']}"
                        )
        else:
            st.info("Please add your HF_TOKEN to the .env file to continue.")