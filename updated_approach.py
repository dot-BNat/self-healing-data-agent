import csv
import os
import tempfile
import duckdb
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from typing_extensions import TypedDict
from typing import Optional

# --- LangGraph Imports ---
from langgraph.graph import END, StateGraph

load_dotenv()
GROQ_TOKEN = os.getenv("GROQ_TOKEN") or os.getenv("HF_TOKEN")


# 1. Define the Graph State
class AgentState(TypedDict):
    user_query: str
    columns: list
    generated_sql: str
    error_message: str
    query_results: Optional[pd.DataFrame]
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

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_path = temp_file.name
            df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)

        return temp_path, df.columns.tolist(), df
    except Exception as e:
        st.error(f"Error parsing dataset: {e}")
        return None, None, None


# --- STREAMLIT UI SETUP ---
st.set_page_config(
    page_title="InsightEngine | Self-Fixing Analyst",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main .block-container { padding-top: 2rem; }
    div[data-testid="stExpander"] { border: 1px solid rgba(49, 51, 63, 0.2); border-radius: 0.5rem; }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/wired/128/6a11cb/artificial-intelligence.png", width=70)
    st.title("InsightEngine")
    st.caption("v2.1 • Powered by LangGraph & HF Router")
    st.markdown("---")

    st.header("⚙️ Core Status")
    if GROQ_TOKEN:
        st.success("🔒 HF API Engine Live")
    else:
        st.error("❌ Missing `HF_TOKEN` / `HUGGINGFACEHUB_API_TOKEN`")

    st.markdown("---")
    st.info(
        "💡 **How it works:** Drop your file, and the ReAct execution graph will auto-generate, debug, and run local DuckDB queries to analyze your data."
    )

# Header Area
col_header, col_upload = st.columns([2, 1])

with col_header:
    st.title("🤖 Self-Fixing Data Agent")
    st.markdown(
        "Drop a messy file, outline your query, and let the agent auto-write, error-check, and fix its own data pipeline in real-time."
    )

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload dataset", type=["csv", "xlsx"], label_visibility="collapsed"
    )

st.markdown("---")

# Main Interactive Workspace
if uploaded_file is not None:
    temp_path, columns, df = preprocess_and_save(uploaded_file)

    if temp_path and columns and df is not None:
        safe_path = temp_path.replace("\\", "/")
        duckdb.execute(f"CREATE OR REPLACE TABLE uploaded_data AS SELECT * FROM read_csv_auto('{safe_path}')")

        st.subheader("📊 Dataset Overview")
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric(label="Total Rows", value=f"{len(df):,}")
        with m2:
            st.metric(label="Total Columns", value=len(columns))
        with m3:
            null_count = int(df.isnull().sum().sum())
            st.metric(
                label="Missing Value Cells",
                value=null_count,
                delta=f"-{null_count}" if null_count > 0 else None,
                delta_color="inverse"
            )
        with m4:
            mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            st.metric(label="Memory Footprint", value=f"{mem_mb:.2f} MB")

        with st.expander("🔍 View Raw Inspection Dataset (First 5 Rows)", expanded=False):
            st.dataframe(df.head(5), use_container_width=True)

        col_explorer_left, col_explorer_right = st.columns([1, 2])

        with col_explorer_left:
            st.markdown("### 🛠️ Schema Inspector")
            st.dataframe(
                pd.DataFrame({"Data Type": df.dtypes.astype(str)}),
                use_container_width=True
            )

        if GROQ_TOKEN:
            client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=GROQ_TOKEN
                )

            # Node 1: SQL Generator
            def generate_sql_node(state: AgentState):
                user_query = state["user_query"]
                cols = state["columns"]
                prev_error = state.get("error_message", "")
                prev_sql = state.get("generated_sql", "")
                retries = state.get("retry_count", 0)

                system_content = (
                    "You are an expert SQL generator. Generate a single, clean, valid DuckDB SQL query. "
                    "The table is named 'uploaded_data'.\n"
                    f"Available columns: {', '.join(cols)}\n\n"
                    "RULES:\n"
                    "- Respond with ONLY the executable SQL query string.\n"
                    "- Do NOT wrap the query in markdown block code tags (no ```sql).\n"
                    "- Lower values in 'SprintFinish' and 'MainRaceFinish' mean better results.\n"
                    "- Use LIKE operations for matching strings (e.g., RiderName LIKE '%Ogura%').\n"
                    "- For consecutive finishes/streaks, use window functions like LAG(MainRaceFinish) OVER (...)."
                )

                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": f"Question: {user_query}"},
                ]

                if prev_error:
                    messages.append({"role": "assistant", "content": prev_sql})
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

                llm = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",  # <-- Groq model string
                    messages=messages,
                    temperature=0.1,
                    max_tokens=256,
                )

                raw_sql = llm.choices[0].message.content
                clean_sql = raw_sql.strip().replace("```sql", "").replace("```", "").strip()

                return {
                    "generated_sql": clean_sql,
                    "retry_count": retries + 1,
                }

            # Node 2: Database Executor
            def execute_sql_node(state: AgentState):
                sql = state["generated_sql"]
                try:
                    res_df = duckdb.query(sql).to_df()
                    return {"query_results": res_df, "error_message": ""}
                except Exception as e:
                    return {"query_results": None, "error_message": str(e)}

            # Router
            def route_post_execution(state: AgentState):
                if not state.get("error_message"):
                    return "success"
                elif state.get("retry_count", 0) >= 3:
                    return "max_retries"
                else:
                    return "retry"

            # Graph Assembly
            workflow = StateGraph(AgentState)
            workflow.add_node("sql_generator", generate_sql_node)
            workflow.add_node("database_executor", execute_sql_node)

            workflow.set_entry_point("sql_generator")
            workflow.add_edge("sql_generator", "database_executor")

            workflow.add_conditional_edges(
                "database_executor",
                route_post_execution,
                {
                    "success": END,
                    "retry": "sql_generator",
                    "max_retries": END,
                },
            )

            agent_graph = workflow.compile()

            # --- EXECUTION TERMINAL ---
            st.markdown("---")
            st.markdown("### ⚡ Execution Terminal")

            user_query = st.text_area(
                "What data insight are you looking for?",
                placeholder="e.g., 'Find the top 5 highest grossing products in Q3.'",
                help="The agent will draft a DuckDB query, execute it, read error codes, and automatically rewrite it up to 3 times."
            )

            if st.button("🚀 Run Analytical Agent", type="primary"):
                if not user_query.strip():
                    st.toast("⚠️ Please type a query or analysis goal first!")
                else:
                    progress_bar = st.progress(0, text="Initializing Agent Infrastructure...")

                    with st.expander("🛠️ Live Agent Execution Stream & Loop State", expanded=True):
                        terminal_col1, terminal_col2 = st.columns([1, 1])
                        with terminal_col1:
                            st.caption("🤖 Thought Process & Logs")
                            status_log = st.empty()
                        with terminal_col2:
                            st.caption("💻 Last Generated Query Draft")
                            code_log = st.empty()

                    initial_state = {
                        "user_query": user_query,
                        "columns": columns,
                        "generated_sql": "",
                        "error_message": "",
                        "query_results": None,
                        "retry_count": 0,
                    }

                    final_output = initial_state.copy()

                    try:
                        # Stream graph events for live UI updates
                        for event in agent_graph.stream(initial_state):
                            for node_name, node_state in event.items():
                                final_output.update(node_state)

                                if node_name == "sql_generator":
                                    attempt = final_output.get("retry_count", 1)
                                    progress_bar.progress(
                                        min(30 * attempt, 90),
                                        text=f"Drafting SQL (Attempt {attempt})..."
                                    )
                                    status_log.info(f"🔄 Generation Loop #{attempt}: SQL Draft generated.")
                                    code_log.code(final_output.get("generated_sql", ""), language="sql")

                                elif node_name == "database_executor":
                                    err = final_output.get("error_message")
                                    if err:
                                        status_log.warning(f"⚠️ SQL Execution Failed:\n`{err}`\n\nRouting to self-healing node...")
                                    else:
                                        status_log.success("✅ Execution Succeeded! Valid query compiled.")

                        progress_bar.progress(100, text="Graph Execution Finalized.")

                    except Exception as e:
                        st.error(f"Fatal Engine Pipeline Crash: {str(e)}")
                        st.stop()

                    # Final Metrics & Tabs
                    st.markdown("### 🎯 Final Execution Insights")
                    metric_col1, metric_col2, metric_col3 = st.columns(3)

                    retries = final_output.get("retry_count", 1) - 1
                    with metric_col1:
                        if retries == 0:
                            st.metric(label="Self-Healing Loops Required", value="0 (Flawless)", delta="First-pass success", delta_color="normal")
                        else:
                            st.metric(label="Self-Healing Loops Required", value=f"{retries} Rounds", delta=f"{retries} errors auto-fixed", delta_color="inverse")

                    with metric_col2:
                        status_verdict = "Success" if not final_output.get("error_message") else "Failed"
                        st.metric(label="Final Run Status", value=status_verdict)

                    with metric_col3:
                        results_df = final_output.get("query_results")
                        row_count = len(results_df) if results_df is not None else 0
                        st.metric(label="Extracted Result Shape", value=f"{row_count} rows")

                    tab_data, tab_code = st.tabs(["📊 Extracted Insights Table", "📜 Inspected Agent Code"])

                    with tab_code:
                        st.caption("This is the final SQL verified by the self-healing compiler loop.")
                        st.code(final_output.get("generated_sql", "-- No SQL generated."), language="sql")

                    with tab_data:
                        if not final_output.get("error_message"):
                            if results_df is not None and not results_df.empty:
                                st.dataframe(results_df, use_container_width=True)
                            else:
                                st.info("The query compiled successfully but returned an empty dataset.")
                        else:
                            st.error(
                                f"🚨 **Self-Healing Failed:** The agent exceeded maximum retries.\n\n"
                                f"**Final Error:** `{final_output['error_message']}`"
                            )
else:
    st.info("📥 Upload a CSV or Excel file above to begin analysis.")