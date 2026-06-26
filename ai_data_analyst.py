import csv
import os
import tempfile
import duckdb
import pandas as pd
import streamlit as st 
from openai import OpenAI

# Retrieve the token safely from environment memory
HF_TOKEN = os.getenv("HF_TOKEN")


def preprocess_and_save(file):
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(
                file, encoding="utf-8", na_values=["NA", "N/A", "missing", " ", ""]
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


st.set_page_config(page_title="Data Analyst Agent", page_icon="📊")
st.title("📊 Data Analyst Chain")

# Sidebar Configuration (Simplified!)
with st.sidebar:
    st.header("Configuration")
    if HF_TOKEN:
        st.success("🔒 API Token loaded")
    else:
        st.error("❌ Missing HF_TOKEN")

uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file", type=["csv", "xlsx"]
)

if uploaded_file is not None:
    temp_path, columns, df = preprocess_and_save(uploaded_file)

    if temp_path and columns and df is not None:
        st.write("### Uploaded Data Preview:")
        st.dataframe(df)

        safe_path = temp_path.replace("\\", "/")
        duckdb.execute(
            f"CREATE OR REPLACE TABLE uploaded_data AS SELECT * FROM read_csv_auto('{safe_path}')"
        )

        # Gate execution on whether the token was found in the environment
        if HF_TOKEN:
            client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=HF_TOKEN,  # Pass the loaded token here
            )

            user_query = st.text_area(
                "Ask an analytical question about your data:"
            )

            if st.button("Submit Query"):
                if user_query.strip() == "":
                    st.warning("Please type a question.")
                else:
                    try:
                        with st.spinner("Compiling database analysis..."):
                            completion = client.chat.completions.create(
                                model="meta-llama/Meta-Llama-3-8B-Instruct",
                                messages=[
                                    {
                                        "role": "system",
                                        "content": (
                                            "You are an expert SQL generator. Generate a single, clean, valid DuckDB SQL query. "
                                            "The table is named 'uploaded_data'.\n"
                                            f"Available columns: {', '.join(columns)}\n\n"
                                            "RULES:\n"
                                            "- Respond with ONLY the executable SQL query string.\n"
                                            "- Do NOT wrap the query in markdown block code tags.\n"
                                            "- Lower values in 'SprintFinish' and 'MainRaceFinish' mean better results.\n"
                                            "- Use LIKE operations for matching strings (e.g., RiderName LIKE '%Ogura%')."
                                        ),
                                    },
                                    {"role": "user", "content": user_query},
                                ],
                                temperature=0.1,
                                max_tokens=256,
                            )

                            raw_sql = completion.choices[0].message.content
                            clean_sql = (
                                raw_sql.strip()
                                .replace("```sql", "")
                                .replace("```", "")
                            )

                            st.code(clean_sql, language="sql")
                            query_results = duckdb.query(clean_sql).to_df()

                        st.markdown("### Output Results View:")
                        st.dataframe(query_results)

                    except Exception as e:
                        st.error(f"Execution Engine Error: {e}")
        else:
            st.info("Please add your HF_TOKEN to the .env file to continue.")
