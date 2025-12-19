import streamlit as st
import os
import pandas as pd
# Import the helper functions
from langchain_helper import run_nl2sql_chain_and_extract, parse_few_shots_input 

# Set page configuration
st.set_page_config(
    page_title="AI NL-to-SQL Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title of the application
st.title("🤖 Natural Language to SQL AI Agent")
st.markdown("Transform plain English questions into database queries.")

# Helper to construct DB URI from Streamlit secrets (if available)
def get_secrets_creds():
    """Tries to get a dictionary of credentials from st.secrets."""
    if "db_connection" in st.secrets:
        return st.secrets["db_connection"]
    return {}

db_creds = get_secrets_creds()
default_index = 0 if db_creds.get('host') else 1 


# ==============================================
# SIDEBAR: Configuration & Settings
# ==============================================
with st.sidebar:
    st.header("⚙️ Agent Configuration")

    # --- Feature 1: Model Selection ---
    st.subheader("1. AI Model")
    selected_model = st.selectbox(
        "Choose Model",
        ("Gemini-2.5-Flash"),
        help="Select the underlying LLM powering the agent. Ensure Ollama is running for Llama."
    )
    st.info(f"Currently using: **{selected_model}**")

    st.divider()

    # --- Feature 2: Database Schema Source ---
    st.subheader("2. Database Schema")
    schema_method = st.radio(
        "How will you provide the schema?",
        ("DB Connection (Inputs)", "Paste DDL/Schema Directly"),
        index=default_index
    )

    db_connection_uri = None
    direct_schema_text = None
    include_data_samples = False 

    if schema_method == "DB Connection (Inputs)":
        st.caption("Provide connection details to fetch the schema and execute the query.")
        
        # User-friendly separate inputs, pre-filled from secrets if available
        db_host = st.text_input("Host", value=db_creds.get('host', 'localhost'))
        col_db_user, col_db_pass = st.columns(2)
        with col_db_user:
            db_user = st.text_input("Username", value=db_creds.get('user', 'root'))
        with col_db_pass:
            db_password = st.text_input("Password", type="password", value=db_creds.get('password', ''))
        
        col_db_name, col_db_port, col_db_driver = st.columns(3)
        with col_db_name:
            db_name = st.text_input("Database Name", value=db_creds.get('name', 'atliq_tshirts'))
        with col_db_port:
            db_port_val = db_creds.get('port', 10540)
            db_port = st.number_input("Port", min_value=1, max_value=65535, value=int(db_port_val) if db_port_val else 3306)
        with col_db_driver:
             db_driver = st.selectbox("Driver Type", ("mysql+pymysql", "postgresql+psycopg2", "sqlite:///"), index=0)

        # Construct the URI
        if db_host and db_user and db_password and db_name:
            db_connection_uri = f"{db_driver}://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        
        st.divider()
        # --- Feature 3: Data Sampling Toggle ---
        st.subheader("3. Data Sampling (Privacy)")
        include_data_samples = st.toggle(
            "Include sample rows in the prompt.", 
            value=False, 
            help="Warning: If enabled, the LLM will be given 3 sample rows from each table, which exposes real data values for context."
        )
        if not include_data_samples:
            st.caption("Data samples are excluded, minimizing privacy risk.")

    else: # Paste DDL/Schema Directly
        direct_schema_text = st.text_area(
            "Paste Schema Definitions (DDL)",
            placeholder="CREATE TABLE t_shirts (...);\nCREATE TABLE discounts (...);",
            height=250,
            help="Paste the CREATE TABLE statements here for context."
        )
        st.divider()
        st.subheader("3. Data Sampling (N/A)")
        st.caption("Data sampling is irrelevant in DDL-only mode.")


    # --- Feature 4: Few-Shot Prompting ---
    st.subheader("4. Few-Shot Examples")
    use_few_shots = st.toggle("Enable Few-Shot Prompting", value=False)

    few_shot_input = None
    if use_few_shots:
        few_shot_input = st.text_area(
            "Input Few-Shot Examples",
            placeholder="Q: Find the price of all Levi t-shirts\nA: SELECT price FROM t_shirts WHERE brand = 'Levi'\n\n---\n\nQ: List all products\nA: SELECT * FROM t_shirts",
            height=300,
            help="Format: Q: [Question]\nA: [SQL Query]. Separate examples with '---'."
        )
    else:
        st.caption("Few-shot disabled.")


# ==============================================
# MAIN PAGE: Input and Results
# ==============================================

# --- Natural Language Input Field ---
st.subheader("🗣️ Ask your question")
nl_query = st.text_area(
    "Enter Natural Language Query here:",
    height=120,
    placeholder="e.g., Show me the brand and color of all shirts that cost more than 30."
)

# Action Button
col_btn, col_space = st.columns([1, 5])
with col_btn:
    generate_btn = st.button("✨ Generate SQL", type="primary", use_container_width=True)


# --- Result Display Logic ---
if generate_btn:
    # --- Input Validation ---
    if not nl_query.strip():
        st.error("⚠️ Please enter a natural language query first.")
        st.stop()
        
    is_ddl_only_and_empty = (schema_method == "Paste DDL/Schema Directly" and not direct_schema_text)
    is_db_conn_missing = (schema_method == "DB Connection (Inputs)" and not db_connection_uri)

    if is_ddl_only_and_empty:
        st.error("⚠️ Please provide the DDL schema in the sidebar.")
        st.stop()
    elif is_db_conn_missing:
         st.error("⚠️ Please fill in all required database connection fields (Host, User, Password, DB Name).")
         st.stop()
        
    
    # --- Execute Chain ---
    with st.spinner(f"Processing query with {selected_model} and generating SQL..."):
        try:
            few_shot_examples = None
            if use_few_shots and few_shot_input:
                few_shot_examples = parse_few_shots_input(few_shot_input)
            
            # Call the new service function
            result = run_nl2sql_chain_and_extract(
                model_name=selected_model,
                db_uri=db_connection_uri if schema_method == "DB Connection (Inputs)" else None,
                ddl_text=direct_schema_text,
                nl_query=nl_query,
                use_few_shots=use_few_shots,
                include_data_samples=include_data_samples,
                few_shot_examples=few_shot_examples
            )
            
            generated_sql_query = result['sql_query']
            final_answer = result['final_answer']
            result_df = result['raw_data']

        except Exception as e:
            st.error(f"Error during execution: {e}")
            if schema_method == "Paste DDL/Schema Directly":
                 st.warning("If using 'Paste DDL/Schema Directly', actual SQL execution is expected to fail.")
            st.stop()


    # --- Output Display ---
    st.success("Generation complete!")

    # Use tabs to organize the outputs cleanly
    tab_results, tab_sql = st.tabs(["📊 Query Results", "📄 Generated SQL Query"])

    with tab_results:
        st.subheader("Final Answer (Natural Language)")
        st.info(final_answer)
        st.subheader("Raw Data Result")
        
        # Display DataFrame
        st.dataframe(result_df, use_container_width=True, hide_index=True)
        
        if generated_sql_query == "/* Domain Constraint Triggered */":
            st.caption("The model determined the question was outside its domain specialty.")
        elif result_df.empty:
             st.caption("No rows were returned by the query or execution was skipped.")
        else:
             st.caption(f"Query was executed and returned {len(result_df)} rows. (Execution status reflected above.)")


    with tab_sql:
        st.subheader("Predicted SQL Query")
        st.code(generated_sql_query, language='sql')
        
        if generated_sql_query.startswith("/*"):
            st.caption("The SQL query is either missing or the domain constraint was triggered.")
        else:
            is_executed = 'DB Connection (Inputs)' == schema_method and not final_answer.startswith("SQL Execution Failed")
            st.caption(f"SQL generated by the {selected_model} model. This query was {'executed successfully' if is_executed else 'not executed/failed execution'}.")
        
        col_copy, col_dl = st.columns([1,4])
        with col_copy:
             st.button("Copy SQL to Clipboard (Simulated)", disabled=True)