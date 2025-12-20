import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, exc
# Import the helper functions
from langchain_helper import run_nl2sql_chain_and_extract, parse_few_shots_input 

# ==============================================
# 1. PAGE CONFIGURATION & THEME-SAFE CSS
# ==============================================
st.set_page_config(
    page_title="AI NL-to-SQL Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme-safe CSS: Uses variables to ensure Dark/Light mode compatibility
st.markdown("""
    <style>
        .stApp {
            background-color: var(--background-color);
        }
        
        /* Sidebar styling with theme-aware borders */
        section[data-testid="stSidebar"] {
            border-right: 1px solid var(--secondary-background-color);
        }
        
        /* Modern Card styling for Session History - Theme Safe */
        .history-card {
            background-color: var(--secondary-background-color);
            padding: 12px;
            border-radius: 10px;
            border-left: 5px solid #4CAF50;
            margin-bottom: 12px;
            color: var(--text-color);
            font-size: 0.9rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        /* Buttons and Inputs */
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 15px;
        }
        .stTabs [data-baseweb="tab"] {
            font-weight: 600;
            padding: 10px 20px;
        }
    </style>
""", unsafe_allow_html=True)

# ==============================================
# 2. SESSION STATE INITIALIZATION
# ==============================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# Header Section with Icons
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    st.image("main_logo.jpg",width=256)
    
st.title("🤖 Inquira - Natural Language to SQL AI Agent")
st.markdown("##### *Transform plain English questions into database queries with conversation history.*")

# Helper to construct DB URI from Streamlit secrets
def get_secrets_creds():
    """Tries to get a dictionary of credentials from st.secrets with error safety."""
    try:
        if "db_connection" in st.secrets:
            return st.secrets["db_connection"]
    except Exception:
        pass
    return {}

db_creds = get_secrets_creds()
default_index = 0 if db_creds.get('host') else 1 

# ==============================================
# 3. SIDEBAR: Configuration & Settings
# ==============================================
with st.sidebar:
    # --- LOGO ---
    st.image("https://thumbs.dreamstime.com/b/vector-halloween-black-bat-animal-icon-sign-isolated-white-background-silhouette-wings-abstract-tattoo-art-concept-101822638.jpg", width=80)
    st.header("⚙️ Agent Configuration")

    # --- Feature 1: Model Selection ---
    with st.expander("🤖 1. AI Model Selection", expanded=True):
        selected_model = st.selectbox(
            "Choose Model",
            ("Gemini-2.5-Flash"),
            help="Select the underlying LLM powering the agent. Ensure Ollama is running for Llama."
        )
        st.info(f"Active: **{selected_model}**")

    st.divider()

    # --- Feature 2: Database Schema Source ---
    with st.expander("📂 2. Database Schema", expanded=True):
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
            
            # VERTICAL ARRANGEMENT OF DB COMPONENTS
            db_host = st.text_input("Host", value=db_creds.get('host', 'localhost'))
            db_user = st.text_input("Username", value=db_creds.get('user', 'root'))
            db_password = st.text_input("Password", type="password", value=db_creds.get('password', ''))
            db_name = st.text_input("Database Name", value=db_creds.get('name', 'atliq_tshirts'))
            
            # PORT WITH PLUS MINUS SELECTOR
            db_port_val = db_creds.get('port', 3306)
            db_port = st.number_input("Port", min_value=1, max_value=65535, value=int(db_port_val))
            
            db_driver = st.selectbox("Driver Type", ("mysql+pymysql", "postgresql+psycopg2", "sqlite:///"), index=0)

            if db_host and db_user and db_password and db_name:
                db_connection_uri = f"{db_driver}://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            
            st.divider()
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
    with st.expander("📝 4. Few-Shot Examples", expanded=False):
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

    # --- SESSION HISTORY ---
    st.divider()
    st.subheader("📜 Session History")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.last_result = None
        st.rerun()
    
    # Render history with custom card styling
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        st.markdown(f"""
            <div class="history-card">
                <small style='color: gray;'>Query {len(st.session_state.chat_history)-i}</small><br>
                <b>Q:</b> {chat['Question'][:60]}...
            </div>
        """, unsafe_allow_html=True)


# ==============================================
# 4. MAIN PAGE: Input and Results
# ==============================================

# User Input Card
st.container()
with st.expander("🗣️ Natural Language Input", expanded=True):
    nl_query = st.text_area(
        "Describe the data you need in plain English:",
        height=120,
        placeholder="e.g., Show me the brand and color of all shirts that cost more than 30."
    )
    
    col_btn, _ = st.columns([1, 5])
    with col_btn:
        generate_btn = st.button("✨ Generate SQL", type="primary", use_container_width=True)

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
         st.error("⚠️ Please fill in all required database connection fields.")
         st.stop()
        
    with st.spinner(f"🧠 AI is thinking... Analyzing with {selected_model}"):
        try:
            # Error safety for few-shot parsing
            few_shot_examples = None
            if use_few_shots and few_shot_input:
                try:
                    few_shot_examples = parse_few_shots_input(few_shot_input)
                except Exception as e:
                    st.warning(f"⚠️ Parsing Error in Few-Shots: {e}. Running with zero-shot instead.")
            
            # Call helper with full Exception Handling for LLM connectivity
            result = run_nl2sql_chain_and_extract(
                model_name=selected_model,
                db_uri=db_connection_uri if schema_method == "DB Connection (Inputs)" else None,
                ddl_text=direct_schema_text,
                nl_query=nl_query,
                use_few_shots=use_few_shots,
                include_data_samples=include_data_samples,
                few_shot_examples=few_shot_examples,
                chat_history=st.session_state.chat_history
            )
            
            # Persist successful results
            st.session_state.last_result = result
            st.session_state.chat_history.append({
                "Question": nl_query,
                "SQL": result.get('sql_query', 'N/A')
            })

        # 1. CATCH DATABASE ERRORS FIRST (Most common)
        except (exc.SQLAlchemyError, RuntimeError) as e:
            st.error(f"🔌 **Database Connectivity Error**\n\n{str(e)}")
            st.info("💡 **Tip:** This usually means the host is unreachable, the port is blocked, or your credentials (User/Pass) are wrong.")
            st.stop()

        # 2. CATCH AI MODEL ERRORS
        except ConnectionError:
            st.error("❌ **API Connection Failed**: Could not reach the AI model provider (Gemini/Ollama).")
            st.stop()

        # 3. CATCH EVERYTHING ELSE (Last Resort)
        except Exception as e:
            st.error(f"❌ **Unexpected Error**\n\n{str(e)}")
            st.stop()

# ==============================================
# 5. DISPLAY RESULTS
# ==============================================
if st.session_state.last_result:
    res = st.session_state.last_result
    st.divider()

    tab_results, tab_logic, tab_edit = st.tabs([
        "📊 Query Results", 
        "💡 Logic Explanation", 
        "🛠️ Manual Correction"
    ])

    with tab_results:
        st.subheader("Final Answer (Natural Language)")
        st.info(res.get('final_answer', "Summary not available."))
        
        st.subheader("Raw Data Results")
        try:
            if res['raw_data'] is not None and not res['raw_data'].empty:
                st.dataframe(res['raw_data'], use_container_width=True, hide_index=True)
                st.caption(f"Success: {len(res['raw_data'])} rows retrieved.")
            else:
                st.warning("No data found for this query.")
        except Exception as e:
            st.error(f"Data Display Error: {e}")

    with tab_logic:
        st.subheader("AI Reasoning")
        st.write(res.get('explanation', "Logic breakdown not available for this run."))
        st.subheader("Generated SQL Query")
        st.code(res.get('sql_query', '-- No SQL code'), language='sql')

    with tab_edit:
        st.subheader("🛠️ Interactive Query Correction")
        st.info("If the AI logic missed a nuance, refine the SQL below and update the results instantly.")
        
        manual_sql = st.text_area("Edit SQL Query:", value=res.get('sql_query', ''), height=250)
        
        col_run, col_info = st.columns([1, 3])
        with col_run:
            if st.button("🚀 Re-execute SQL", use_container_width=True):
                try:
                    if schema_method == "DB Connection (Inputs)" and db_connection_uri:
                        engine = create_engine(db_connection_uri)
                        with engine.connect() as conn:
                            updated_df = pd.read_sql(manual_sql, conn)
                        
                        # Update the session state
                        st.session_state.last_result['raw_data'] = updated_df
                        st.session_state.last_result['sql_query'] = manual_sql
                        if st.session_state.chat_history:
                            st.session_state.chat_history[-1]['SQL'] = manual_sql 
                        
                        st.success("✅ Results Updated!")
                        st.rerun()
                    else:
                        st.error("⚠️ Manual re-execution requires an active DB connection.")
                except exc.ProgrammingError as e:
                    st.error(f"⚠️ SQL Syntax Error: {e.orig}")
                except Exception as e:
                    st.error(f"⚠️ Execution Error: {str(e)}")
        with col_info:
            st.caption("Editing the SQL here will also update the context for your next conversational question.")