import streamlit as st
from database_manager import init_history_db, save_chat_log, get_user_history
import pandas as pd
from sqlalchemy import create_engine, exc
# Import the helper functions
from langchain_helper import run_nl2sql_chain_and_extract, parse_few_shots_input 

# ==============================================
# 1. PAGE CONFIG & GLOSSY CSS
# ==============================================
st.set_page_config(
    page_title="Inquira AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        /* Base Dark Theme */
        .stApp {
            background-color: #0b0e14;
            color: #ffffff;
        }

        /* Glassmorphism Effect for Chat Bubbles */
        .chat-bubble {
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(12px);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }
        
        .user-bubble {
            background: rgba(255, 255, 255, 0.05);
            margin-left: 20%;
            border-right: 4px solid #4CAF50;
        }

        .ai-bubble {
            background: rgba(76, 175, 80, 0.05);
            margin-right: 10%;
            border-left: 4px solid #4CAF50;
        }

        /* Sidebar Styling - Glossy Sidebar */
        section[data-testid="stSidebar"] {
            background: rgba(22, 25, 34, 0.95);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }

        /* Customizing Headers and Cards */
        .stExpander {
            background: rgba(255, 255, 255, 0.02) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            border-radius: 10px !important;
        }

        /* Submit Button Styling */
        .stButton>button {
            background: linear-gradient(45deg, #2e7d32, #4caf50) !important;
            color: white !important;
            border: none !important;
            font-weight: 600 !important;
            transition: 0.3s all ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(76, 175, 80, 0.4);
        }
    </style>
""", unsafe_allow_html=True)

# ==============================================
# 2. SESSION STATE & AUTH
# ==============================================
if not st.user.is_logged_in:
    st.title("🔐 AI SQL Agent: Restricted Access")
    st.info("Please log in with your Google account to access the agent and your saved history.")
    st.button("Log in with Google", on_click=st.login)
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ==============================================
# 3. SIDEBAR: ALL FEATURES PRESERVED
# ==============================================
with st.sidebar:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://thumbs.dreamstime.com/b/vector-halloween-black-bat-animal-icon-sign-isolated-white-background-silhouette-wings-abstract-tattoo-art-concept-101822638.jpg", width=80)
    
    st.success(f"Connected as: {st.user.name}")
    st.button("Log out", on_click=st.logout)

    if st.user.is_logged_in and "history_loaded" not in st.session_state:
        try:
            st.session_state.chat_history = get_user_history(st.user.email)
            st.session_state.history_loaded = True
        except:
            st.session_state.chat_history = []
    
    st.header("⚙️ Agent Configuration")

    with st.expander("🤖 1. AI Model Selection", expanded=False):
        selected_model = st.selectbox(
            "Choose Model",
            ("Gemini-2.5-Flash", "Llama 3.1 (Inquira)"),
            help="Select the underlying LLM powering the agent. Ensure Ollama is running for Llama."
        )

    with st.expander("📂 2. Database Schema", expanded=True):
        db_creds = st.secrets.get("db_connection", {})
        schema_method = st.radio(
            "How will you provide the schema?",
            ("DB Connection (Inputs)", "Paste DDL/Schema Directly"),
            index=0 if db_creds.get('host') else 1
        )

        db_connection_uri = None
        direct_schema_text = None
        include_data_samples = False 

        if schema_method == "DB Connection (Inputs)":
            db_host = st.text_input("Host", value=db_creds.get('host', 'localhost'))
            db_user = st.text_input("Username", value=db_creds.get('user', 'root'))
            db_password = st.text_input("Password", type="password", value=db_creds.get('password', ''))
            db_name = st.text_input("Database Name", value=db_creds.get('name', 'atliq_tshirts'))
            db_port = st.number_input("Port", min_value=1, max_value=65535, value=int(db_creds.get('port', 25305)))
            db_driver = st.selectbox("Driver Type", ("mysql+pymysql", "postgresql+psycopg2", "sqlite:///"), index=0)

            if db_host and db_user and db_password and db_name:
                db_connection_uri = f"{db_driver}://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            
            st.divider()
            st.subheader("3. Data Sampling (Privacy)")
            include_data_samples = st.toggle("Include sample rows in the prompt.", value=False)
        else:
            direct_schema_text = st.text_area(
                "Paste Schema Definitions (DDL)",
                placeholder="CREATE TABLE t_shirts (...);\nCREATE TABLE discounts (...);",
                height=250
            )

    with st.expander("📝 4. Few-Shot Examples", expanded=False):
        use_few_shots = st.toggle("Enable Few-Shot Prompting", value=False)
        few_shot_input = None
        if use_few_shots:
            few_shot_input = st.text_area(
                "Input Few-Shot Examples",
                placeholder="Q: Find the price of all Levi t-shirts\nA: SELECT price FROM t_shirts WHERE brand = 'Levi'\n\n---\n\nQ: List all products\nA: SELECT * FROM t_shirts",
                height=300
            )

    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.last_result = None
        st.rerun()

# ==============================================
# 4. MAIN CHAT AREA
# ==============================================
st.title("🤖 Inquira - Natural Language to SQL")
st.markdown("##### *Transform plain English questions into database queries with conversation history.*")

# Scrollable Chat Container
chat_container = st.container()

with chat_container:
    # Reverse history for chat flow
    for chat in reversed(st.session_state.chat_history):
        st.markdown(f"""<div class="chat-bubble user-bubble"><b>You:</b><br>{chat['Question']}</div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="chat-bubble ai-bubble"><b>Inquira:</b><br>Generated SQL: <code>{chat['SQL']}</code></div>""", unsafe_allow_html=True)

# Chat Input at bottom
nl_query = st.chat_input("Ask a question about your database...")

if nl_query:
    if not nl_query.strip():
        st.error("⚠️ Please enter a natural language query.")
    elif (schema_method == "Paste DDL/Schema Directly" and not direct_schema_text):
        st.error("⚠️ Please provide the DDL schema in the sidebar.")
    elif (schema_method == "DB Connection (Inputs)" and not db_connection_uri):
        st.error("⚠️ Please fill in database connection fields.")
    else:
        with st.spinner("🧠 AI is processing..."):
            try:
                few_shot_examples = parse_few_shots_input(few_shot_input) if use_few_shots and few_shot_input else None
                init_history_db()
                
                result = run_nl2sql_chain_and_extract(
                    model_name=selected_model,
                    db_uri=db_connection_uri if schema_method == "DB Connection (Inputs)" else None,
                    ddl_text=direct_schema_text,
                    nl_query=nl_query,
                    use_few_shots=use_few_shots,
                    include_data_samples=include_data_samples,
                    few_shot_examples=few_shot_examples,
                    chat_history=list(reversed(st.session_state.chat_history[:4]))
                )
                
                if result and 'sql_query' in result:
                    save_chat_log(st.user.email, nl_query, result['sql_query'])
                    st.session_state.chat_history = get_user_history(st.user.email)
                    st.session_state.last_result = result
                    st.rerun()

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ==============================================
# 5. RESULTS DISPLAY (DYNAMIC)
# ==============================================
if st.session_state.last_result:
    res = st.session_state.last_result
    st.divider()
    
    tab_results, tab_logic, tab_edit = st.tabs(["📊 Results", "💡 Logic", "🛠️ Manual Edit"])
    
    with tab_results:
        st.info(res.get('final_answer', "Summary not available."))
        if not res['raw_data'].empty:
            st.dataframe(res['raw_data'], use_container_width=True, hide_index=True)
            
    with tab_logic:
        st.write(res.get('explanation'))
        st.code(res.get('sql_query'), language='sql')

    with tab_edit:
        manual_sql = st.text_area("Edit SQL:", value=res.get('sql_query', ''), height=200)
        if st.button("🚀 Execute Correction"):
            engine = create_engine(db_connection_uri)
            with engine.connect() as conn:
                st.session_state.last_result['raw_data'] = pd.read_sql(manual_sql, conn)
                st.success("Results Updated!")
                st.rerun()
