import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, exc
from database_manager import init_history_db, save_chat_log, get_user_history
from langchain_helper import run_nl2sql_chain_and_extract, parse_few_shots_input 

# Basic page setup - keeping it simple
st.set_page_config(page_title="Inquira SQL AI", page_icon="🤖", layout="wide")

# --- AUTH GATE ---
# Throwing a quick check here to make sure they're logged in before loading everything else
if not st.user.is_logged_in:
    st.title("🔐 Access Restricted")
    st.warning("You gotta log in with Google to use the SQL agent and see your history.")
    st.button("Log in with Google", on_click=st.login)
    st.stop()

# --- STYLING (The 'Inquira' look) ---
st.markdown("""
    <style>
        .stApp { background-color: var(--background-color); }
        /* Clean sidebar border */
        section[data-testid="stSidebar"] { border-right: 1px solid #444; }
        /* History cards that actually look good */
        .history-card {
            background: #262730; padding: 15px; border-radius: 8px;
            border-left: 4px solid #00d4ff; margin-bottom: 10px;
            font-size: 0.85rem; box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        }
        .stButton>button { border-radius: 6px; }
    </style>
""", unsafe_allow_html=True)

# --- APP STATE ---
# Just initializing things if they don't exist yet
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# Grab history from the DB if we haven't already this session
if "history_loaded" not in st.session_state:
    try:
        st.session_state.chat_history = get_user_history(st.user.email)
        st.session_state.history_loaded = True
    except Exception:
        st.session_state.chat_history = []

# --- SIDEBAR CONFIG ---
with st.sidebar:
    st.header("Inquira Settings")
    
    # Profile bit
    st.write(f"Logged in as: **{st.user.name}**")
    if st.button("Sign Out"):
        st.logout()
    
    st.divider()

    # DB Connection Setup
    with st.expander("🛠️ Database Connection", expanded=True):
        # Fallback to secrets if they exist, otherwise use defaults
        creds = st.secrets.get("db_connection", {})
        
        host = st.text_input("Host", value=creds.get('host', 'localhost'))
        user = st.text_input("User", value=creds.get('user', 'root'))
        pw = st.text_input("Password", type="password", value=creds.get('password', ''))
        db = st.text_input("Database", value=creds.get('name', 'atliq_tshirts'))
        port = st.number_input("Port", value=int(creds.get('port', 3306)))
        driver = st.selectbox("Driver", ["mysql+pymysql", "postgresql+psycopg2", "sqlite:///"])

        uri = f"{driver}://{user}:{pw}@{host}:{port}/{db}"
        sampling = st.toggle("Include sample data in prompt?", value=False)

    # Training/Examples
    with st.expander("🧠 Few-Shot Training", expanded=False):
        use_shots = st.toggle("Use examples?", value=False)
        shots_raw = st.text_area("Q: Question... A: SQL...", height=150)

    # History list (only showing the last 5 for neatness)
    st.divider()
    st.subheader("Your Recent Queries")
    if st.button("Clear All History"):
        st.session_state.chat_history = []
        st.rerun()
    
    for item in reversed(st.session_state.chat_history[-5:]):
        st.markdown(f'<div class="history-card">{item["Question"][:60]}...</div>', unsafe_allow_html=True)

# --- MAIN CONTENT ---
st.title("Inquira: Talk to your Data")
st.write("Turn your English questions into SQL queries instantly.")

# Quick Suggestions (UX trick to get people started)
cols = st.columns(3)
chips = ["Show all inventory", "Top 5 brands", "Stock by color"]
for i, chip in enumerate(chips):
    if cols[i].button(chip, use_container_width=True):
        st.session_state.temp_prompt = chip

# Show the chat history as bubbles
for msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(msg["Question"])
    with st.chat_message("assistant"):
        st.code(msg.get("SQL", ""), language="sql")

# The Chat Bar
user_input = st.chat_input("What would you like to know?")

# Handling suggestion click or typing
active_input = getattr(st.session_state, 'temp_prompt', user_input)
if active_input:
    if hasattr(st.session_state, 'temp_prompt'):
        del st.session_state.temp_prompt # clean up

    with st.chat_message("user"):
        st.write(active_input)

    # The 'Thinking' Phase
    with st.status("Working on your query...", expanded=True) as status:
        try:
            st.write("Checking schema and connection...")
            init_history_db()
            
            shots = None
            if use_shots and shots_raw:
                shots = parse_few_shots_input(shots_raw)
            
            st.write("Consulting Gemini for the SQL...")
            # Grabbing the last few turns for context
            context = list(reversed(st.session_state.chat_history[:4]))
            
            res = run_nl2sql_chain_and_extract(
                model_name="Gemini-2.5-Flash",
                db_uri=uri,
                ddl_text=None,
                nl_query=active_input,
                use_few_shots=use_shots,
                include_data_samples=sampling,
                few_shot_examples=shots,
                chat_history=context
            )
            
            if res and 'sql_query' in res:
                save_chat_log(st.user.email, active_input, res['sql_query'])
                st.session_state.chat_history = get_user_history(st.user.email)
                st.session_state.last_result = res
                status.update(label="Got it!", state="complete", expanded=False)
                st.rerun()
                
        except Exception as e:
            status.update(label="Something went wrong", state="error")
            st.error(f"Error: {e}")

# --- THE RESULTS VIEW ---
if st.session_state.last_result:
    res_data = st.session_state.last_result
    st.divider()
    
    tab_data, tab_details, tab_edit = st.tabs(["📊 Table", "🧠 Logic", "🔧 Edit"])
    
    with tab_data:
        st.success(res_data.get('final_answer', 'Done!'))
        if res_data.get('raw_data') is not None:
            st.dataframe(res_data['raw_data'], use_container_width=True, hide_index=True)
    
    with tab_details:
        st.write("**How I figured this out:**")
        st.write(res_data.get('explanation', 'No explanation provided.'))
        st.code(res_data.get('sql_query', ''), language='sql')

    with tab_edit:
        new_sql = st.text_area("Tweak the SQL here:", value=res_data.get('sql_query', ''), height=150)
        if st.button("Run Updated SQL"):
            try:
                # Create a local engine and run the manual query
                eng = create_engine(uri)
                df = pd.read_sql(new_sql, eng)
                st.session_state.last_result['raw_data'] = df
                st.session_state.last_result['sql_query'] = new_sql
                st.rerun()
            except Exception as e:
                st.error(f"SQL Error: {e}")
