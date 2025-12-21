from sqlalchemy import create_engine, text
import streamlit as st

url = st.secrets["connections"]["history_db"]["url"]

# Setup the history engine using secrets
history_engine = create_engine(
    url,
    connect_args={
        "ssl": {"ssl_mode": "REQUIRED"} 
    },
    pool_pre_ping=True # Checks if connection is alive before using it
)

def init_history_db():
    """Ensures the history table exists."""
    try:
        with history_engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS chat_logs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_email VARCHAR(255),
                    user_query TEXT,
                    generated_sql TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.commit()
    except Exception as e:
        st.error(f"Failed to initialize History DB: {e}")

def save_chat_log(user_email, question, sql):
    """Saves a single entry to the history database."""
    with history_engine.connect() as conn:
        conn.execute(
            text("INSERT INTO chat_logs (user_email, user_query, generated_sql) VALUES (:email, :q, :s)"),
            {"email": user_email, "q": question, "s": sql}
        )
        conn.commit()

def get_user_history(user_email, limit=10):
    """Fetches past queries for the specific user."""
    try:
        with history_engine.connect() as conn:
            result = conn.execute(
                text("SELECT user_query, generated_sql FROM chat_logs WHERE user_email = :email ORDER BY timestamp DESC LIMIT :limit"),
                {"email": user_email, "limit": limit}
            )
            # Fetchall() ensures the connection is closed before we process the data
            rows = result.fetchall()
            return [{"Question": row.user_query, "SQL": row.generated_sql} for row in rows]
    except Exception as e:
        # Return an empty list so the UI doesn't crash if the DB is temporarily down
        print(f"Error fetching history: {e}")
        return []