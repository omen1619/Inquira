from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama
from langchain_community.utilities import SQLDatabase 
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate    
import streamlit as st    

from sqlalchemy import create_engine, exc
import os
import re 
from dotenv import load_dotenv
import pandas as pd 
from typing import List, Dict, Union

load_dotenv()

def get_api_key():
    # Priority 1: Check Streamlit Secrets (Cloud Dashboard or .streamlit/secrets.toml)
    if "GOOGLE_API_KEY" in st.secrets:
        return st.secrets["GOOGLE_API_KEY"]
    # Priority 2: Fallback to local .env file
    return os.getenv("GOOGLE_API_KEY")

# --- Llama-Instruct Template Markers (matching with TRAINING TEMPLATE) ---
BEGIN_OF_TEXT  = "<|begin_of_text|>"

START_SYSTEM   = "<|start_header_id|>system<|end_header_id|>\n\n" 
END_OF_TEXT    = "<|eot_id|>"

START_USER     = "<|start_header_id|>user<|end_header_id|>\n\n" 

START_ASSISTANT= "<|start_header_id|>assistant<|end_header_id|>\n\n"

# --- SYSTEM PROMPT CONTENT ---
SYSTEM_PROMPT_CONTENT = ("You are an **Expert Text-to-SQL Assistant** specializing in complex query generation from Natural Language (NL). "
"Your final output **MUST** be a single, executable SQL query."
"\n\n**Primary Goal:** Convert the NL question into a correct SQL query against the **provided SCHEMA**."
"\n\n**Core Constraints (CRITICAL):**\n1.  **STRICT SCHEMA USAGE (Case Sensitive):** You **MUST** use the exact table and column names from the SCHEMA. **Maintain the EXACT CASE (e.g., if the schema says Customers, use Customers, not customers).** Also, use column name with table name from which it belongs to. **DO NOT** use table or column names found in the EXAMPLES; use only from SCHEMA and treat EXAMPLES as just an reference to understand logic or concept for solving problem statement. **This rule is absolute.**"
"\n2.  **Joining:** You must use explicit `FOREIGN KEY` joins if it is necessary but do not use if it is not necessary and can be solved simply without it."
"\n3.  **Output:** Return **ONLY** the SQL query. **DO NOT** include the CoT/Plan in the final output."
"\n4.  **Hallucination:** Do not hallucinate any table or column name, use only table or column name from provided SCHEMA, not even from EXAMPLES."
"\n5.  **Output Format:** Return **ONLY** the SQL query string. Do not include any markdown fences (```sql), comments, or conversational text."
"\n6.  **Strict Domain Constraint:** Only answer questions strictly related to the provided SCHEMA. If the request is irrelevant (e.g., greetings, non-database queries), you MUST output: 'I am only allowed to answer questions based on the database schema.'"
)

EXPLAIN_PROMPT_CONTENT = (
    "You are a Business Intelligence Analyst. Your task is to explain SQL queries "
    "to non-technical users. \n"
    "**RULES:**\n"
    "1. Do NOT mention technical details like 'ASCII', 'CHAR(37)', or 'Concatenation'.\n"
    "2. Translate 'LIKE' patterns into plain English (e.g., 'starts with', 'contains', 'ends with').\n"
    "3. Focus on the business intent: What specific data is being filtered and why?\n"
    "4. Keep it to 2-3 natural sentences."
)

# --- Few-Shot Example Template ---
FEW_SHOT_TEMPLATE = "Question: {Question}\nSQL: {sql_query}"

FEW_SHOT_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["Question", "sql_query"],
    template=FEW_SHOT_TEMPLATE,
)

# --- EXECUTION GUIDED REFINEMENT PROMPT TEMPLATE (EGR) ---
EGR_PROMPT_TEMPLATE = """
**REFINEMENT REQUIRED (Attempt {attempt_num})**
The previous attempt failed execution. You must fix the SQL and try again.
- **FAILED QUERY:** {failed_sql}
- **EXECUTION ERROR:** {error_message}
- **Original Question:** {nl_query}
SQL:
"""

# --- Helper functions (get_db_connection, get_schema_from_db, etc.) ---
def get_db_connection(db_uri: str, sample_rows: int) -> SQLDatabase:
    """Initializes the SQLDatabase object."""
    try:
        return SQLDatabase.from_uri(db_uri, sample_rows_in_table_info=sample_rows)
    except Exception as e:
        raise RuntimeError(f"{str(e)}")

def get_schema_from_db(db: SQLDatabase) -> str:
    """Generates the schema text."""
    table_info = db.get_table_info() 
    return "SCHEMA:\n" + table_info

# --- HELPER FUNCTIONS FOR CASING AND WILDCARDS ---

def get_schema_map(schema_text: str) -> Dict[str, str]:
    """Parses DDL text to create a robust lowercase-to-CorrectCase mapping for all identifiers."""
    schema_map = {}
    all_words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', schema_text)
    
    sql_keywords_to_ignore = {
        'SELECT', 'FROM', 'WHERE', 'JOIN', 'ON', 'GROUP', 'ORDER', 'BY', 'LIMIT', 'SUM', 'COUNT', 'AS', 
        'CREATE', 'TABLE', 'INT', 'INTEGER', 'VARCHAR', 'TEXT', 'DECIMAL', 'ENUM', 'DATE', 'PRIMARY', 'FOREIGN', 
        'KEY', 'CONSTRAINT', 'CHECK', 'BETWEEN', 'NOT', 'NULL', 'AUTO_INCREMENT', 'DEFAULT', 'COLLATE', 
        'ENGINE', 'REFERENCES', 'INNODB', 'INSERT', 'INTO', 'VALUES', 'AND', 'OR', 'UNION', 'EXISTS'
    }
    
    for word in all_words:
        upper_word = word.upper()
        if upper_word not in sql_keywords_to_ignore:
            schema_map[word.lower()] = word
            
    return schema_map

def normalize_sql_casing(sql_query: str, schema_map: Dict[str, str]) -> str:
    """Replaces lowercase identifiers in the generated SQL with their correct case."""
    def replacer(match):
        identifier = match.group(0) 
        lower_id = identifier.lower()
        if lower_id in schema_map:
            return schema_map[lower_id]
        return identifier 

    pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
    return re.sub(pattern, replacer, sql_query)

def fix_like_wildcards(sql_query: str) -> str:
    """
    ULTIMATE FIX: Converts standard SQL LIKE patterns (e.g., 'A%B%') into the 
    environment-safe CONCAT(A, CHAR(37), B, CHAR(37)) structure using regex.
    This bypasses the Python driver's string formatting bug for the '%' character.
    """
    
    # Pattern: Finds LIKE or NOT LIKE followed by any quoted string containing one or more '%'
    pattern = r'(\b(?:NOT\s+)?LIKE\s+)(\'[^\']*\%[^\']*\')'
    
    def replace_wildcard_parts(match):
        like_clause_prefix = match.group(1) 
        quoted_search_term = match.group(2) 
        
        search_term = quoted_search_term.strip("'")
        
        # 1. Split the string by the literal '%' character
        parts = search_term.split('%')
        
        # 2. Build the new CONCAT arguments list
        concat_args = []
        
        # Handle leading wildcard
        if search_term.startswith('%'):
            concat_args.append('CHAR(37)')
        
        for i, part in enumerate(parts):
            if part: # Add the string literal part
                concat_args.append(f"'{part}'")
            
            # Add CHAR(37) after the part if it was followed by a '%' 
            if i < len(parts) - 1:
                concat_args.append('CHAR(37)')
        
        # Final cleanup for uniqueness and correctness (removing adjacent CHAR(37))
        final_args = []
        for arg in concat_args:
             if arg == 'CHAR(37)' and final_args and final_args[-1] == 'CHAR(37)':
                 continue
             final_args.append(arg)
             
        # Handling the case where the term is ONLY '%'
        if not final_args and search_term == '%':
             final_args = ['CHAR(37)']

        # 4. Assembling the final replacement string
        new_search_term = f"CONCAT({', '.join(final_args)})"
        
        # Rebuild the full clause: LIKE CONCAT(...)
        return like_clause_prefix + new_search_term

    # Apply the regex substitution (case-insensitive flag is important for 'LIKE')
    return re.sub(pattern, replace_wildcard_parts, sql_query, flags=re.IGNORECASE)


# --- MAIN EXECUTION FUNCTION ---
def run_nl2sql_chain_and_extract(
    model_name: str,
    db_uri: str,
    ddl_text: str,
    nl_query: str,
    use_few_shots: bool,
    include_data_samples: bool,
    few_shot_examples: List[Dict[str, str]] = None,
    chat_history: List[Dict[str, str]] = None # Added for history context
) -> Dict[str, Union[str, pd.DataFrame]]:
    
    # ... (Model initialization and Schema setup) ...

    api_key = get_api_key()
    
    # 1. Initialize LLM (Conditional Model Selection)
    if model_name == "Gemini-2.5-Flash":
        llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash',google_api_key=api_key, temperature=0.0)
        is_llama = False
    elif model_name == "Llama 3.1 (Inquira)":
        llm = Ollama(model="inquira", temperature=0.0) 
        is_llama = True
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # 2. Determine Schema Source and Sampling Level
    db = None
    schema_text = ddl_text
    
    sample_rows_count = 3 if include_data_samples and db_uri else 0
    
    if db_uri:
        try:
            db = get_db_connection(db_uri, sample_rows_count) 
            schema_text = get_schema_from_db(db) 
        except RuntimeError as re:
            # Re-raise the clean error so the UI sees it
            raise re
        except Exception as e:
            if not ddl_text:
                raise RuntimeError(f"Database connection failed: {str(e)}")
    
    # 3. Build Schema Casing Map (Crucial for post-processing)
    schema_map = get_schema_map(schema_text)


    # 4. Assemble Few-Shot Examples
    selected_examples = []
    
    # If few-shots are enabled, use ALL provided examples
    if use_few_shots and few_shot_examples and len(few_shot_examples) > 0:
         selected_examples = few_shot_examples 

    # 5. Construct Base Prompt (For first attempt)
    system_block = SYSTEM_PROMPT_CONTENT

    # Build Conversational Context
    history_block = ""
    if chat_history:
        history_block = "\n\n**PREVIOUS CONVERSATION:**\n"
        for turn in chat_history[-3:]: # Send last 3 turns for context
            history_block += f"User: {turn['Question']}\nSQL: {turn['SQL']}\n"

    # --- Start of EGR Implementation ---
    MAX_ATTEMPTS = 4
    raw_llm_output = ""
    predicted_sql = ""
    final_sql_query = ""
    execution_success = False
    execution_error_message = ""
    raw_data_df = pd.DataFrame({'Note': ["No Execution Data"], 'Status': ['Skipped']})
    final_answer = "SQL Execution Skipped"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        # 5. Construct Final Prompt (Conditional Templating)
        if attempt == 1:
            # First attempt: Use the standard prompt structure
            if is_llama:
                # Llama Instruct Format (Turn 1)
                llama_prompt = f"{BEGIN_OF_TEXT}{START_SYSTEM}{system_block}{history_block}{END_OF_TEXT}"
                llama_prompt += f"{START_USER}{schema_text}{END_OF_TEXT}"
                for ex in selected_examples:
                    llama_prompt += f"{START_USER}Question: {ex['Question']}{END_OF_TEXT}"
                    llama_prompt += f"{START_ASSISTANT}{ex['sql_query']}{END_OF_TEXT}"
                
                final_user_turn_content = f"Question: {nl_query}\nSQL:"
                llama_prompt += f"{START_USER}{final_user_turn_content}{END_OF_TEXT}"
                final_prompt = f"{llama_prompt}{START_ASSISTANT}"
            else:
                # Gemini Format (Turn 1)
                prompt_examples_text = "\n\n".join(
                    FEW_SHOT_PROMPT_TEMPLATE.format(Question=ex['Question'], sql_query=ex['sql_query'])
                    for ex in selected_examples
                )
                final_prompt = f"""
                **SYSTEM INSTRUCTION:**
                {system_block}\n

                **conversation history:**
                {history_block}
                
                **FEW-SHOT EXAMPLES:**
                {prompt_examples_text if prompt_examples_text else 'No few-shot examples provided.'}

                **USER INPUT:**
                {schema_text}
                Question: {nl_query}
                SQL:
                """
        else:
            # Refinement attempt (Attempt 2): Use the EGR template
            
            refinement_prompt_content = EGR_PROMPT_TEMPLATE.format(
                attempt_num=attempt,
                failed_sql=final_sql_query,
                error_message=execution_error_message,
                nl_query=nl_query
            )
            
            if is_llama:
                # Llama refinement: New User turn with error feedback
                llama_prompt += f"{START_USER}{refinement_prompt_content}{END_OF_TEXT}"
                final_prompt = f"{llama_prompt}{START_ASSISTANT}"
            else:
                # Gemini refinement
                final_prompt = f"""
                **SYSTEM INSTRUCTION:** {system_block}
                **SCHEMA:** {schema_text}
                **REFINEMENT INPUT:** {refinement_prompt_content}
                """
        
        # 6. Invoke LLM for Query Generation
        print(f"____ Final Prompt (Attempt {attempt}) ____\n{final_prompt}")
        raw_llm_response = llm.invoke(final_prompt)

        if hasattr(raw_llm_response, 'content'):
            raw_llm_output = raw_llm_response.content.strip()
        else:
            raw_llm_output = str(raw_llm_response).strip()

        # 7. Extract SQL Query
        predicted_sql = raw_llm_output.split(END_OF_TEXT)[0].strip()

        # 8. POST-PROCESSING FIXES
        final_sql_query = normalize_sql_casing(predicted_sql, schema_map) 
        final_sql_query = final_sql_query.replace('"', "'")

        print("predicted query : ",final_sql_query,"\n")

        # --- WILDCARD FIX: REGEX-BASED TRANSFORMATION ---
        final_sql_query = fix_like_wildcards(final_sql_query)
        # ----------------------------------------------------------------

        print(f"Attempt {attempt} SQL generated: {final_sql_query}")

        # 9. Check for Domain Constraint/Hallucination
        if predicted_sql.lower().startswith("i am only allowed to answer"):
            return {
                "sql_query": "/* Domain Constraint Triggered */",
                "final_answer": predicted_sql,
                "raw_data": pd.DataFrame({'Note': [predicted_sql], 'Status': ['Refused']})
            }
        
        # 10. Execute SQL Query
        if db:
            try:
                # Execute the fully corrected SQL query
                raw_data_df = pd.read_sql(final_sql_query, db._engine)
                execution_success = True
                print(f"Attempt {attempt} SUCCESSFUL.")
                break # Exit the loop upon successful execution
            except Exception as e:
                execution_error_message = f"SQL error: {e.__class__.__name__}: {e}"
                print(f"Attempt {attempt} FAILED with error: {execution_error_message}")
                # Loop continues to refinement attempt (if attempt < MAX_ATTEMPTS)

        else:
            # If no DB connection (DDL mode), we assume success after post-processing
            execution_success = True
            break
            

    # --- End of EGR Loop ---

    # 11. Final Summary Generation (Runs only once after successful SQL or max attempts reached)
    explanation = ""
    if execution_success:
        # Build history context for the explainer
        explain_history = ""
        if chat_history:
            for turn in chat_history[-2:]:
                explain_history += f"Previous Q: {turn['Question']}\n"
                
        # Construct the multi-context prompt
        explain_prompt = (
            f"{EXPLAIN_PROMPT_CONTENT}\n\n"
            f"**CONVERSATION CONTEXT:**\n{explain_history}\n"
            f"**USER'S CURRENT QUESTION:** {nl_query}\n"
            f"**GENERATED SQL:** {final_sql_query}\n\n"
            "Explanation:"
        )
        explanation_res = llm.invoke(explain_prompt)
        explanation = explanation_res.content if hasattr(explanation_res, 'content') else str(explanation_res)

        if db:
            try:
                # Build history string for the summarizer
                summary_context = ""
                if chat_history:
                    for turn in chat_history[-2:]:
                        summary_context += f"Previous Question: {turn['Question']}\n"
                nl_output_prompt = (
                    "You are an **Expert Data Presentation Assistant** specializing in converting SQL results "
                    "back into clear, detailed natural language. \n"
                    "Your task is to convert the following SQL query results into a natural language sentence "
                    "that **directly and politely answers the user's original question.** \n"
                    "**CONTEXT OF CONVERSATION:**\n"
                    f"{summary_context}\n"  # <--- Added context here
                    "**CRITICAL:** Ensure the final answer explicitly mentions and lists ALL columns "
                    "retrieved in the results (e.g., both the name and the email, if both were selected).\n"
                    f"Current User Question: {nl_query}\n" # Changed to 'Current' for clarity
                    f"SQL Results : {raw_data_df.to_string()}\n\n"
                    "Generate a natural language answer based on these results, being sure to mention ALL retrieved details (e.g., name AND email)."
                )
                
                summary_response = llm.invoke(nl_output_prompt)
                if hasattr(summary_response, 'content'):
                    final_answer = summary_response.content.strip()
                else:
                    final_answer = str(summary_response).strip()
                
            except Exception as e:
                final_answer = f"Summary Generation Failed: {e}"
                
        else:
            final_answer = "SQL Execution Skipped (DDL only mode)"
            raw_data_df = pd.DataFrame({'Note': [f"Execution successful after {attempt} attempt(s) (DDL mode)."], 'Status': ['Skipped']})
    else:
        # If loop finished and execution still failed
        final_answer = f"SQL Execution failed after {MAX_ATTEMPTS} attempts. Last error: {execution_error_message}"
        raw_data_df = pd.DataFrame({'Note': [final_answer], 'Status': ['Error']})

    return {
        "sql_query": final_sql_query, 
        "final_answer": final_answer,
        "raw_data": raw_data_df,
        "explanation": explanation
    }
        
# --- Helper function for parsing few-shot input ---
def parse_few_shots_input(few_shot_input_text: str) -> List[Dict[str, str]]:
    """Parses the streamlit text area input into a list of few-shot dictionaries."""
    examples = []
    raw_examples = few_shot_input_text.strip().split('---') 
    
    for raw_ex in raw_examples:
        raw_ex = raw_ex.strip()
        if not raw_ex:
            continue
            
        q_start = raw_ex.find("Q:")
        a_start = raw_ex.find("A:")
        
        if q_start != -1 and a_start != -1 and a_start > q_start:
            question = raw_ex[q_start + 2:a_start].strip()
            sql_query = raw_ex[a_start + 2:].strip() 
            
            if question and sql_query:
                examples.append({
                    "Question": question, 
                    "sql_query": sql_query,
                })
    return examples