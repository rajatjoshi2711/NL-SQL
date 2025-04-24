# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 00:07:19 2025

@author: yashr
"""

import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
# import plotly.graph_objects as go
import json
import re
import os
import openai

# Set page config
st.set_page_config(page_title="SQL Data Explorer", layout="wide")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "db_schema" not in st.session_state:
    st.session_state.db_schema = ""

# Function to connect to SQLite DB
@st.cache_resource
def get_connection(db_path):
    return sqlite3.connect(db_path, check_same_thread=False)

# Function to get database schema
def get_db_schema(conn):
    cursor = conn.cursor()
    
    # Get list of tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    schema = {}
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        
        # Format column information
        col_info = []
        for col in columns:
            col_info.append({
                "name": col[1],
                "type": col[2],
                "primary_key": bool(col[5])
            })
        
        schema[table_name] = col_info
        
        # Get a sample of data to understand content
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
        sample_data = cursor.fetchall()
        
        # Create sample data display
        sample_rows = []
        for row in sample_data:
            sample_rows.append(dict(zip([col["name"] for col in col_info], row)))
        
        schema[f"{table_name}_sample"] = sample_rows
    
    return schema

# Function to execute SQL query
def execute_query(conn, query):
    try:
        df = pd.read_sql_query(query, conn)
        return df, None
    except Exception as e:
        return None, str(e)

def get_openai_response(question, schema_info):
    
    # Base OpenAI API Configuration
    OPENAI_API_KEY = key  # Your OpenAI API key here
    MODEL_NAME = "gpt-4o-mini"  # Use appropriate OpenAI model name
    
    agent_prompt = f"""You are an expert in SQL and data analysis. Based on the database schema below, generate a SQL query to answer the user's question.
    
DATABASE SCHEMA:
{json.dumps(schema_info, indent=2)}
USER QUESTION: {question}
Provide your response in the following JSON format:
{{
    "explanation": "Brief explanation of how you'll solve this",
    "sql_query": "The SQL query to run",
    "visualization": "none/bar/line/pie/scatter",
    "visualization_explanation": "Why this visualization type is appropriate (or why none is needed)",
    "x_axis": "Column name for x-axis if visualization needed",
    "y_axis": "Column name for y-axis if visualization needed",
    "title": "Suggested title for the visualization",
    "color": "Column name for color differentiation if applicable (optional)"
}}
Only include a visualization if it would meaningfully enhance understanding of the data.
Be careful to use only tables and columns that exist in the schema.
"""
    
    try:
        # Create standard OpenAI client
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
         
        response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a SQL and data analysis expert"},
                    {"role": "user", "content": agent_prompt}
                ],
                temperature=0.001
            )
         
        output_text = response.choices[0].message.content.strip()
    
        # Look for JSON pattern
        json_match = re.search(r'({[\s\S]*})', output_text)
        if json_match:
            content = json_match.group(1)
    
        result = json.loads(content)
        return result, None
    except Exception as e:
        return None, f"Error with OpenAI API: {str(e)}"



# Function to get query from OpenAI
# def get_sql_query_from_llm(question, schema_info):
#     client = OpenAI(api_key=st.session_state.openai_api_key)
    
#     prompt = f"""You are an expert in SQL and data analysis. Based on the database schema below, generate a SQL query to answer the user's question.
    
# DATABASE SCHEMA:
# {json.dumps(schema_info, indent=2)}

# USER QUESTION: {question}

# Provide your response in the following JSON format:
# {{
#     "explanation": "Brief explanation of how you'll solve this",
#     "sql_query": "The SQL query to run",
#     "visualization": "none/bar/line/pie/scatter",
#     "visualization_explanation": "Why this visualization type is appropriate (or why none is needed)",
#     "x_axis": "Column name for x-axis if visualization needed",
#     "y_axis": "Column name for y-axis if visualization needed",
#     "title": "Suggested title for the visualization",
#     "color": "Column name for color differentiation if applicable (optional)"
# }}

# Only include a visualization if it would meaningfully enhance understanding of the data.
# Be careful to use only tables and columns that exist in the schema.
# """

#     try:
#         response = client.chat.completions.create(
#             model="gpt-4-turbo-preview",  # Or use gpt-3.5-turbo if preferred
#             messages=[
#                 {"role": "system", "content": "You generate SQL queries from natural language questions and determine appropriate visualizations."},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.1
#         )
        
#         # Extract JSON from response
#         content = response.choices[0].message.content
#         # Look for JSON pattern
#         json_match = re.search(r'({[\s\S]*})', content)
#         if json_match:
#             content = json_match.group(1)
        
#         result = json.loads(content)
#         return result, None
#     except Exception as e:
#         return None, f"Error with OpenAI API: {str(e)}"

# Function to create visualization based on dataframe and visualization type
def create_visualization(df, viz_info):
    viz_type = viz_info.get("visualization", "none")
    
    if viz_type == "none" or df.empty:
        return None
    
    x_axis = viz_info.get("x_axis")
    y_axis = viz_info.get("y_axis")
    title = viz_info.get("title", "Data Visualization")
    color = viz_info.get("color")
    
    # Check if the specified columns exist in the dataframe
    if x_axis and x_axis not in df.columns:
        return {"error": f"Column '{x_axis}' not found in query results"}
    if y_axis and y_axis not in df.columns:
        return {"error": f"Column '{y_axis}' not found in query results"}
    if color and color not in df.columns:
        color = None  # Make color optional
    
    try:
        if viz_type == "bar":
            if color:
                fig = px.bar(df, x=x_axis, y=y_axis, color=color, title=title)
            else:
                fig = px.bar(df, x=x_axis, y=y_axis, title=title)
            return fig
        
        elif viz_type == "line":
            if color:
                fig = px.line(df, x=x_axis, y=y_axis, color=color, title=title)
            else:
                fig = px.line(df, x=x_axis, y=y_axis, title=title)
            return fig
        
        elif viz_type == "pie":
            fig = px.pie(df, names=x_axis, values=y_axis, title=title)
            return fig
        
        elif viz_type == "scatter":
            if color:
                fig = px.scatter(df, x=x_axis, y=y_axis, color=color, title=title)
            else:
                fig = px.scatter(df, x=x_axis, y=y_axis, title=title)
            return fig
        
        else:
            return {"error": f"Unsupported visualization type: {viz_type}"}
    
    except Exception as e:
        return {"error": f"Error creating visualization: {str(e)}"}

# App title
st.title("🔍 SQL Database Explorer with AI")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Database selection
    db_path = st.text_input("Database Path (SQLite file)", "database.db")
    
    # OpenAI API Key
    # openai_api_key = st.text_input("OpenAI API Key", type="password")
    # st.session_state.openai_api_key = openai_api_key
    
    # Database connection
    if st.button("Connect to Database"):
        if os.path.exists(db_path):
            try:
                # Connect to the database
                conn = get_connection(db_path)
                st.session_state.conn = conn
                
                # Get database schema
                schema = get_db_schema(conn)
                st.session_state.db_schema = schema
                
                st.success("Successfully connected to the database!")
                
                # Display schema info
                st.subheader("Database Schema")
                for table, columns in schema.items():
                    if not table.endswith("_sample"):
                        st.write(f"**Table: {table}**")
                        for col in columns:
                            primary_key = "🔑 " if col["primary_key"] else ""
                            st.write(f"- {primary_key}{col['name']} ({col['type']})")
            except Exception as e:
                st.error(f"Error connecting to database: {str(e)}")
        else:
            st.error(f"Database file not found: {db_path}")

# Main content
st.header("Ask Questions About Your Data")

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display query results if available
        if "dataframe" in message:
            st.dataframe(message["dataframe"], use_container_width=True)
        
        # Display visualization if available
        if "visualization" in message and message["visualization"] is not None:
            if isinstance(message["visualization"], dict) and "error" in message["visualization"]:
                st.error(message["visualization"]["error"])
            else:
                st.plotly_chart(message["visualization"], use_container_width=True)

# Chat input
if prompt := st.chat_input("Ask a question about your data..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Check database connection
    if "conn" not in st.session_state or "db_schema" not in st.session_state:
        with st.chat_message("assistant"):
            st.markdown("Please connect to a database first using the sidebar.")
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Please connect to a database first using the sidebar."
        })
    # Check OpenAI API key
    # elif not st.session_state.openai_api_key:
    #     with st.chat_message("assistant"):
    #         st.markdown("Please enter your OpenAI API key in the sidebar.")
    #     st.session_state.messages.append({
    #         "role": "assistant", 
    #         "content": "Please enter your OpenAI API key in the sidebar."
    #     })
    else:
        # Get SQL query from OpenAI
        with st.spinner("Analyzing your question..."):
            llm_response, llm_error = get_openai_response(prompt, st.session_state.db_schema)
        
        if llm_error:
            # Handle LLM error
            with st.chat_message("assistant"):
                st.markdown(f"Error generating SQL query: {llm_error}")
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"Error generating SQL query: {llm_error}"
            })
        else:
            # Execute the generated SQL query
            with st.spinner("Executing SQL query..."):
                df, query_error = execute_query(st.session_state.conn, llm_response["sql_query"])
            
            if query_error:
                # Handle query execution error
                response_content = f"""
                I tried to generate a SQL query based on your question, but encountered an error when executing it.
                
                **My approach:**
                {llm_response["explanation"]}
                
                **Generated SQL query:**
                ```sql
                {llm_response["sql_query"]}
                ```
                
                **Error:**
                ```
                {query_error}
                ```
                
                Please try rephrasing your question or specifying more details.
                """
                with st.chat_message("assistant"):
                    st.markdown(response_content)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_content
                })
            else:
                # Generate visualization if needed
                visualization = None
                if llm_response["visualization"] != "none":
                    visualization = create_visualization(df, llm_response)
                
                # Prepare response content
                response_content = f"""
{llm_response["explanation"]}

**SQL Query:**
```sql
{llm_response["sql_query"]}
```

**Results:**
                """
                
                # Add visualization explanation if applicable
                if llm_response["visualization"] != "none":
                    response_content += f"\n\n**Visualization:** {llm_response['visualization_explanation']}"
                
                # Display the response
                with st.chat_message("assistant"):
                    st.markdown(response_content)
                    
                    if not df.empty:
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("The query returned no results.")
                    
                    if visualization is not None:
                        if isinstance(visualization, dict) and "error" in visualization:
                            st.error(visualization["error"])
                        else:
                            st.plotly_chart(visualization, use_container_width=True)
                
                # Add to chat history
                message_data = {
                    "role": "assistant",
                    "content": response_content,
                    "dataframe": df
                }
                
                if visualization is not None:
                    message_data["visualization"] = visualization
                
                st.session_state.messages.append(message_data)

# Display a welcome message when the app is first loaded
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("""
        👋 Welcome to the SQL Database Explorer!
        
        To get started:
        1. Enter your SQLite database path in the sidebar
        2. Enter your OpenAI API key
        3. Click "Connect to Database"
        4. Ask questions about your data in natural language
        
        Example questions:
        - "Show me the top 5 products by sales"
        - "What was the total revenue by month last year?"
        - "Which customers made the most purchases?"
        - "Compare sales performance across regions"
        
        I'll analyze your question, generate an appropriate SQL query, and visualize the results when helpful.
        """)



