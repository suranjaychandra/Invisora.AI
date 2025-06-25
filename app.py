# --- Imports ---
from appwrite.client import Client
from appwrite.services.storage import Storage
from appwrite.services.account import Account
import base64
import textwrap
import plotly.io as pio
import streamlit as st
import pandas as pd
import plotly.express as px
from openai import AzureOpenAI
import pdfkit
import time
from dotenv import load_dotenv
import os

# --- Environment Setup ---
load_dotenv()
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_BASE = os.getenv("AZURE_OPENAI_API_BASE")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_API_BASE
)

PDFKIT_CONFIG = pdfkit.configuration(wkhtmltopdf=r"/usr/local/bin/wkhtmltopdf")

# --- Utility Functions ---
def load_uploaded_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
        else:
            df = pd.read_excel(uploaded_file)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def get_column_types(df):
    return {col: str(df[col].dtype) for col in df.columns}

def get_chart_suggestions(column_types):
    prompt = f"""
    Given these columns and their data types:
    {column_types}
    Suggest 5 suitable visualization types (Pie, Bar, Line, Scatter, Histogram).
    Only return the chart names as a comma-separated list.
    """
    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful data analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=100
        )
        ai_text = response.choices[0].message.content
        viz_types = [v.strip().lower() for v in ai_text.split(",") if v.strip().lower() in ["pie", "bar", "line", "scatter", "histogram"]]
        if len(viz_types) < 5:
            viz_types = ["bar", "line", "pie", "scatter", "histogram"][:5]
        return viz_types
    except Exception as e:
        st.error(f"AI Error: {e}")
        return []

def generate_charts(df, viz_types):
    charts = []
    for viz in viz_types:
        st.subheader(f"📌 {viz.capitalize()} Chart")
        fig = None
        if viz == "pie":
            column = st.selectbox("Select column for Pie Chart", df.columns, key="pie")
            fig = px.pie(df, names=column, title=f"Distribution of {column}")
        elif viz == "bar":
            x_col = st.selectbox("X-axis for Bar Chart", df.columns, key="bar_x")
            y_col = st.selectbox("Y-axis for Bar Chart", df.columns, key="bar_y")
            fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
        elif viz == "line":
            x_col = st.selectbox("X-axis for Line Chart", df.columns, key="line_x")
            y_col = st.selectbox("Y-axis for Line Chart", df.columns, key="line_y")
            fig = px.line(df, x=x_col, y=y_col, title=f"Trend of {y_col} over {x_col}")
        elif viz == "scatter":
            x_col = st.selectbox("X-axis for Scatter Plot", df.columns, key="scatter_x")
            y_col = st.selectbox("Y-axis for Scatter Plot", df.columns, key="scatter_y")
            fig = px.scatter(df, x=x_col, y=y_col, title=f"Scatter Plot of {y_col} vs {x_col}")
        elif viz == "histogram":
            column = st.selectbox("Select column for Histogram", df.columns, key="hist")
            fig = px.histogram(df, x=column, title=f"Distribution of {column}")
        if fig:
            charts.append((fig, viz))
            st.plotly_chart(fig, use_container_width=True)
    return charts

def generate_ai_summary(df):
    summary_prompt = f"Given this dataset with columns: {', '.join(df.columns)}, provide a short summary of insights, trends, and possible business improvements."
    try:
        summary_response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful data analyst."},
                {"role": "user", "content": summary_prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        return summary_response.choices[0].message.content
    except Exception as e:
        return f"No insights generated. Error: {e}"

def generate_pdf_report(summary_text, charts):
    chart_images = []
    summary_html = f"""
    <h1 style=\"text-align:center; color:#2C3E50;\"> Invisora.AI</h1>
    <h2 style=\"color:#1F618D;\"> Key Insights</h2>
    <p style=\"font-size:16px; line-height:1.6; color:#283747;\">{summary_text}</p>
    <hr>
    <h2 style=\"color:#1F618D;\"> Data Visualizations</h2>
    <p>Charts below are static snapshots. For interactive version, visit: <a href='https://yourapp.streamlit.app'>Open Invisora.AI App</a></p>
    """
    for i, (fig, viz_name) in enumerate(charts):
        chart_path = f"chart_{i}.png"
        pio.write_image(fig, chart_path)
        chart_images.append(chart_path)
        with open(chart_path, "rb") as img_file:
            base64_img = base64.b64encode(img_file.read()).decode()
        summary_html += f"""
        <div style=\"margin-bottom:30px;\">
          <h3 style=\"color:#1F618D;\">Chart {i+1} – {viz_name.capitalize()} Chart</h3>
          <p style=\"color:#555;\">Snapshot of data visualization from your uploaded dataset.</p>
          <img src=\"data:image/png;base64,{base64_img}\" style=\"width:100%; margin-bottom:20px;\">
        </div>
        """
    pdf_path = "Invisora_Report.pdf"
    pdfkit.from_string(summary_html, pdf_path, configuration=PDFKIT_CONFIG)
    return pdf_path

def chatbot_response(df, user_query):
    sample_data = df.head(3).to_dict(orient="records")
    column_types = get_column_types(df)
    query_prompt = f"""
You are a business analyst.

Dataset info:
- Columns: {', '.join(df.columns)}
- Data Types: {column_types}
- Sample Rows: {sample_data}

User Query:
{user_query}

Answer based on this dataset only.
"""
    chat_response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful business analyst."},
            {"role": "user", "content": query_prompt}
        ],
        temperature=0.7,
        max_tokens=400
    )
    return chat_response.choices[0].message.content.strip()

# --- Streamlit App ---
st.set_page_config(page_title="Invisora.AI – AI-Powered Business Insights", layout="wide")
st.title("🤖 Invisora.AI")
st.write("Upload your business data to get interactive reports and AI-driven insights.")

uploaded_file = st.file_uploader("Upload CSV or Excel File", type=["csv", "xlsx"])

if uploaded_file:
    df = load_uploaded_file(uploaded_file)
    if df is None:
        st.stop()

    st.subheader("Select Columns to Keep")
    selected_columns = st.multiselect("Choose columns:", df.columns.tolist(), default=df.columns.tolist())
    if selected_columns:
        df = df[selected_columns]

    st.write("### Preview of Processed Data")
    st.dataframe(df.head())

    st.write("🔍 *Analyzing Data...* Please wait while AI generates insights.")
    time.sleep(2)

    column_types = get_column_types(df)
    viz_types = get_chart_suggestions(column_types)
    st.write(f"*AI Selected Visualizations:* {', '.join(viz_types)}")

    charts = generate_charts(df, viz_types)

    st.markdown("---")
    st.subheader("📝 AI-Generated Business Insights")
    st.write("Analyzing data for key takeaways...")
    summary_text = generate_ai_summary(df)
    st.write(f"*Business Insights:* {summary_text}")

    st.markdown("---")
    st.subheader("📈 Data Visualizations (5 Worksheets)")
    cols = st.columns(3)
    for i, (fig, _) in enumerate(charts[:5]):
        with cols[i % 3]:
            st.plotly_chart(fig, use_container_width=True, key=f"worksheet_{i}")

    if st.button("📅 Download Report as PDF"):
        st.write("🔄 Generating Report... Please wait.")
        pdf_path = generate_pdf_report(summary_text, charts)
        with open(pdf_path, "rb") as file:
            st.download_button("📅 Download Report", file, file_name="Invisora_Report.pdf", mime="application/pdf")

    st.markdown("---")
    st.subheader("🧐 AI Chatbot for Data Queries")
    chat_history = st.session_state.get("chat_history", [])

    with st.expander("💬 Open AI Chatbot"):
        st.write("Ask questions about your uploaded data.")
        user_query = st.text_input("Enter your query:")
        if st.button("Ask AI"):
            if user_query:
                response = chatbot_response(df, user_query)
                chat_history.append({"query": user_query, "response": response})
                st.session_state.chat_history = chat_history
        if chat_history:
            for chat in reversed(chat_history):
                st.markdown(f"**You:** {chat['query']}")
                st.markdown(f"**AI:** {chat['response']}")
