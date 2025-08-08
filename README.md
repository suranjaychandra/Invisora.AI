# Invisora.AI-

STEP 1:

# 🚀 Invisora.AI – GenAI-Powered Data Insights & Reporting Tool

**Invisora.AI** is an AI-powered application that allows users to upload PDF reports, extract and understand key insights, generate interactive visualizations, and export summarized reports — all within a few clicks.

It combines the power of **Retrieval-Augmented Generation (RAG)**, **LLMs (Gemini 4.0)**, **Streamlit UI**, and **Plotly** for an intelligent and user-friendly data reporting solution.

---

## 🧠 Key Features

✅ **Streamlit Web Interface**  
Upload PDFs and interact with results using an intuitive UI.

✅ **Smart PDF Parsing**  
PDF content is extracted and parsed using **PyMuPDF**, optimized for structured data extraction.

✅ **Retrieval-Augmented Generation (RAG)**  
Relevant chunks of data are retrieved and sent to **Gemini 4.0** using **RetrievalQA** for precise summaries.

✅ **Interactive Charts**  
Data is converted into clear, interactive charts using **Plotly**.

✅ **AI-Generated Summary**  
LLM-generated summaries of the uploaded content — concise and insightful.

✅ **PDF Report Export**  
Summarized content and visuals are exported into a downloadable **PDF** using **PDFKit**.

---

## 📁 Project Structure

Invisora.AI/
│
├── app.py # Main Streamlit app
├── pdf_parser.py # PDF extraction using PyMuPDF
├── rag_engine.py # RAG pipeline with FAISS & RetrievalQA
├── visualizer.py # Chart generation with Plotly
├── report_generator.py # Generate and export PDF with PDFKit
├── requirements.txt # All Python dependencies
└── README.md # Project documentation




## 🛠️ Tech Stack

| Tech / Tool        | Purpose                                   |
|--------------------|--------------------------------------------|
| **Streamlit**      | UI and user interaction                    |
| **PyMuPDF**         | PDF text and data extraction               |
| **LangChain**       | For RAG pipeline and LLM orchestration     |
| **Gemini 4.0 API**  | Large Language Model for summarization     |
| **Plotly**          | Generate interactive visualizations        |
| **PDFKit**          | Convert HTML output to downloadable PDF    |
| **FAISS**           | Vector store for chunk similarity search   |

---

## ⚙️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Invisora.AI.git
cd Invisora.AI


STEP 2 : Create a virtual environment & activate it

python -m venv .venv
source .venv/bin/activate  # For Windows: .venv\Scripts\activate


STEP 3 :
 Install dependencies
 pip install -r requirements.txt



STEP 4:

Run the Streamlit app
streamlit run app.py



🧪 How It Works
Upload a PDF via the web UI

The PDF is parsed and broken into relevant text chunks

Chunks are embedded and passed through a FAISS vector store

Gemini 4.0 is queried using RetrievalQA to generate summaries

Plotly creates graphs from structured data

Summary + charts are converted into a downloadable PDF report

 Sample Output
AI-generated charts and summaries based on a real uploaded PDF

![AI-generated chart preview](assets/sample_chart.png)


 Use Cases
Business report summarization

Market research insights

Automated data storytelling

Academic paper analysis

PDF analytics for non-technical users
outputs:
you can see here : ![Invisora Demo](assets/one.png)
![Invisora Demo](assets/two.png)
![Invisora Demo](assets/three.png)
![Invisora Demo](assets/four.png)
![Invisora Demo](assets/five.png)
![Invisora Demo](assets/chart_0.png)
![Invisora Demo](assets/chart_1.png)
![Invisora Demo](assets/chart_2.png)
![Invisora Demo](assets/chart_3.png)
![Invisora Demo](assets/chart_4.png)


 Author
Suranjay Kumar
B.Tech Computer Engineering | Marwadi University
✉️ suranjaykumar.119084@marwadiuniversity.ac.in


Contributions Welcome
If you’d like to suggest improvements or report issues, feel free to open a pull request or issue.







