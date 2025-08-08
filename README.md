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

```
PDF analytics for non-technical users
outputs:
[![Invisora Demo](assets/one.png)](assets/<img width="1411" height="675" alt="one" src="https://github.com/user-attachments/assets/b50e7cc1-cb5b-4faf-ba5c-272b248b7b6c" />
)
[![Invisora Demo](assets/two.png)](assets/<img width="1408" height="682" alt="two" src="https://github.com/user-attachments/assets/37983f3e-b4f4-4542-bb21-dd8244ceb700" />
)
[![Invisora Demo](assets/three.png)](assets/<img width="1346" height="632" alt="three" src="https://github.com/user-attachments/assets/517691f0-e5c8-446b-b42a-f196a5a98afc" />
)
[![Invisora Demo](assets/four.png)](assets/<img width="1394" height="711" alt="four" src="https://github.com/user-attachments/assets/b8a6f27d-01de-4bb4-831c-f04dca5b31ad" />
)
[![Invisora Demo](assets/five.png)](assets/<img width="1404" height="800" alt="five" src="https://github.com/user-attachments/assets/a4e78ec8-e633-44c7-8149-2929e13764e3" />
)
[![Invisora Demo](assets/six.png)](assets/<img width="1416" height="770" alt="six" src="https://github.com/user-attachments/assets/102766ec-83ab-4237-afd1-b474e63a62d0" />
)



 Author
Suranjay Kumar
B.Tech Computer Engineering | Marwadi University
✉️ suranjaykumar.119084@marwadiuniversity.ac.in


Contributions Welcome
If you’d like to suggest improvements or report issues, feel free to open a pull request or issue.







