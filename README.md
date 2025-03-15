# 📄 AI Powered Resume Analyzer Using BERT - Transformer

This project is an AI-powered resume analyzer that uses BERT (Bidirectional Encoder Representations from Transformers) to compare a candidate's resume with a job description and provide a similarity score. The application is built using **Streamlit** for a user-friendly interface.

---

## 🚀 Features

- 📤 Upload your resume in PDF format
- 📝 Enter the job description text
- 🧠 Extracts and processes text from the resume
- 🔍 Uses BERT to generate embeddings for both resume and job description
- 📊 Computes similarity score based on **cosine similarity**
- 📈 Displays a **matching percentage** to help candidates assess their suitability

---

## 🛠️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Ananthakrishnan12/Resume-Analyzer-Using-BERT.git
cd Resume-Analyzer-Using-BERT
```

### 2️⃣ Create a virtual environment (Optional but recommended)

```bash
python -m venv CV_parser
source CV_parser/bin/activate   # On macOS/Linux
CV_parser\Scripts\activate      # On Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📌 How to Run

Run the Streamlit app:

```bash
streamlit run main.py
```

---

## 🏗️ Tech Stack

- **Python** 🐍
- **Streamlit** 🎨 (for UI)
- **PyMuPDF (fitz)** 📄 (for PDF text extraction)
- **Transformers** 🤗 (BERT-based embeddings)
- **Torch (PyTorch)** 🔥 (for embedding computations)

---

## 🔍 How It Works

1. **Resume Text Extraction**  
   - The uploaded PDF file is processed using **PyMuPDF (fitz)** to extract text.

2. **BERT Embedding Generation**  
   - The extracted resume text and job description are tokenized and converted into embeddings using **BERT**.

3. **Cosine Similarity Computation**  
   - The similarity score between resume and job description embeddings is calculated using **cosine similarity**.

4. **Result Display**  
   - The application displays the extracted resume text (preview), job description, and **matching score** as a percentage.

---

## 📷 DEMO

![Resume Analyzer UI](https://your-screenshot-url.png)  

---


