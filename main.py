import streamlit as st
import fitz  # PyMuPDF for PDF text extraction
import torch
from torch.nn.functional import cosine_similarity
from transformers import BertTokenizer,BertModel
import tempfile
import os


# Load BertTokenizer & BertModel:
bert_model=BertModel.from_pretrained("bert-base-uncased")
tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")

# Extract text from pdf:
def extract_text_from_pdf(pdf_path):
    doc=fitz.open(pdf_path)
    text="\n".join(page.get_text() for page in doc)
    return text


# Encoding Text using BERT Embeddings:
def encode_text(text):
    inputs=tokenizer(text,return_tensors="pt",truncation=True,padding=True,max_length=512)
    with torch.no_grad():
        outputs=bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)


def match_resume_with_jobdescr(resume_emb,job_emb):
    similarity=cosine_similarity(resume_emb,job_emb)
    return similarity.item()


# Streamlit UI:
st.title(" 📄 AI Powered Resume Analyzer Using BERT - Transformer")

uploaded_file=st.file_uploader("📤 Upload Your Resume in PDF", type=["pdf"])
job_description=st.text_area("📝Enter the Job Description", value="")

if uploaded_file and job_description:
    with st.spinner("Processing"):
        temp_dir=tempfile.gettempdir()
        temp_path=os.path.join(temp_dir,uploaded_file.name)
        
        with open(temp_path,"wb") as temp_file:
            temp_file.write(uploaded_file.read())
            
        try:
            resume_text=extract_text_from_pdf(temp_path)
            
            resume_embedding=encode_text(resume_text)
            job_embedding=encode_text(job_description)
            
            similarity_score=match_resume_with_jobdescr(resume_embedding,job_embedding)
            
            
            # Display Results:
            
            st.subheader("Resume Analysis Results")
            st.write("Job Description:",job_description)
            st.write("Resume Text(First 1000 charcters):")
            st.text(resume_text)
            st.metric("Matching score:",round(similarity_score*100,2),"%")
            
        finally:
            os.remove(temp_path)
            
