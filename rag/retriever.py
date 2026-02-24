# rag/retriever.py

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from rag.prompts import explanation_prompt
import subprocess
from langchain_huggingface import HuggingFaceEmbeddings


def call_ollama(prompt, model="llama3"):
    process = subprocess.Popen(
        ["ollama", "run", model],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )
    output, _ = process.communicate(prompt)
    return output


def explain_with_rag(diagnosis):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        "rag/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    query = ", ".join([f[0] for f in diagnosis["ranked_root_causes"]])
    docs = vectorstore.similarity_search(query, k=3)

    context = "\n".join([d.page_content for d in docs])
    prompt = explanation_prompt(context, diagnosis)

    return call_ollama(prompt)