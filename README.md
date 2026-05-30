# 🔍 AI-Powered Person of Interest Retrieval using Vision-Language Models

## 📌 Overview

This project implements a **multimodal retrieval system** capable of identifying and retrieving a *person of interest* using both **text queries** and **image inputs**.

It leverages **Vision-Language Models (VLMs)** such as CLIP/SigLIP to generate embeddings for images and text, enabling efficient and accurate similarity-based search.

---

## 🚀 Key Features

* 🔗 **Multimodal Search** — Query using text or images
* 🧠 **Embedding-Based Retrieval** — Uses VLM embeddings
* ⚡ **Efficient Similarity Matching** — Fast vector-based search
* 📊 **Re-ranking System** — Improves retrieval accuracy
* 🧩 **Modular Architecture** — Easy to extend and maintain

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/ShamithaJain/Person-of-Interest-Search-using-Vision-Language-Models.git
cd Person-of-Interest-Search-using-Vision-Language-Models
```

---

### 2. Install dependencies (using uv)

```bash
uv sync
```

> If you don’t have uv installed:

```bash
pip install uv
```

---

## ▶️ Running the Project

### Step 1: Build the index

```bash
python scripts/build_index.py
```

---

### Step 2: Launch the Streamlit app

```bash
streamlit run src\person_of_interest\app.py
```

---

## 🧠 How It Works

1. **Input Handling**

   * Accepts text queries or image inputs

2. **Embedding Generation**

   * Converts input into vector representations using VLMs

3. **Similarity Search**

   * Matches query embeddings with indexed dataset

4. **Re-ranking**

   * Improves relevance of retrieved results

5. **Output**

   * Returns top matches for the person of interest

---

## 📊 Example Use Cases

* 🔍 Surveillance and security systems
* 🧑‍💼 Identity-based retrieval
* 📸 Image search engines
* 🧠 Multimodal AI research

---

## 🧠 Tech Stack

* Python
* Vision-Language Models (CLIP / SigLIP)
* Streamlit
* NumPy / PyTorch

---

## ⚠️ Important Notes

* Run all commands from the **project root directory**
* Ensure dataset is available before building index
* Avoid committing `.venv`, `__pycache__`, or large files

---

## 🔮 Future Improvements

* 🔍 Integrate vector databases (FAISS / ChromaDB)
* 🌐 Build API with FastAPI
* ⚡ Optimize large-scale retrieval
* 🧠 Fine-tune models for domain-specific use

---

## 👩‍💻 Author

**Shamitha Jain**

---

## 📬 Feedback

Feel free to open issues or contribute improvements!

---
