# RAGchatbot


Here’s a well-structured README for your project:  

---

# **RAG Document Q&A with GROQ and Llama/Gemma**  

This project is a **Retrieval-Augmented Generation (RAG) chatbot** that enables users to query research papers using **GROQ's Llama model**. It leverages **LangChain**, **FAISS for vector storage**, and **Ollama embeddings** to retrieve and process relevant document information.  

---

## **Features**  
- **Upload & Process PDFs**: Extracts text from research papers stored in the `research_papers` directory.  
- **Chunking & Embedding**: Uses **RecursiveCharacterTextSplitter** to split documents and FAISS for efficient retrieval.  
- **GROQ-Powered LLM**: Uses **Llama 3 (3B)** via **LangChain's ChatGroq API** to answer queries based on retrieved documents.  
- **Document Similarity Search**: Displays the most relevant document excerpts supporting the response.  

---

## **Tech Stack & Dependencies**  

- **Python**  
- **Streamlit** (For UI)  
- **LangChain** (For LLM and retrieval)  
- **FAISS** (For vector storage)  
- **Ollama Embeddings** (For text embeddings)  
- **PyPDFLoader** (For extracting text from PDFs)  
- **Dotenv** (For managing API keys)  

---

## **Installation & Setup**  

### **1. Clone the repository**  
```sh
git clone https://github.com/yourusername/RAGchatbot.git
cd RAGchatbot
```

### **2. Create a virtual environment (Optional but recommended)**  
```sh
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
```

### **3. Install dependencies**  
```sh
pip install -r requirements.txt
```

### **4. Set up environment variables**  
Create a `.env` file in the project directory and add your **GROQ API key**:  
```
GROQ_API_KEY=your_api_key_here
```

### **5. Run the application**  
```sh
streamlit run app.py
```

---

## **Usage**  

1. Place your research papers (PDFs) in the `research_papers/` directory.  
2. Start the Streamlit app using the command above.  
3. Click **"Document Embeddings"** to generate vector embeddings.  
4. Enter your query related to the research papers in the text box.  
5. View the Llama-powered response along with relevant document excerpts.  

---

## **Future Enhancements**  
- Support for **multiple embedding models** (e.g., HuggingFace embeddings).  
- Improved **document chunking** for better retrieval performance.  
- UI enhancements for a **better user experience**.  
- **File upload feature** to dynamically process PDFs.  

---

## **Contributing**  
Pull requests are welcome! Feel free to improve functionality or optimize performance.  

---

## **License**  
This project is licensed under the **MIT License**.  

---
