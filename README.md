# Simple Retrieval Augmented Generation (RAG) Pipeline

### Components
- **ChromaDB**: Used as vector storage for efficient context retrieval.
- **Mistral 7B**: LLM used to generate text based on the context provided by ChromaDB.
- **BGE-Small-EN-v1.5**: Embeddings model

## Setup instructions

### Preliminary
- Python 3.8 or higher
- pip
- Access to compute resources capable of running deep learning models (preferably with CUDA support for GPU acceleration).
  
### Installation
1. **Clone the repository**
      ```bash
      git clone https://github.com/AlisherAmirbek/simple-rag-pipeline.git
      CD simple-rag-pipeline
      ```
2. **Activate the virtual environment in python**
2. **Install Python dependencies**
      ```bash
      pip install -r requirements.txt
      ```
3. **Set environment variables** (GPU only)
      ```bash
      CMAKE_ARGS="-DLLAMA_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir --verbose
      ```
      
## Usage

To run the RAG pipeline, go to the project root directory and run:
```bash
python main.py
