from utils.langchain_utils import setup_rag_pipeline, ask_question
from utils.download_utils import download_file
import os

def main():
    
    repo_id = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    file_name = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"

    model_path = download_file(repo_id, file_name)

    qa_chain = setup_rag_pipeline(model_path)
    print("Chain initialized")

    while True:
        query = input("Ask a question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        
        result = ask_question(qa_chain, query)
        print(result)

if __name__ == '__main__':
    main()