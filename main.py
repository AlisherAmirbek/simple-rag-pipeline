from langchain_utils import setup_rag_pipeline, ask_question
from download_utils import download_file
import os

def main():
    
    model = "phi-2"
    repo_id = "TheBloke/phi-2-GGUF"
    file_name = "phi-2.Q8_0.gguf"

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