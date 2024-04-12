from langchain.chains import RetrievalQA
from pdf_utils import process_pdf
from models_utils import initialize_models
from vectorstore_utils import initialize_vectorstore

def setup_rag_pipeline(model_path):
    chunked_documents = process_pdf()
    print("Documents loaded")

    llm, embedding_function, prompt = initialize_models(model_path) 

    retriever = initialize_vectorstore(chunked_documents, embedding_function)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, chain_type_kwargs={"prompt": prompt}
    )
    print("Qa_chain initialized")

    return qa_chain

def ask_question(qa_chain, query):
    result = qa_chain.run({"query": query})
    return result