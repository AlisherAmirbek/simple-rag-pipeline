from langchain_community.vectorstores import Chroma

def initialize_vectorstore(chunked_documents, embedding_function):
    vectorstore = Chroma.from_documents(
        documents=chunked_documents, 
        embedding=embedding_function,
        persist_directory="chroma_store/")
        
    vectorstore.persist()
    retriever = vectorstore.as_retriever()
    print("VectorStore initialized")

    return retriever