from pdf_utils import load_or_parse_data
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
import time
from langchain_community.embeddings import InfinityEmbeddingsLocal


def create_vector_database():

    llama_parse_documents = load_or_parse_data()

    with open('data/output.md', 'a') as f:  # Open the file in append mode ('a')
        for document in llama_parse_documents:
            f.write(document.page_content + '\n')

    markdown_path = "data/output.md"
    loader = UnstructuredMarkdownLoader(markdown_path)

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    print(f"length of documents loaded: {len(documents)}")
    print(f"total number of document chunks generated :{len(docs)}")

    embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    print("Vector DB initialized")
    start_time = time.time()
    vs = Chroma.from_documents(
        documents=docs,
        embedding=embed_model,
        persist_directory="chroma_db",
        collection_name="maintenance_report"
    )
    retriever=vs.as_retriever(search_kwargs={'k': 5})
    end_time = time.time()
    print(f"Time taken to initialize the vector database: {end_time - start_time} seconds")

    print('Vector DB created successfully !')
    return retriever,embed_model