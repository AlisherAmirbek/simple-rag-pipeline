from langchain.chains import RetrievalQA
from utils.vectorstore_utils import create_vector_database
from utils.models_utils import initialize_models

def setup_rag_pipeline(model_path):

    retriever, embed_model = create_vector_database()

    chat_model, prompt = initialize_models(model_path)

    qa = RetrievalQA.from_chain_type(llm=chat_model,
                               chain_type="stuff",
                               retriever=retriever,
                               return_source_documents=True,
                               chain_type_kwargs={"prompt": prompt})

    return qa

def ask_question(qa_chain, query):
    result = qa_chain.invoke({"query": query})
    return result['result']