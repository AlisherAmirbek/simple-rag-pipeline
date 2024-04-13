from langchain.chains import RetrievalQA
from vectorstore_utils import create_vector_database
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate

def setup_rag_pipeline(model_path):

    vs, embed_model = create_vector_database()

    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=8196,
        verbose=False,
    )

    vectorstore = Chroma(embedding_function=embed_model,
                        persist_directory="chroma_db_llamaparse1",
                        collection_name="rag")

    retriever=vectorstore.as_retriever(search_kwargs={'k': 3})

    prompt_template = """<s>[INST] Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    {context}
    Question: {question}
    [/INST]"""
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])

    qa = RetrievalQA.from_chain_type(llm=chat_model,
                               chain_type="stuff",
                               retriever=retriever,
                               return_source_documents=True,
                               chain_type_kwargs={"prompt": prompt})

    return qa

def ask_question(qa_chain, query):
    result = qa_chain.invoke({"query": query})
    return result['result']