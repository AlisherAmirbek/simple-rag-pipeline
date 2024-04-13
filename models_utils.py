from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain import hub



def initialize_models(model_path):
    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=4096,
        verbose=False,
    )
    print("LanguageModel initialized")

    embedding_function = HuggingFaceEmbeddings(model_name='mixedbread-ai/mxbai-embed-large-v1')
    print("Embedding function initialized")

    prompt_template = """Instruct: Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    {context}
    Question: {question}
    Output:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    print("Prompt initialized: ", PROMPT)

    return llm, embedding_function, PROMPT