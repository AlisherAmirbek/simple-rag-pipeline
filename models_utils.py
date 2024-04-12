from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain import hub


def initialize_models(model_path):
    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=2048,
        verbose=False,
    )
    print("LanguageModel initialized")

    embedding_function = HuggingFaceEmbeddings(model_name='mixedbread-ai/mxbai-embed-large-v1')
    print("Embedding function initialized")

    prompt = hub.pull("rlm/rag-prompt", api_url="https://api.hub.langchain.com")
    print("Prompt initialized")

    return llm, embedding_function, prompt