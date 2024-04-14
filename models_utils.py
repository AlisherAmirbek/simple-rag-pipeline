from langchain.prompts import PromptTemplate
from groq import Groq
from langchain_groq import ChatGroq
from langchain_community.llms import LlamaCpp


def initialize_models(model_path):

    chat_model = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=-1,
        n_batch=32,
        verbose=True,
        n_ctx = 8192,
    )
    '''chat_model = ChatGroq(temperature=0,
                        model_name="mixtral-8x7b-32768",
                        api_key="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")'''
    print("LanguageModel initialized")

    prompt_template = """<s>[INST] Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Write as concise and clear as possible.
    Context: {context}
    Question: {question}
    [/INST]"""
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=['context', 'question'])

    return chat_model, prompt