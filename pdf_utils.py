from llama_parse import LlamaParse
import os
import re
import joblib
from langchain_community.document_loaders import PyPDFLoader


def clean_text(text):
    cleaned_text = re.sub(r'\n\s*\n', '\n', text)
    cleaned_text = cleaned_text.strip()
    return cleaned_text


def load_or_parse_data():
    '''data_file = "data/parsed_data.pkl"

    if os.path.exists(data_file):
        parsed_data = joblib.load(data_file)
    else:
        parser = LlamaParse(api_key="llx-3VvtNvZm6GPm9B5udedQYzSE0prIiX3gDWGc6M2MRTgCa5UR",
                            result_type="markdown")
        llama_parse_documents = parser.load_data("data/Long Term Maintenance Report.pdf")


        print("Saving the parse results in .pkl format ..........")
        joblib.dump(llama_parse_documents, data_file)

        parsed_data = llama_parse_documents'''

    data_file = "data/"
    documents = []
    for file in os.listdir(data_file):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(data_file, file)
            loader = PyPDFLoader(pdf_path)
            loaded_docs = loader.load()
            
            for doc in loaded_docs:
                if hasattr(doc, 'page_content'):
                    doc.page_content = clean_text(doc.page_content)
            documents.extend(loaded_docs)

    return documents