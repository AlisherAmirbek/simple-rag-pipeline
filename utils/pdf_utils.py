from langchain_community.document_loaders import PyPDFLoader
import os
import re

def clean_text(text):
    cleaned_text = re.sub(r'\n\s*\n', '\n', text)
    cleaned_text = cleaned_text.strip()
    return cleaned_text


def load_or_parse_data():

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