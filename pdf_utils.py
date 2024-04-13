from llama_parse import LlamaParse
import os
import joblib

def load_or_parse_data():
    data_file = "data/parsed_data.pkl"

    if os.path.exists(data_file):
        parsed_data = joblib.load(data_file)
    else:
        parser = LlamaParse(api_key="llx-3VvtNvZm6GPm9B5udedQYzSE0prIiX3gDWGc6M2MRTgCa5UR",
                            result_type="markdown")
        llama_parse_documents = parser.load_data("data/Long Term Maintenance Report.pdf")


        print("Saving the parse results in .pkl format ..........")
        joblib.dump(llama_parse_documents, data_file)

        # Set the parsed data to the variable
        parsed_data = llama_parse_documents

    return parsed_data