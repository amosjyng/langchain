from langchain.chains.multiple_outputs import GetMultipleOutputsChain
from langchain.llms.openai import OpenAI


llm = OpenAI(temperature=0, model_name="text-davinci-002")
print(GetMultipleOutputsChain(
    llm=llm,
    prefix="Generate a fictional person for me. Make sure to put all attributes in quotes.\n\n",
    variables={
        "First Name": "first_name",
        "Last Name": "last_name",
        "Date of birth": "dob",
        "Gender": "gender",
        "Address": "address",
    }
)({}))
