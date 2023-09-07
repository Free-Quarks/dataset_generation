# Check out: https://huggingface.co/blog/starcoder for another model

#import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import (
    StructuredOutputParser,
    ResponseSchema
)

from langchain import PromptTemplate, HuggingFaceHub, LLMChain


#------------HUGGINGFACE STUFF------------------------------------

h = open('huggingface_key.txt', 'r')

huggingface_key = h.read()

h.close()

# initialize HF LLM
flan_t5 = HuggingFaceHub(
    repo_id="google/flan-t5-xxl",
    model_kwargs={"temperature":0.5, "max_length": 64},
    huggingfacehub_api_token=huggingface_key
)

# build prompt template for simple question-answering
template = """Question: {question}

Answer: """
prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(
    prompt=prompt,
    llm=flan_t5
)

question = "Which NFL team won the Super Bowl in the 2010 season?"

print(llm_chain.run(question))


#------------------------------------------------------------------
'''
f = open('openai_key.txt', 'r')

openai_key = f.read()

f.close()

# This parsing struggles if trying to find line numbers, it will often return the lines in their entirety
# however, it seems to work well for returning the name of the function that can be used to define the dynamics
response_schemas = [
    ResponseSchema(name="code", description="the entire code"),
    ResponseSchema(name="function_name", description="The name of the function implementating the model")
]

# for structured output parsing
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# for structured output parsing
format_instructions = output_parser.get_format_instructions()

temperature = 0.8

# initialize the models
openai = ChatOpenAI(
    temperature=temperature,
    model_name='gpt-3.5-turbo',
    openai_api_key=openai_key
)

programmer_type_list = ["college student", "software engineer"]
language_list = ["python", "fortran"]
model_list = ["SIR", "SEIR", "SERID", "SIDARTHE"] # could probably make up things and it would make diff eq's for them
method_list = ["Euler", "odeint", "RK2", "RK3", "RK4"]
# if plannign to make a ML model to predict the line numbers for labels, 
# adding a field for data augmentation from lines numbers could be useful too

template="You are a {programmer_type} that writes {language} code to simulate and plot epidemiology compartimental models."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template="{model} using {method} \n{format_instructions}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

formatted_prompt = chat_prompt.format_prompt(programmer_type="college student", language="python", model="SIDARTHE", method="RK3", format_instructions = format_instructions).to_messages()

output = openai(formatted_prompt)

parsed_output = output_parser.parse(output.content)

print(parsed_output['code'])
print("\n")
print(parsed_output['function_name'])

with open('output.py', 'w') as f:
    print(parsed_output['code'], file=f)
f.close()

with open('output.txt', 'w') as f:
    print(parsed_output['function_name'], file=f)
f.close()
'''
