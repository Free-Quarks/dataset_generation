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

f = open('openai_key.txt', 'r')

openai_key = f.read()

f.close()

# This parsing struggles if trying to find line numbers, it will often return the lines in their entirety
# however, it seems to work well for returning the name of the function that can be used to define the dynamics
response_schemas = [
    ResponseSchema(name="code", description="the code simulating the model"),
    ResponseSchema(name="function_name", description="The name of the function implementating the model dynamics")
]

# for structured output parsing
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# for structured output parsing
format_instructions = output_parser.get_format_instructions()

temperature = 0.7

# initialize the models
openai = ChatOpenAI(
    temperature=temperature,
    model_name='gpt-3.5-turbo',
    openai_api_key=openai_key
)

template="You are a {programmer_type} that writes {language} code to simulate epidemiology compartimental models."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template="{model} using {method} \n{format_instructions}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

formatted_prompt = chat_prompt.format_prompt(programmer_type="college student", language="python", model="SIR", method="euler", format_instructions = format_instructions).to_messages()

output = openai(formatted_prompt)

parsed_output = output_parser.parse(output.content)

print(parsed_output['code'])
print("\nname:")
print(parsed_output['function_name'])