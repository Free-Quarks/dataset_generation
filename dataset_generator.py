# Check out: https://huggingface.co/blog/starcoder for another model

#import openai

from langchain.llms import OpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.llms import HuggingFaceHub
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
        ChatPromptTemplate,
        PromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
        )
from langchain.output_parsers import (
        StructuredOutputParser,
        ResponseSchema
        )
from langchain.memory import ConversationBufferMemory
from langchain import ConversationChain
from tqdm import tqdm
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
import json
import os
#from langchain import PromptTemplate, HuggingFaceHub, LLMChain
'''

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

openai_key = f.readline()

f.close()

# This parsing struggles if trying to find line numbers, it will often return the lines in their entirety
# however, it seems to work well for returning the name of the function that can be used to define the dynamics
response_schemas = [
    ResponseSchema(name="code", description="Generated code"),
    ResponseSchema(name="function_name", description="The name of the function implementating the model")
]

class OutputModel(BaseModel):
    code: str = Field(description="the entire code")
    function_name: str = Field(description="The name of the function that contains the model dynamics")

# for structured output parsing
#output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
output_parser = PydanticOutputParser(pydantic_object=OutputModel)

# for structured output parsing
format_instructions = output_parser.get_format_instructions()

temperature = 0.8

# initialize the models
#openai = ChatOpenAI(
#    temperature=temperature,
#    model_name='gpt-3.5-turbo',
#    openai_api_key='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',
#)
os.environ['OPENAI_API_KEY'] = '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'
openai_api_key = os.getenv('OPENAI_API_KEY')
# initialize the models
openai = ChatOpenAI(
    temperature=temperature,
    model_name='gpt-4',
    #openai_api_key='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',
    openai_api_key= openai_api_key, #mine
)


#programmer_type_list = ["postdoc,","beginner","intermediate", "expert"]
programmer_type_list = ["expert"]
#coding_style_list = ["layperson", "verbose", "concise", "efficient","inefficient", "messy", "clean", "obfuscated", "clear"]
#coding_style_list = ["verbose", "concise"]
coding_style_list = ["verbose"]
language_list = ["python"]
#model_list = ["SIR", "SEIR", "SERID", "SIDARTHE"] # could probably make up things and it would make diff eq's for them
#method_list = ["Euler", "odeint", "RK2", "RK3", "RK4"]

method_list = ["Euler"]
model_list = ["SIR"]
#method_list = ["RK4"]
# if plannign to make a ML model to predict the line numbers for labels,
# adding a field for data augmentation from lines numbers could be useful too
'''
template="You are a {programmer_type} that writes {language} code to simulate and plot epidemiology compartimental models."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template="{model} using {method} \n{format_instructions}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)


chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

formatted_prompt = chat_prompt.format_prompt(programmer_type="student",
                                             language="python",
                                             model="SIR", method="RK4", format_instructions = format_instructions).to_messages()

output = openai(formatted_prompt)

parsed_output = output_parser.parse(output.content)

print(parsed_output['code'])
print("\n")
print(parsed_output['function_name'])

with open('dshahi-output19.py', 'w') as f:
    print(parsed_output['code'], file=f)
f.close()

with open('dshahi-output19.txt', 'w') as f:
    print(parsed_output['function_name'], file=f)
f.close()
'''

documentation_list = ["docstrings", "all"]
#from langchain.prompts.chat import Message
counter = 0
for i in tqdm(range(2)):
    for model in tqdm(model_list):
        for language in tqdm(language_list):
            for coding_style in tqdm(coding_style_list):
                for programmer_type in tqdm(programmer_type_list):
                    #for model in tqdm(model_list):
                    for method in tqdm(method_list):
                        for documentation in tqdm(documentation_list):

                            template="You are a {$programmer_type} that writes {coding_style} code in {language} programming language to simulate and plot epidemiology compartmental models that includes {documentation} documentation"
                            #template = "You write a {programmer_type} {language} programming code that writes {coding_style} code to simulate and plot epidemiology compartmental models that includes {documentation} documentation"
                            system_message_prompt = SystemMessagePromptTemplate.from_template(template)
                            #system_message_prompt = PromptTemplate.from_template(template)
                            human_template="{model} using {method} \n{format_instructions}"
                            #human_message_prompt = HumanMessagePromptTemplate.from_template(template)
                            human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
                            #prompt = PromptTemplate(
                            #        template="You are a {programmer_type} that writes {coding_style} code in {language} programming language to simulate and plot epidemiology compartmental models that includes          {documentation} documentation {model} using {method} \n{format_instructions}",
                             #       input_variables=["programmer_type","coding_style","language","model","method","documentation"],
                             #       partial_variables={"format_instructions":
                            #                            output_parser.get_format_instructions()},
                            #        )
                            #formatted_prompt = prompt.format_prompt(programmer_type=programmer_type,coding_style=coding_style, language=language,model=model, method=method,documentation=documentation)

                            # combining the templates for a chat template
                            chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
                            #chat_prompt = PromptTemplate.from_template([system_message_prompt, human_message_prompt])

                            # formatting the prompt with input variables
                            formatted_prompt=chat_prompt.format_prompt(programmer_type=programmer_type,coding_style=coding_style, language=language,model=model, method=method,documentation=documentation, format_instructions = format_instructions).to_messages()
                            # Format the templates
                            #formatted_system_message =system_message_prompt.format(programmer_type=programmer_type, coding_style=coding_style,language=language, documentation=documentation)
                            #formatted_human_message =human_message_prompt.format(model=model, method=method, format_instructions=format_instructions)

                            # Combine the formatted messages
                            #formatted_prompt = [formatted_system_message, formatted_human_message]
                            # Convert the formatted messages into Message instances
                            #system_message = Message(role='system', content=formatted_system_message)
                            #human_message = Message(role='user', content=formatted_human_message)

                            # Combine the Message instances into a list
                            #formatted_prompt = [system_message, human_message]


                            #memory = ConversationBufferMemory(return_messages=True)
                            # running the model
                            output = openai(formatted_prompt)
                            print(f"output:{output}")
                            #print(f"--->{output.content}<---------------------")
                            #conversation = ConversationChain(memory=memory,
                            #                                 prompt=chat_prompt,
                            #                                 llm=openai)
                            #output = conversation.predict(input=formatted_promt)
                            #output_content = output.content
                            parsed_output = output_parser.parse(output)
                            print(f"|||||=========: {parsed_output}")

                            #print(type(output_content))
                            #output_dict = json.loads(output_content)
                            #print('output_content=', output_content)
                            #print('-------------------------------')
                            #print(output_content['code'])
                            #print(output_content['function_name'])
                            # Manually parse the output
                            #parsed_output = {}
                            #parsed_output['code'] = output_content.split('\n')[0]
                            #parsed_output['function_name'] = output_content.split('\n')[1]

                            # parsing the output into our json like format
                            try:
                                parsed_output =output_parser.parse(output) #json.loads(output) #output_parser.parse(output.content)
                                #parsed_output = output_parser.parse(json.loads(output)) #json.loads(output) #output_parser.parse(output.content)
                                #output_content = output.content
                                print("-----------------")
                                #print(f"output_content:{output_content}")
                                print(f"=========: {parsed_output}")
                                #output_dict = json.loads(output_content)
                                #output_dict = output_dict.strip().strip('```')
                                #print(f"output_dict:{output_dict}")
                                #print(output_dict)
                                #code_lines = output_dict['code'].split(output_dict['function_name']) #output_dict['code'].split(function_name)
                                # Check if the keys exist and have content
                                '''for key, value in output_dict.items():
                                    print(f"Key: {key}, Value: {value}")
                                if 'code' in output_dict:
                                    code_value = output_dict['code']
                                    if code_value.strip():  # Check if code_value is not an empty string after stripping whitespace
                                        print("Code:", code_value)
                                    else:
                                        print("Code is empty.")
                                else:
                                    print("Code key is not present in the dictionary.")
                                '''
                                field_names = OutputModel.__fields__.keys()
                                print(f"field_names:", {field_names})
                                #print(output_dict['code'])
                                print(f"parsed_output.code:",{parsed_output['code']})
                                print(f"parsed_output.func:",{parsed_output['function_name']})
                                #print(code_lines[0])
                                print("\n")
                                #print(code_lines[1])

                                #print(output_dict['function_name'])

                                with open(f"./data/code/GPT4/SIR/expert/test-{model}-{programmer_type}-{language}-{coding_style}-{method}-{documentation}.txt", 'w') as f:
                                    #print(code_lines[0])
                                    #print(output_dict, file=f)
                                    f.write(json.dumps(parsed_output, default= str, indent=4))
                                f.close()

                                with open(f"./data/code/GPT4/SIR/expert/output-code-{model}-{programmer_type}-{language}-{coding_style}-{method}-{documentation}.py", 'w') as f:
                                    #print(code_lines[0])
                                    print(parsed_output['code'], file=f)
                                f.close()

                                with open(f"./data/code/GPT4/SIR/expert/output-function-{model}-{programmer_type}-{language}-{coding_style}-{method}-{documentation}.txt", 'w') as f:
                                    #print(code_lines[1])
                                    print(parsed_output['function_name'], file=f)
                                f.close()

                                counter += 1
                            except:
                                print("Missed a parse")
                                counter += 1
