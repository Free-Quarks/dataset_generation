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

from tqdm import tqdm

from langchain import PromptTemplate, HuggingFaceHub, LLMChain


#------------HUGGINGFACE STUFF------------------------------------
'''
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

'''
#------------------------------------------------------------------

f = open('openai_key.txt', 'r')

openai_key = f.read()

f.close()

# This parsing struggles if trying to find line numbers, it will often return the lines in their entirety
# however, it seems to work well for returning the name of the function that can be used to define the dynamics
# makes the output into a json like structure
response_schemas = [
    ResponseSchema(name="code", description="the entire code"),
    ResponseSchema(name="model_function", description="The name of the function that contains the model dynamics")
]

# for structured output parsing, converts schema to langhchain object
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# for structured output parsing, makes the instructions to be passed as a variable to prompt template
format_instructions = output_parser.get_format_instructions()

temperature = 0.8

# initialize the models
openai = ChatOpenAI(
    temperature=temperature,
    model_name='gpt-3.5-turbo',
    openai_api_key=openai_key
)

# potential other options: 
# coding styles
# student/graduate student, postdoc, expert, layperson, verbose, concise/compact, efficient, inefficient,
# beginner, intermediate, expert, messy, clean, obfuscated, clear
# look into taxomology of code_style
# compartmental model with 3 compartments, 4 compartments, 5 compartments
# language - fortran/python/matlab

programmer_type_list = ["college student", "software engineer"]
language_list = ["python"]
model_list = ["SIR", "SEIR", "SEIRD", "SIDARTHE", "SEIRHD"] # could probably make up things and it would make diff eq's for them
method_list = ["Euler", "odeint", "RK2", "RK3", "RK4"]
# if plannign to make a ML model to predict the line numbers for labels, 
# adding a field for data augmentation from lines numbers could be useful too

# chat models have two templates, one is their "role", the system prompt, and the other is the input or "human prompt", kind of like 
# zero shot learning in a sense

counter = 0

for i in tqdm(range(2)):
    for programmer_type in tqdm(programmer_type_list):
        for language in tqdm(language_list):
            for model in tqdm(model_list):
                for method in tqdm(method_list):


                    template="You are a {programmer_type} that writes {language} code to simulate and plot epidemiology compartmental models."
                    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
                    human_template="{model} using {method} \n{format_instructions}"
                    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

                    # combining the templates for a chat template
                    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

                    # formatting the prompt with input variables
                    formatted_prompt = chat_prompt.format_prompt(programmer_type=programmer_type, language=language, model=model, method=method, format_instructions = format_instructions).to_messages()

                    # running the model
                    output = openai(formatted_prompt)

                    # parsing the output into our json like format
                    try: 
                        parsed_output = output_parser.parse(output.content)

                        #print(parsed_output['code'])
                        #print("\n")

                        #print(parsed_output['model_function'])

                        with open(f"./data/code/output-code-{counter}.py", 'w') as f:
                            print(parsed_output['code'], file=f)
                        f.close()

                        with open(f"./data/code/output-function-{counter}.txt", 'w') as f:
                            print(parsed_output['model_function'], file=f)
                        f.close()

                        counter += 1
                    except:
                        print("Missed a parse")
                        counter += 1
