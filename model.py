import os
import pandas as pd
import numpy as np
import torch
from torch import nn

# data ingestion
data_dir = "./data/function_nets_tokenized"
labels_dir = "./data/labels_tokenized"

"""
Let's replace as the following:
empty: 0
FUNCTION: 1-99
PREDICATE: 100-199
LANGUAGE_PRIMITIVE: 200-299
ABSTRACT: 300-399
EXPRESSION: 400-499
LITERAL: 500-599
IMPORTED: 600-699
IMPORTED_METHOD: 700-799
This allows replacement while also counting max vocab. 
Will need to run through again to pad by longest sentence with 0's
"""
function_max = 0
predicate_max = 0
primitive_max = 0
abstract_max = 0
expression_max = 0
literal_max = 0
imported_max = 0
imported_method_max = 0
sentence_max = 0


train_data = []
dev_data = []
input = []
output = []

for filename in os.listdir(data_dir):
    file_id = filename.split("-")[2].split(".")[0]
    f = os.path.join(data_dir, filename)
    function_net = []
    with open(f,'r') as fn_file:
        lines = fn_file.readline()
        fn_file.close()
    for line in lines:
        line_split = line.split()
        function_net.append(line_split)
    input.append(function_net)
    label_found = False
    for outputname in os.listdir(labels_dir):
        output_id = outputname.split("-")[2].split(".")[0]
        if output_id == file_id and not label_found:
            o = os.path.join(labels_dir, outputname)
            with open(o,'r') as out_file:
                lines = out_file.readline()
                out_file.close()
            for line in lines:
                line_split = [line]
                output.append(line_split)
            label_found = True
    if not label_found:
        print(f"WARNING!! Label missed for file id:{file_id}\n")

train_data.append(input)
train_data.append(output)

# I now need to assign an index for each word in my vocabulary, and replace the words in 
# the data with those indexes, will need to pad each sentences too up to the longest in the data



# let's first make the embedding layer, There are 8 different words,
# each of which can have n indexes mentioning it. Initially, perhaps a 
# dimension of 8, one basis vector for each direction, but I don't think
# we need that much information, so let's reduce it to 3. Let's overscope
# for this test model. In the future I want an embedding for each 8 words
# and then another one for every number from 0-500, just adding them together
# when relevant, for now let's assume 50 for each words, so vocab of 400

embedding = nn.Embedding(400, 3)
input = torch.LongTensor([[3, 7, 34]])
print(embedding(input))