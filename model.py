import os
import pandas as pd
import numpy as np
import torch
from torch import nn

# data ingestion
data_dir = "./data/function_nets_tokenized"
labels_dir = "./data/labels_tokenized"

# attempt to test this with the preclustering indexes as below verse
# just running them exactly in sequence 

"""
Let's replace as the following:
empty: 0
FUNCTION: 1-200
PREDICATE: 201-400
LANGUAGE_PRIMITIVE: 401-600
EXPRESSION: 601-800
ABSTRACT: 801-1000
LITERAL: 1001-1200
IMPORTED_METHOD: 1201-1400
IMPORTED: 1401+
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
program_max = 0


train_data = []
dev_data = []
input = []
output = []

# imported data into lists 
for filename in os.listdir(data_dir):
    file_id = filename.split("-")[2].split(".")[0]
    f = os.path.join(data_dir, filename)
    function_net = []
    with open(f,'r') as fn_file:
        # entry for each line
        lines = fn_file.readlines()
        fn_file.close()
    for line in lines:
        # each line split by token
        line_split = line.split()
        function_net.append(line_split)
        # find max sentence length for padding
        line_len = len(line_split)
        if line_len > sentence_max:
            sentence_max = line_len
    input.append(function_net)
    # find max number of lines for padding
    fn_len = len(function_net)
    if fn_len > program_max:
        program_max = fn_len
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

print(f"sentence_max={sentence_max}") # currently 55, pad to 65
print(f"program_max={program_max}") # currently 82, pad to 90

# padding the data to all be same dimensions

# sentence padding and conversion to indexes and conversion to np arrays
np_train_data = []
for fun_net in train_data[0]:
    np_fun_net = []
    function_count = 0
    predicate_count = 0
    primitive_count = 0
    abstract_count = 0
    expression_count = 0
    literal_count = 0
    imported_count = 0
    imported_method_count = 0
    for line in fun_net:
        np_line = []
        for token in line:
            if token[:4] == "FUNC":
                count = int(token[8:])
                np_token = count
                function_count += 1
            elif token[:4] == "PRED":
                count = int(token[9:])
                np_token = count + 200
                predicate_count += 1
            elif token[:4] == "PRIM":
                count = int(token[9:])
                np_token = count + 400
                primitive_count += 1
            elif token[:4] == "EXPR":
                count = int(token[10:])
                np_token = count + 600
                expression_count += 1
            elif token[:4] == "ABST":
                count = int(token[8:])
                np_token = count + 800
                abstract_count += 1
            elif token[:4] == "LITE":
                count = int(token[7:])
                np_token = count + 1000
                literal_count += 1
            elif token[:4] == "IMPO":
                if len(token) > 11:
                    count = int(token[15:])
                    np_token = count + 1200
                    imported_method_count += 1
                else:
                    count = int(token[8:])
                    np_token = count + 1400
                    imported_count += 1
            np_line.append(np_token)
        while len(np_line) < 65:
            np_line.append(0)
        np_fun_net.append(np_line)
    np_train_data.append(np_fun_net)
    # counting max of each token in a file
    if function_count > function_max:
        function_max = function_count
    if predicate_count > predicate_max:
        predicate_max = predicate_count
    if primitive_count > primitive_max:
        primitive_max = primitive_count    
    if expression_count > expression_max:
        expression_max = expression_count    
    if literal_count > literal_max:
        literal_max = literal_count
    if abstract_count > abstract_max:
        abstract_max = abstract_count
    if imported_count > imported_max:
        imported_max = imported_count
    if imported_method_count > imported_method_max:
        imported_method_max = imported_method_count

print(f"function_max={function_max}")
print(f"primitive_max={primitive_max}")
print(f"predicate_max={predicate_max}")
print(f"expression={expression_max}")
print(f"literal_max={literal_max}")
print(f"abstract_max={abstract_max}")
print(f"imported_max={imported_max}")
print(f"imported_method_max={imported_method_max}")
print(f"Total vocab: {function_max+primitive_max+predicate_max+expression_max+literal_max+abstract_max+imported_max+imported_method_max}")

# program length padding
for fun_net in np_train_data:
    while len(fun_net) < 90:
        padded_sent = np.zeros(65)
        fun_net.append(padded_sent)

np_train_data = np.array(np_train_data, dtype=int)
print(np_train_data.shape)


dataset = [np_train_data, output]
# I now need to assign an index for each word in my vocabulary, and replace the words in 
# the data with those indexes, will need to pad each sentences too up to the longest in the data



# let's first make the embedding layer, There are 8 different words,
# each of which can have n indexes mentioning it. Initially, perhaps a 
# dimension of 8, one basis vector for each direction, but I don't think
# we need that much information, so let's reduce it to 3. Let's overscope
# for this test model. In the future I want an embedding for each 8 words
# and then another one for every number from 0-500, just adding them together
# when relevant, for now let's assume 50 for each words, so vocab of 400

embedding = nn.Embedding(800, 3)
input = torch.LongTensor([[3, 7, 34]])
#print(embedding(torch.from_numpy(np_train_data[0])))