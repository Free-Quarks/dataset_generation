import json
import os
import numpy as np

# class for storing info about each token, needed to make sure references to code defined later don't get new tokens
class toke:
    def __init__(self):
        self.type = "temp"
        self.idx = 0
        self.name = "temp"
    def __repr__(self):
        return f"type: {self.type}, idx: {self.idx}"
    def __str__(self):
        return f"type: {self.type}, idx: {self.idx}"

# creates a tokenized function network and label, requires both directories to be the same size
def tokenize(fn_directory, labels_directory, output_tokens_directory, output_labels_directory):

    function_type_list = ["FUNCTION", "PREDICATE","LANGUAGE_PRIMITIVE","ABSTRACT","EXPRESSION","LITERAL","IMPORTED","IMPORTED_METHOD"]

    for filename in os.listdir(fn_directory):
        f = os.path.join(fn_directory, filename)

        with open(f,'r') as fn_file:
            fn_data = json.load(fn_file)
            fn_file.close()

        # construct a sequence of sequences of tokens
        full_seq = []
        for i, obj in enumerate(fn_data['modules'][0]['fn_array']):
            #print('----------------')
            #print('top level')
            #print(obj['b'][0]['function_type'])
            #print('contains')
            t = toke()
            t.type = obj['b'][0]['function_type']
            t.idx = i+1
            if t.type =="FUNCTION":
                try:
                    t.name = obj['b'][0]['name'][:-4]
                except:
                    None
            seq = [t]
            try:
                for content in obj['bf']:
                    t1 = toke()
                    #print(content['function_type'])
                    t1.type = content['function_type']
                    if t1.type == "EXPRESSION" or t1.type == "FUNCTION" or t1.type == "PREDICATE":
                        t1.idx = content['body']
                    else:
                        t1.idx = i
                    if t1.type == "FUNCTION":
                        t1.name = content['name'][:-4]
                    seq.append(t1)
            except:
                None
            full_seq.append(seq)


        # this adds the numbering for referencing
        for f_type in function_type_list:
            count = 1
            if f_type != "EXPRESSION" and f_type != "FUNCTION" and f_type != "PREDICATE":
                for seq in full_seq:
                    for token in seq:
                        if token.type == f_type:
                            if f_type == "LANGUAGE_PRIMITIVE":
                                token.type = "PRIMITIVE"+f"{count}"
                                count += 1
                            else:
                                token.type = f_type+f"{count}"
                                count += 1
            else:
                for seq in full_seq:
                    for token in seq:
                        if token.type == f_type:
                            token.type = f_type+f"{count}"
                            count += 1
                            for seq1 in full_seq:
                                for token1 in seq1:
                                    if token1.type == f_type and token1.idx == token.idx:
                                        token1.type = token.type
        full_seq_tokens = []
        for seq in full_seq:
            s = []
            for token in seq:
                s.append(f"{token.type}")
            full_seq_tokens.append(s)

        full_seq_strings = []
        for seq in full_seq_tokens:
            s = ""
            for token in seq:
                s = s+" "+token
            s = s+"\n"
            full_seq_strings.append(s)

        file_num = filename[12:-5]

        # find the rigth label file and get it label and name
        for label_file in os.listdir(labels_directory):
            if label_file[16:-4] == file_num:
                fl = os.path.join(labels_directory, label_file)
                label_file_name = label_file
                with open(fl,'r') as l_file:
                    label = l_file.read()
                    l_file.close()
        # find the token corresponding to it's name
        for seq in full_seq:
            for token in seq:
                if token.name == label.strip():
                    tokenized_label = token.type
        
        with open(f"{output_labels_directory}/{label_file_name}", "w") as tl_file:
            tl_file.write(tokenized_label)
            tl_file.close()            

        with open(f"{output_tokens_directory}/{filename[:-5]}.txt", "w") as outfile:
            outfile.writelines(full_seq_strings)
            outfile.close()


# This code takes in the labeled data and converts them to indicies for future embedding models
def tokens_to_indices(output_tokens_directory, output_labels_directory):
    # data ingestion
    data_dir = output_tokens_directory
    labels_dir = output_labels_directory

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
                    lines = out_file.readlines()
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

    # converting the output into indicies as well
    i_output = []
    for entry in output:
        ind = int(entry[0][8:])
        i_output.append(ind)

    # program length padding
    for fun_net in np_train_data:
        while len(fun_net) < 90:
            padded_sent = np.zeros(65)
            fun_net.append(padded_sent)

    np_output = np.array(i_output, dtype=int)
    np_train_data = np.array(np_train_data, dtype=int)
    print("np_train_data shape:", np_train_data.shape)

    dataset = []
    dataset.append(np_train_data)
    dataset.append(np_output.reshape((90,1)))
    return dataset


if __name__ == "__main__":

    fn_directory = '../../../data/function_nets'
    labels_directory = '../../../data/labels'
    output_tokens_directory = '../../../data/function_nets_tokenized'
    output_labels_directory = '../../../data/labels_tokenized'

    tokenize(fn_directory, labels_directory, output_tokens_directory, output_labels_directory)