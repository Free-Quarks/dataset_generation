import json
import os

# class for storing info about each token, needed to make sure references to code defined later don't get new tokens
class toke:
    def __inti__(self):
        self.type = "temp"
        self.idx = 0
    def __repr__(self):
        return f"type: {self.type}, idx: {self.idx}"
    def __str__(self):
        return f"type: {self.type}, idx: {self.idx}"

function_type_list = ["FUNCTION", "PREDICATE","LANGUAGE_PRIMITIVE","ABSTRACT","EXPRESSION","LITERAL","IMPORTED","IMPORTED_METHOD"]

directory = './data/function_nets'

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)

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
                seq.append(t1)
        except:
            None
        full_seq.append(seq)


    #print(full_seq)
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

    with open(f"./data/function_nets_tokenized/{filename[:-5]}.txt", "w") as outfile:
        outfile.writelines(full_seq_strings)
        outfile.close()