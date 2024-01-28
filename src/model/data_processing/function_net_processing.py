import requests
import json
import os
import time

SKEMA_ADDRESS = os.environ.get("SKEMA_ADDRESS", "https://api.askem.lum.ai")
URL = f"{SKEMA_ADDRESS}/code2fn/fn-given-filepaths"


def dictionary_to_gromet_json(
    o, fold_level=5, indent=4, level=0, parent_key=""
):
    if level < fold_level:
        newline = "\n"
        space = " "
    else:
        newline = ""
        space = ""
    ret = ""
    if isinstance(o, str):
        ret += json.dumps(
            o
        )  # json.dumps() will properly escape special characters
    elif isinstance(o, bool):
        ret += "true" if o else "false"
    elif isinstance(o, float):
        ret += "%.7g" % o
    elif isinstance(o, int):
        ret += str(o)
    elif isinstance(o, list):
        ret += "[" + newline
        comma = ""
        for e in o:
            ret += comma
            comma = "," + newline
            ret += space * indent * (level + 1)
            ret += dictionary_to_gromet_json(
                e, fold_level, indent, level + 1, parent_key
            )
        ret += newline + space * indent * level + "]"
    elif isinstance(o, dict):
        ret += "{" + newline
        comma = ""
        for k, v in o.items():
            ret += comma
            comma = "," + newline
            ret += space * indent * (level + 1)
            ret += '"' + str(k) + '":' + space
            if k == "fn":
                ret += dictionary_to_gromet_json(v, 4, indent, level + 1, k)
            elif k == "attributes":
                ret += dictionary_to_gromet_json(v, 6, indent, level + 1, k)
            elif k == "bf" and parent_key == "fn":
                ret += dictionary_to_gromet_json(v, 5, indent, level + 1, k)
            elif k == "bf" and parent_key == "value":
                ret += dictionary_to_gromet_json(v, 7, indent, level + 1, k)
            else:
                ret += dictionary_to_gromet_json(
                    v, fold_level, indent, level + 1, k
                )
        ret += newline + space * indent * level + "}"
    elif o is None:
        ret += "null"
    else:
        # NOTE: We added this check here to catch any Python objects that
        # didn't get turned into dictionaries.
        # This is to circumvent Swagger's inability to generate to_dicts that support
        # multi-dimensional dictionaries. This becomes an issue for us when we're storing
        # an array of metadata arrays
        if hasattr(o, "to_dict"):
            temp = del_nulls(o.to_dict())
            ret += dictionary_to_gromet_json(
                temp, fold_level, indent, level, parent_key
            )
        else:
            ret += str(o)
    return ret


def del_nulls(d):
    for key, value in list(d.items()):
        if isinstance(value, list):
            for elem in value:
                if isinstance(elem, dict):
                    del_nulls(elem)
        if isinstance(value, dict):
            del_nulls(value)
        if value is None:
            del d[key]

    return d

if __name__ == "__main__":

    directory = '../../data/code'
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)

        try:
            single_snippet_payload = {
                    "files": [filename],
                    "blobs": [open(f).read()],
                }
            
            time.sleep(1.5)
            response = requests.post(URL, json=single_snippet_payload)
            print(response)
            gromet = json.dumps(response.json(), indent=4)

            gromet_json = json.loads(gromet)
            try:
                with open(f"../../../data/function_nets/{filename[:-3]}.json", "w") as outfile:
                    gromet_collection_dict = (
                                    gromet_json
                                )
                    outfile.write(
                        dictionary_to_gromet_json(
                            del_nulls(gromet_collection_dict), level=0
                            )
                        )
            except:
                gromet_collection_dict = (gromet_json)
                with open(f"../../../data/function_nets/{filename[:-3]}.json", "w") as outfile:
                    outfile.write(f"{dictionary_to_gromet_json(del_nulls(gromet_collection_dict), level=0)}")
        except:
            print(f"{f} failed to process\n")