
import json

json_path="Magazine_Subscriptions_5"

def parse(path):
    with open(json_path+".json") as f:
        for l in f:
            yield json.loads(l)

attributes=["reviewerID","asin","reviewText","overall","summary"]

with open(json_path+"_reduced"+".json",mode='w') as reducedf:
    for record in parse(json_path):
        temp_record={attr:record[attr] for attr in record.keys() if attr in attributes}
        json.dump(temp_record,reducedf)
        print("",file=reducedf)
