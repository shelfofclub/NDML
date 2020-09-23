import pickle
import json
import csv
import gzip

with open("asin2itemid.pkl",mode='rb') as f:
    asin2itemid=pickle.load(f)
with open("itemid2asin.pkl",mode='rb') as f:
    itemid2asin=pickle.load(f)
# with open("reviewerid2userid.pkl",mode='rb') as f:
#     reviewerid2userid=pickle.load(f)

def parse(path):
    with gzip.open(path+".json.gz") as f:
        for l in f:
            yield json.loads(l)

with open("item_neighborhood.csv", mode="w", newline='') as fnei:
    nei_csvwriter=csv.writer(fnei)
    for record in parse("meta_Luxury_Beauty"):
        also_bought=record.get("also_buy")
        # raise NotImplementedError
        if also_bought is not None:
            itemid=asin2itemid.get(record["asin"])
            if itemid is not None:
                neighborsid=[itemid]
                for neighbor_asin in also_bought:
                    neighborid=asin2itemid.get(neighbor_asin)
                    if neighborid is not None:
                        neighborsid.append(neighborid)
                nei_csvwriter.writerow(neighborsid)