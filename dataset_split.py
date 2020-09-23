import json
import csv
import pickle
import gensim
import os
import string
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

reduced_json_path="Magazine_Subscriptions_5_reduced"

def parse(path):
    with open(path+".json") as f:
        for l in f:
            yield json.loads(l)

item_id=1
user_id=1
asin2itemid={}
itemid2asin={}
reviewerid2userid={}
corpus={}

with open("rating.csv",mode='w') as frating:
    rating_csvwriter=csv.writer(frating)
    header=["itemID","userID"]
    ratings=[]
    rating_csvwriter.writerow(header)
    for record in parse(reduced_json_path):
        reviewerID=record["reviewerID"]
        asin=record["asin"]
        if reviewerID not in reviewerid2userid:
            reviewerid2userid[reviewerID]=user_id # string->int, start from 0
            user_id+=1
        if asin not in asin2itemid:
            asin2itemid[asin]=item_id # string->int
            itemid2asin[item_id]=asin # int->string
            item_id+=1
        ratings.append([asin2itemid[asin],reviewerid2userid[reviewerID]])
        review_text=record.get("reviewText")
        if review_text:
            temp_text=corpus.get(asin)
            if temp_text:
                temp_text+='\n'+review_text
            else:
                temp_text=review_text
            corpus[asin]=temp_text
        else:
            print(record)
    
    def read_corpus(corpus):
        for asin, doc in corpus.items():
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(doc),[asin])
    train_corpus=list(read_corpus(corpus))
    model=gensim.models.doc2vec.Doc2Vec(vector_size=100, min_count=2, epochs=40, workers=16)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    # fdoc2vec=gensim.test.utils.get_tmpfile("doc2vec_model")
    model.save("doc2vec_model")

    rating_csvwriter.writerows(ratings)

    with open("asin2itemid.pkl",mode='wb') as f:
        pickle.dump(asin2itemid,f)
    with open("itemid2asin.pkl",mode='wb') as f:
        pickle.dump(itemid2asin,f)
    with open("reviewerid2userid.pkl",mode='wb') as f:
        pickle.dump(reviewerid2userid,f)
