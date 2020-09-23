# import scipy.sparse as sp
import numpy as np
from scipy.sparse import dok_matrix,lil_matrix
import pandas as pd
import csv
from tqdm import tqdm
import pickle
import gensim
import time

class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        self.allMatrix=self.load_rating_file_as_matrix(path+"/rating.csv")
        self.trainMatrix, self.validMatrix, self.testRatings=self.split_data(self.allMatrix,seed=int(time.time()))
        self.num_users, self.num_items = self.trainMatrix.shape
        self.textualfeatures = self.load_textual_image_features(path)
        self.item_neighborhood=self.load_itemneighborhood_as_list(path+"/item_neighborhood.csv")
        self.user_neighborhood=self.load_userneighborhood_as_list()

    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        df = pd.read_csv(filename, index_col=None, usecols=None)
        for index, row in df.iterrows():
            u, i = int(row['userID']), int(row['itemID'])
            num_users = max(num_users, u)
            num_items = max(num_items, i)
        # Construct matrix
        mat = dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        for index, row in df.iterrows():
            user, item = int(row['userID']), int(row['itemID'])
            mat[user, item] = 1.0
        return mat

    def load_textual_image_features(self,data_path):
        with open(data_path+"/itemid2asin.pkl","rb") as i2af:
            itemid2asin=pickle.load(i2af)
        doc2vec_model=gensim.models.doc2vec.Doc2Vec.load(data_path+"/doc2vec_model")

        features=[]
        for i in range(self.num_items):
            features.append(doc2vec_model.docvecs[itemid2asin[i]])
        return np.asarray(features,dtype=np.float32)

    # not used
    def load_itemneighborhood_as_matrix(self,data_path):
        mat=dok_matrix((self.num_items+1, self.num_items+1))
        # 读取csv
        with open(data_path,mode="r", newline='') as fnei:
            neighbor_reader=csv.reader(fnei)
            for row in neighbor_reader:
                itemid=int(row[0])
                for neighborid in row[1:]:
                    mat[itemid,neighborid]=1
        
        return mat

    def load_itemneighborhood_as_list(self, data_path):
        lst=[[]]*(self.num_items)
        with open(data_path,mode="r", newline='') as fnei:
            neighbor_reader=csv.reader(fnei)
            for row in neighbor_reader:
                itemid=int(row[0])
                for neighborid in row:
                    lst[itemid].append(int(neighborid))
        
        return lst
    
    def load_userneighborhood_as_list(self):
        lst=[[]]*(self.num_items)
        rarrary,carrary=self.trainMatrix.nonzero()
        for (user,item) in zip(rarrary,carrary):
            lst[item].append(user)
        
        return lst
    
    def split_data(self,user_item_matrix, split_ratio=(3, 1, 1), seed=1):
        # set the seed to have deterministic results
        # np.random.seed(seed)
        train = dok_matrix(user_item_matrix.shape)
        validation = dok_matrix(user_item_matrix.shape)
        test = dok_matrix(user_item_matrix.shape)
        # convert it to lil format for fast row access
        user_item_matrix = lil_matrix(user_item_matrix)
        for user in tqdm(range(user_item_matrix.shape[0]), desc="Split data into train/valid/test"):
            items = list(user_item_matrix.rows[user])
            if len(items) >= 5:

                np.random.shuffle(items)

                train_count = int(len(items) * split_ratio[0] / sum(split_ratio))
                valid_count = int(len(items) * split_ratio[1] / sum(split_ratio))

                for i in items[0: train_count]:
                    train[user, i] = 1
                for i in items[train_count: train_count + valid_count]:
                    validation[user, i] = 1
                for i in items[train_count + valid_count:]:
                    test[user, i] = 1
        print("{}/{}/{} train/valid/test samples".format(
            len(train.nonzero()[0]),
            len(validation.nonzero()[0]),
            len(test.nonzero()[0])))
        return train, validation, test
