import numpy as np
import tensorflow as tf
from scipy.sparse import lil_matrix
from datetime import datetime

class RecallEvaluator(object):
    def __init__(self, model, train_user_item_matrix,test_user_item_matrix ):
        self.model = model
        self.train_user_item_matrix = lil_matrix(train_user_item_matrix)
        self.test_user_item_matrix = lil_matrix(test_user_item_matrix)
        self.k=tf.placeholder(tf.int32)

        n_users = train_user_item_matrix.shape[0]
        self.user_to_test_set = {u: set(self.test_user_item_matrix.rows[u])
                                 for u in range(n_users) if self.test_user_item_matrix.rows[u]}

        if self.train_user_item_matrix is not None:
            self.user_to_train_set = {u: set(self.train_user_item_matrix.rows[u])
                                      for u in range(n_users) if self.train_user_item_matrix.rows[u]}
            self.max_train_count = max(len(row) for row in self.train_user_item_matrix.rows)
        else:
            self.max_train_count = 0
        
        # topk运算结点需要一开始就存下来
        # 如果每次评估时，都调用一次top_k
        # 会新建一个topk结点，显存泄漏
        # 当模型较复杂时，很快就耗尽显存资源
        self.topk=tf.nn.top_k(self.model.item_scores,self.k+self.max_train_count)

    def eval(self, sess, users, item_neighbors, user_neighbors,k=50,ftest=None):
        # 首先去随机数用于后面采样定位
        # 实验中所用的数据集中，用户物品数量都在1000000以内
        # 足够覆盖每一个邻居
        # 若在使用个别更大的数据集，可以相应调大数值
        item_neis_id_ranindex_test=np.random.randint(1000000,size=self.model.n_items)

        # 取模后访问列表的元素，
        # 这种实现方式是尝试的多种采样方法中最快的
        # 可以在几毫秒或几十毫秒内完成采样
        # 而其他方法则较慢
        # 采样用户邻居时，可以先转换为numpy的array，可以使用列表进行索引
        # 但是实际效果很慢，不如一个个取数，最后转为numpy的array
        # 输入到tensorflow中
        # (N_ITEM, 1)
        item_neis_id_sample_test=[nei[randindex % len(nei)] for nei,randindex in zip(item_neighbors,item_neis_id_ranindex_test)]
        # print("item nei eval sample done",datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])

        user_neis_id_ranindex_test=np.random.randint(1000000,size=(self.model.n_items, len(users)))
        # (N_USER_IDS*N_ITEM, 1)
        user_neis_id_sample_test=(np.array([[nei[ra] for ra in (ran % len(nei))] for nei,ran in zip(user_neighbors,user_neis_id_ranindex_test)])).T.flatten()
        # print("user nei eval sample done",datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
        # raise NotImplementedError

        _, user_tops = sess.run(self.topk,
                                {self.model.score_user_ids: users,self.k:k, self.model.item_neis_id_sample_test:item_neis_id_sample_test, self.model.user_neis_id_sample_test:user_neis_id_sample_test})
        recalls = []
        precisions = [0]
        hit_ratios = [0]
        ndcgs = [0]

        for user_id, tops in zip(users, user_tops):
            print("user", user_id,file=ftest)
            train_set = self.user_to_train_set.get(user_id, set())
            test_set = self.user_to_test_set.get(user_id, set())
            print("test",test_set,file=ftest)
            top_n_items = 0
            hits = 0
            # dcg = 0.0
            # idcg = 0.0
            # for i in range(min(len(test_set), k)):
            #     idcg += 1 / (np.log(i + 2) / np.log(2))
            print("pred",end=' ',file=ftest)
            for i in tops:
                # ignore item in the training set
                if i in train_set:
                    continue
                elif i in test_set:
                    print(i, end=' ',file=ftest)
                    # dcg += 1 / (np.log(top_n_items + 2) / np.log(2))
                    hits += 1
                top_n_items += 1
                if top_n_items == k:
                    break
            print("\n",file=ftest)
            recalls.append(hits / float(len(test_set)))
            # precisions.append(hits / float(k))
            # hit_ratios.append(1.0) if hits > 0 else hit_ratios.append(0.0)
            # ndcgs.append(dcg/idcg)


        return recalls,ndcgs,hit_ratios,precisions
