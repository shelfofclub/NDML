import functools
import numpy
import tensorflow as tf
import toolz
from tqdm import tqdm
from evaluator import RecallEvaluator
from sampler import WarpSampler
import Dataset
import pandas as pd
from tensorflow.contrib.layers.python.layers import regularizers
import os
import argparse
from datetime import datetime
from tensorflow.python import debug as tf_debug
import random

random.seed(2019)
numpy.random.seed(2020)
tf.set_random_seed(2021)



def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)

    return decorator

@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    # name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            # with tf.variable_scope(name, *args, **kwargs):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class NDML(object):
    def __init__(self,
                 n_users,
                 n_items,
                 embed_dim=20,
                 batch_size = 10,
                 n_negative=20,
                 textualfeatures=None,
                 margin=1.5,
                 master_learning_rate=0.1,
                 clip_norm=1.0,
                 hidden_layer_dim=128,
                 dropout_rate=0.2,
                 feature_l2_reg=0.1,
                 feature_projection_scaling_factor=0.5,
                 use_rank_weight=True,
                 use_cov_loss=True,
                 cov_loss_weight=0.1,
                 item_neighbors=None,
                 user_neighbors=None
                 ):
        """

        :param n_users: number of users i.e. |U|
        :param n_items: number of items i.e. |V|
        :param embed_dim: embedding size i.e. K (default 20)
        :param features: (optional) the feature vectors of items, shape: (|V|, N_Features).
               Set it to None will disable feature loss(default: None)
        :param margin: hinge loss threshold i.e. z
        :param master_learning_rate: master learning rate for AdaGrad
        :param clip_norm: clip norm threshold (default 1.0)
        :param hidden_layer_dim: the size of feature projector's hidden layer (default: 128)
        :param dropout_rate: the dropout rate between the hidden layer to final feature projection layer
        :param feature_l2_reg: feature loss weight
        :param feature_projection_scaling_factor: scale the feature projection before compute l2 loss. Ideally,
               the scaled feature projection should be mostly within the clip_norm
        :param use_rank_weight: whether to use rank weight
        :param use_cov_loss: use covariance loss to discourage redundancy in the user/item embedding
        """

        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.n_negative = n_negative
        self.clip_norm = clip_norm
        self.margin = margin
        # 实际上一定使用特征的
        # 这是cml的历史遗留问题
        if textualfeatures is not None:
            self.textualfeatures = tf.constant(textualfeatures, dtype=tf.float32)
        else:
            self.textualfeatures = None
        self.master_learning_rate = master_learning_rate
        self.hidden_layer_dim = hidden_layer_dim
        self.dropout_rate = dropout_rate
        self.feature_l2_reg = feature_l2_reg
        self.feature_projection_scaling_factor = feature_projection_scaling_factor
        self.use_rank_weight = use_rank_weight
        self.use_cov_loss = use_cov_loss
        self.cov_loss_weight = cov_loss_weight
        self.user_positive_items_pairs = tf.placeholder(tf.int32, [self.batch_size, 2])
        self.negative_samples = tf.placeholder(tf.int32, [self.batch_size, self.n_negative])
        self.score_user_ids = tf.placeholder(tf.int32, [None])
        # self.max_train_count = tf.placeholder(tf.int32, None)
        # self.all_item_neis_id_padded=tf.placeholder(tf.int32, [self.batch_size+self.batch_size*self.n_negative, None])
        # self.all_item_mask=tf.placeholder(tf.bool, [self.batch_size+self.batch_size*self.n_negative, None])
        # self.user_neis_id_padded=tf.placeholder(tf.int32, [self.batch_size+self.batch_size*self.n_negative, None])
        # self.user_mask=tf.placeholder(tf.bool, [self.batch_size+self.batch_size*self.n_negative, None])
        # self.item_neis_id_padded_test=tf.placeholder(tf.int32, [self.n_items, None])
        # self.item_mask_test=tf.placeholder(tf.bool, [self.n_items, None])
        # self.user_neis_id_padded_test=tf.placeholder(tf.int32, [None, None])
        # self.user_mask_test=tf.placeholder(tf.bool, [None, None])
        # self.user_ids = tf.placeholder(tf.int32, [None])
        # self.item_ids = tf.placeholder(tf.int32, [None])
        # 邻居也是一定使用的
        if item_neighbors is not None:
            self.item_neighbors=tf.constant(item_neighbors,dtype=tf.int32)
        else:
            self.item_neighbors=None
        if user_neighbors is not None:
            self.user_neighbors=tf.constant(user_neighbors,dtype=tf.int32)
        else:
            self.user_neighbors=None
        self.all_item_neis_id_sample=tf.placeholder(tf.int32,[self.batch_size+self.batch_size*self.n_negative,])
        self.user_neis_id_sample=tf.placeholder(tf.int32,[self.batch_size+self.batch_size*self.n_negative,])
        self.item_neis_id_sample_test=tf.placeholder(tf.int32,[self.n_items,])
        self.user_neis_id_sample_test=tf.placeholder(tf.int32,[None])

        self.user_embeddings
        self.item_embeddings
        self.embedding_loss
        self.feature_loss
        self.loss
        self.optimize
    
    @define_scope
    def user_embeddings(self):
        return tf.Variable(tf.random_normal([self.n_users, self.embed_dim],
                                            stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32))
    
    @define_scope
    def item_embeddings(self):
        return tf.Variable(tf.random_normal([self.n_items, self.embed_dim],
                                            stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32))

    @define_scope
    def mlp_layer_1(self):
        return tf.layers.dense(inputs=tf.nn.l2_normalize(self.textualfeatures,dim=1),
                               units=self.hidden_layer_dim,
                               activation=tf.nn.relu, name="mlp_layer_1")

    @define_scope
    def mlp_layer_2(self):
        dropout_mlp2 = tf.layers.dropout(inputs=self.mlp_layer_1, rate=self.dropout_rate)
        hidden_layer1_mlp2 = tf.layers.dense(inputs=tf.nn.l2_normalize(dropout_mlp2,dim=1), units=self.hidden_layer_dim/2 ,activation=tf.nn.relu, name="mlp_layer_2")
        dropout1 = tf.layers.dropout(inputs=hidden_layer1_mlp2, rate=self.dropout_rate)
        hidden_layer2_mlp2 = tf.layers.dense(inputs=tf.nn.l2_normalize(dropout1,dim=1), units=self.hidden_layer_dim/2, activation=tf.nn.relu,name="mlp_layer_3")
        dropout2 = tf.layers.dropout(inputs=hidden_layer2_mlp2, rate=self.dropout_rate)
        return  tf.layers.dense(inputs=dropout2, units=self.embed_dim, name="mlp_layer_4")

    @define_scope
    def feature_projection(self):
        """
        :return: the projection of the feature vectors to the user-item embedding
        """

        # 特征需要注意
        # 因为是外部输入的数值
        # 需要保证输入的都是正确的
        # 如果有坏的数，例如Nan
        # 会导致特征的loss变成Nan
        # 经过反向传播，Nan扩散
        # 用户、物品的embeddings也变成Nan

        # feature loss
        if self.textualfeatures is not None:
            # fully-connected layer
            output = self.mlp_layer_2 * self.feature_projection_scaling_factor

            # projection to the embedding
            return tf.clip_by_norm(output, self.clip_norm, axes=[1], name="feature_projection")

    @define_scope
    def feature_loss(self):
        """
        :return: the l2 loss of the distance between items' their embedding and their feature projection
        """
        loss = tf.constant(0, dtype=tf.float32)
        if self.feature_projection is not None:
            # the distance between feature projection and the item's actual location in the embedding
            feature_distance = tf.reduce_sum(tf.squared_difference(
                self.item_embeddings,
                self.feature_projection), 1)
            # apply regularization weight
            loss += tf.reduce_sum(feature_distance, name="feature_loss") * self.feature_l2_reg
        return loss

    @define_scope
    def covariance_loss(self):
        X = tf.concat((self.item_embeddings, self.user_embeddings), 0)
        n_rows = tf.cast(tf.shape(X)[0], tf.float32)
        X = X - (tf.reduce_mean(X, axis=0))
        cov = tf.matmul(X, X, transpose_a=True) / n_rows
        return tf.reduce_sum(tf.matrix_set_diag(cov, tf.zeros(self.embed_dim, tf.float32))) * self.cov_loss_weight

    @define_scope
    def embedding_loss(self):
        """
        :return: the distance metric loss
        """
        # Let
        # N = batch size,
        # K = embedding size,
        # W = number of negative samples per a user-positive-item pair
        # user embedding (N, K)
        users = tf.nn.embedding_lookup(self.user_embeddings,
                                       self.user_positive_items_pairs[:, 0],
                                       name="users")

        pos_items = tf.nn.embedding_lookup(self.item_embeddings, self.user_positive_items_pairs[:, 1],
                                           name="pos_items")

        # negative item embedding (N, W, K)
        neg_items = tf.nn.embedding_lookup(self.item_embeddings, self.negative_samples, name="neg_items")

        # (N, FEATURE)
        pos_items_f = tf.nn.embedding_lookup(self.feature_projection, self.user_positive_items_pairs[:, 1],
                                             name="pos_items_f")
        # (N, W, FEATURE)
        neg_items_f = tf.nn.embedding_lookup(self.feature_projection, self.negative_samples, name="neg_items_f")

        # 在模型之外,已经完成了所有的铺垫
        # 邻居都已经准备好了
        # 可以直接进行embedding_lookup
        # 加快了模型的速度
        # (N+N*W, K)
        all_item_neis=tf.nn.embedding_lookup(self.item_embeddings, self.all_item_neis_id_sample, name="all_item_neis")

        # (N+N*W, K)
        user_neis=tf.nn.embedding_lookup(self.user_embeddings, self.user_neis_id_sample, name="user_neis")

        input_pos = tf.concat([users,pos_items, pos_items_f], 1,name='input_pos')
        input_neg = tf.reshape(
            tf.concat([tf.tile(tf.expand_dims(users, 1), [1, self.n_negative, 1]),neg_items,neg_items_f], 2),
            [-1, self.embed_dim * 3],name='input_neg')
        # (N+N*W, 3K)
        input = tf.concat([input_pos, input_neg], 0,name='input')

        # 拼接邻居信息,准备好输入数据的格式
        # (N+N*W, 2K)
        input_item_neis=tf.concat([tf.concat([pos_items,tf.reshape(neg_items, [-1,self.embed_dim])], axis=0), all_item_neis],axis=1,name="input_item_neis")

        input_user_neis=tf.concat([tf.concat([users,tf.reshape(tf.tile(tf.expand_dims(users, 1), [1, self.n_negative, 1]), [-1,self.embed_dim])],axis=0), user_neis],axis=1,name="input_item_neis")

        with tf.variable_scope("dense"):
            # 用户-物品注意力
            hidden_layer = tf.layers.dense(inputs=tf.nn.l2_normalize(input,dim=1), units=5*self.embed_dim,
                                          kernel_regularizer=regularizers.l2_regularizer(100.0), activation=tf.nn.tanh,
                                         name='hidden_layer')
            dropout_hl = tf.layers.dropout(hidden_layer,0.05)
            hidden_layer1 = tf.layers.dense(inputs=tf.nn.l2_normalize(dropout_hl,dim=1), units=1 * self.embed_dim,
                                            kernel_regularizer=regularizers.l2_regularizer(100.0),
                                            activation=tf.nn.relu,
                                            name='hidden_layer1')
            attention_layer_all = (self.embed_dim*1.0/3)*tf.nn.softmax(hidden_layer1,dim=-1)

            # 物品邻居信息注意力
            hidden_layer_bitem = tf.layers.dense(inputs=tf.nn.l2_normalize(input_item_neis,dim=1), units=5*self.embed_dim,
                                          kernel_regularizer=regularizers.l2_regularizer(100.0), activation=tf.nn.tanh,
                                         name='hidden_layer_bitem')
            dropout_hl_bitem = tf.layers.dropout(hidden_layer_bitem,0.05)
            hidden_layer1_bitem = tf.layers.dense(inputs=tf.nn.l2_normalize(dropout_hl_bitem,dim=1), units=1 * self.embed_dim,
                                            kernel_regularizer=regularizers.l2_regularizer(100.0),
                                            activation=tf.nn.relu,
                                            name='hidden_layer1_bitem')
            attention_layer_all_bitem = (self.embed_dim*1.0/3)*tf.nn.softmax(hidden_layer1_bitem,dim=-1)

            # 用户邻居信息注意力
            hidden_layer_buser = tf.layers.dense(inputs=tf.nn.l2_normalize(input_user_neis,dim=1), units=5*self.embed_dim,
                                          kernel_regularizer=regularizers.l2_regularizer(100.0), activation=tf.nn.tanh,
                                         name='hidden_layer_buser')
            dropout_hl_buser = tf.layers.dropout(hidden_layer_buser,0.05)
            hidden_layer1_buser = tf.layers.dense(inputs=tf.nn.l2_normalize(dropout_hl_buser,dim=1), units=1 * self.embed_dim,
                                            kernel_regularizer=regularizers.l2_regularizer(100.0),
                                            activation=tf.nn.relu,
                                            name='hidden_layer1_buser')
            attention_layer_all_buser = (self.embed_dim*1.0/3)*tf.nn.softmax(hidden_layer1_buser,dim=-1)

        # 区分正例和负例的注意力
        # 两部分分别处理
        attention_layer_pos, attention_layer = tf.split(attention_layer_all, [self.batch_size, self.batch_size*self.n_negative], 0)
        attention_layer_pos_bitem, attention_layer_bitem = tf.split(attention_layer_all_bitem, [self.batch_size, self.batch_size*self.n_negative], 0)
        attention_layer_pos_buser, attention_layer_buser = tf.split(attention_layer_all_buser, [self.batch_size, self.batch_size*self.n_negative], 0)

        # 将因子除以3,然后相加
        # 相当于取平均值
        # 比使用平均值的函数更简便
        # positive item to user distance (N)
        pos_distances = tf.reduce_sum(
            tf.squared_difference(tf.multiply(tf.add(tf.add(attention_layer_pos, attention_layer_pos_buser),attention_layer_pos_bitem), users), tf.multiply(tf.add(tf.add(attention_layer_pos,attention_layer_pos_buser), attention_layer_pos_bitem), pos_items)),
            1, name="pos_distances")
        attention_reshape = tf.transpose(tf.reshape(attention_layer, [-1, self.n_negative, self.embed_dim]), [0, 2, 1])
        attention_reshape_bitem=tf.transpose(tf.reshape(attention_layer_bitem, [-1,self.n_negative, self.embed_dim]), [0,2,1])
        attention_reshape_buser=tf.transpose(tf.reshape(attention_layer_buser, [-1, self.n_negative, self.embed_dim]), [0,2,1])
        # distance to negative items (N x W)
        distance_to_neg_items = tf.reduce_sum(
            tf.squared_difference(tf.multiply(tf.add(tf.add(attention_reshape,attention_reshape_buser),attention_reshape_bitem), tf.expand_dims(users, -1)),
                                  tf.multiply(tf.add(tf.add(attention_reshape,attention_reshape_buser),attention_reshape_bitem), tf.transpose(neg_items, [0, 2, 1]))), 1,
            name="distance_to_neg_items")
        print('distance_to_neg_items.shape:', distance_to_neg_items.shape)

        # best negative item (among W negative samples) their distance to the user embedding (N)
        closest_negative_item_distances = tf.reduce_min(distance_to_neg_items, 1, name="closest_negative_distances")
        print('closest_negtive_item_distances:',closest_negative_item_distances.shape)
        # compute hinge loss (N)
        loss_per_pair = tf.maximum(pos_distances - closest_negative_item_distances + self.margin, 0,
                                   name="pair_loss")

        # 排名权重
        if self.use_rank_weight:
            # indicator matrix for impostors (N x W)
            impostors = (tf.expand_dims(pos_distances, -1) - distance_to_neg_items + self.margin) > 0
            # approximate the rank of positive item by (number of impostor / W per user-positive pair)
            rank = tf.reduce_mean(tf.cast(impostors, dtype=tf.float32), 1, name="rank_weight") * self.n_items
            # apply rank weight
            loss_per_pair *= tf.log(rank + 1)
        # the embedding loss
        loss = tf.reduce_sum(loss_per_pair, name="loss")
        return loss

    @define_scope
    def loss(self):
        """
        :return: the total loss = embedding loss + feature loss
        """
        loss = self.embedding_loss + self.feature_loss
        if self.use_cov_loss:
            loss += self.covariance_loss
        return loss

    @define_scope
    def clip_by_norm_op(self):
        return [tf.assign(self.user_embeddings, tf.clip_by_norm(self.user_embeddings, self.clip_norm, axes=[1])),
                tf.assign(self.item_embeddings, tf.clip_by_norm(self.item_embeddings, self.clip_norm, axes=[1]))]

    @define_scope
    def optimize(self):
        # have two separate learning rates. The first one for user/item embedding is un-normalized.
        # The second one for feature projector NN is normalized by the number of items.
        gds = []

        # 使用adam而不是adagrad
        # 具体的实现延续了cml的历史传统
        gds.append(tf.train
                   .AdamOptimizer(self.master_learning_rate)
                   .minimize(self.loss, var_list=[self.user_embeddings, self.item_embeddings]))
        if self.feature_projection is not None:
            gds.append(tf.train
                       .AdamOptimizer(self.master_learning_rate)
                       .minimize(self.feature_loss / self.n_items))
        with tf.control_dependencies(gds):
            return gds + [self.clip_by_norm_op]

    @define_scope
    def item_scores(self):

        # 这部分用于评估用,在训练集训练中不涉及
        # 只与验证集,测试集有关

        # (N_USER_IDS, 1, K)
        user = tf.expand_dims(tf.nn.embedding_lookup(self.user_embeddings, self.score_user_ids), 1,name='user_test')
        # (H_USER_IDS, N_ITEM, K)
        item = tf.tile(tf.expand_dims(self.item_embeddings, 0), [tf.shape(user)[0], 1, 1],name='item_test')
        feature = tf.tile(tf.expand_dims(self.feature_projection, 0), [tf.shape(user)[0], 1, 1],name='feature_test')

        # 同样,在外面已经准备好了邻居数据
        # 这里直接embedding_lookup
        # (N_ITEM, K)
        item_neis_test=tf.nn.embedding_lookup(self.item_embeddings, self.item_neis_id_sample_test, name="item_neis_test")

        # (N_USER_IDS*N_ITEM, K)
        user_neis_test=tf.nn.embedding_lookup(self.user_embeddings, self.user_neis_id_sample_test, name="user_neis_test")

        # user=tf.Print(user,[tf.shape(user)],"check shape user")
        # item=tf.Print(item,[tf.shape(item)],"check shape item")
        # feature=tf.Print(feature,[tf.shape(feature)],"check shape feature")

        # 类似于之前的操作,拼接数据,输入全连接层中
        # (N_USER_IDS*N_ITEM, 3k)
        input_test = tf.concat(
            [tf.reshape(tf.tile(user, [1, tf.shape(item)[1], 1]), [-1, self.embed_dim]),
             tf.reshape(item, [-1, self.embed_dim]),tf.reshape(feature, [-1, self.embed_dim])], 1,name='input_test')
        # (N_ITEM, 2K)
        input_item_neis_test=tf.concat([self.item_embeddings, item_neis_test],axis=1,name="input_item_neis_test")
        # (N_USER_IDS*N_ITEM, 2K)
        input_user_neis_test=tf.concat([tf.reshape(tf.tile(tf.expand_dims(tf.nn.embedding_lookup(self.user_embeddings, self.score_user_ids), 1), [1,self.n_items, 1]),[-1,self.embed_dim]),user_neis_test],axis=1, name="input_user_neis_test")

        with tf.variable_scope('dense'):
            # 这里只是评估模型的性能
            # 所以并不训练模型
            # 重用之前的参数,并且设置
            # trainable=False

            # 用户-物品注意力
            hidden_layer_test = tf.layers.dense(inputs=tf.nn.l2_normalize(input_test, dim=1), units=5 * self.embed_dim,
                                           trainable=False,
                                           kernel_regularizer=regularizers.l2_regularizer(100.0), activation=tf.nn.tanh,
                                           name='hidden_layer', reuse=True)

            hidden_layer1_test = tf.layers.dense(inputs=tf.nn.l2_normalize(hidden_layer_test, dim=1), units=1 * self.embed_dim,
                                             trainable=False,
                                             kernel_regularizer=regularizers.l2_regularizer(100.0),
                                             activation=tf.nn.relu,
                                             name='hidden_layer1', reuse=True)
            attention_layer_score = (self.embed_dim*1.0/3) * tf.nn.softmax(hidden_layer1_test, dim=-1)

            # 物品邻居信息注意力
            hidden_layer_bitem_test = tf.layers.dense(inputs=tf.nn.l2_normalize(input_item_neis_test, dim=1), units=5 * self.embed_dim,
                                           trainable=False,
                                           kernel_regularizer=regularizers.l2_regularizer(100.0), activation=tf.nn.tanh,
                                           name='hidden_layer_bitem', reuse=True)

            hidden_layer1_bitem_test = tf.layers.dense(inputs=tf.nn.l2_normalize(hidden_layer_bitem_test, dim=1), units=1 * self.embed_dim,
                                             trainable=False,
                                             kernel_regularizer=regularizers.l2_regularizer(100.0),
                                             activation=tf.nn.relu,
                                             name='hidden_layer1_bitem', reuse=True)
            attention_layer_score_bitem = (self.embed_dim*1.0/3) * tf.nn.softmax(hidden_layer1_bitem_test, dim=-1)

            # 用户邻居信息注意力
            hidden_layer_buser_test = tf.layers.dense(inputs=tf.nn.l2_normalize(input_user_neis_test, dim=1), units=5 * self.embed_dim,
                                           trainable=False,
                                           kernel_regularizer=regularizers.l2_regularizer(100.0), activation=tf.nn.tanh,
                                           name='hidden_layer_buser', reuse=True)

            hidden_layer1_buser_test = tf.layers.dense(inputs=tf.nn.l2_normalize(hidden_layer_buser_test, dim=1), units=1 * self.embed_dim,
                                             trainable=False,
                                             kernel_regularizer=regularizers.l2_regularizer(100.0),
                                             activation=tf.nn.relu,
                                             name='hidden_layer1_buser', reuse=True)
            attention_layer_score_buser = (self.embed_dim*1.0/3) * tf.nn.softmax(hidden_layer1_buser_test, dim=-1)

        # (N_USER_IDS, N_ITEM, K)
        attention_layer_score_bitem=tf.tile(tf.expand_dims(attention_layer_score_bitem, 0), [tf.shape(user)[0],1,1])
        # (N_USER_IDS, N_ITEM, K)
        attention_layer_score_buser=tf.reshape(attention_layer_score_buser, [tf.shape(user)[0], tf.shape(item)[1], self.embed_dim])

        # 同样因子除以3,加一起
        # 简便地实现求平均值

        # (N_USER_IDS, N_ITEM, K)
        attention_reshape_test = tf.reshape(attention_layer_score, [-1, tf.shape(item)[1], self.embed_dim],name='attention_test')
        scores = -tf.reduce_sum(
            tf.squared_difference(tf.multiply(tf.add(tf.add(attention_reshape_test, attention_layer_score_buser),attention_layer_score_bitem), tf.tile(user, [1, tf.shape(item)[1], 1])), tf.multiply(tf.add(tf.add(attention_reshape_test, attention_layer_score_buser), attention_layer_score_bitem), item)), 2,
            name="scores")
        
        # 去掉了重复的top_k
        # top_n = tf.nn.top_k(scores, 10 + self.max_train_count,name='top_n')[0]
        # return top_n
        return scores

def optimize(model, sampler, train, valid, test, args, item_neighbors, user_neighbors, early_stopping_n=5):
    """
    Optimize the model. DONETODO: implement early-stopping
    :param model: model to optimize
    :param sampler: mini-batch sampler
    :param train: train user-item matrix
    :param valid: validation user-item matrix
    :return: None
    """
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    if model.feature_projection is not None:
        # initialize item embedding with feature projection
        sess.run(tf.assign(model.item_embeddings, model.feature_projection))
    # all test users to calculate recall validation
    valid_users=numpy.asarray(list(set(valid.nonzero()[0])),dtype=numpy.int32)
    test_users = numpy.asarray(list(set(test.nonzero()[0])),dtype=numpy.int32)
    validresult=RecallEvaluator(model,train,valid)
    testresult = RecallEvaluator(model, train, test)

    # 这里较为特殊
    # 每若干批数据训练后,进行一次评估,则记为一轮训练
    # 这是延续cml和maml的历史设计
    # cml为了提升速度,使用多进程并行地
    # 在训练集中取每一批数据
    # 所以难以界定何时便利了所有训练集数据
    # 并且cml使用的并行采样,抛弃最后构不成一批的数据
    # 不能按照传统的方式定义一轮训练
    epoch = 0
    # 用于early stopping的计数
    fail_cnt=0

    # best_ndcg=-100.0
    best_recall=-100.0
    # best_hr=-100.0
    # best_pr=-100.0
    saver=tf.train.Saver()
    
    while True:
        print('\nepochs:{}'.format(epoch),file=outputfile)
        epoch += 1
        # train model
        losses = []
        # run n mini-batches
        for _ in tqdm(range(args.eva_batches), desc="Optimizing..."):
            user_pos, neg = sampler.next_batch()
            # print("get next batch",datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])

            # (N+N*W, 1)
            all_item_ids=numpy.concatenate((user_pos[:,1],numpy.reshape(neg,(-1))), axis=0)
            # print("concat all item",datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])

            # 首先去随机数用于后面采样定位
            # 实验中所用的数据集中，用户物品数量都在1000000以内
            # 足够覆盖每一个邻居
            # 若在使用个别更大的数据集，可以相应调大数值
            all_item_neis_id_ranindex=numpy.random.randint(1000000,size=all_item_ids.shape[0])

            # 取模后访问列表的元素，
            # 这种实现方式是尝试的多种采样方法中最快的
            # 可以在几毫秒或几十毫秒内完成采样
            # 而其他方法则较慢
            # 采样用户邻居时，可以先转换为numpy的array，可以使用列表进行索引
            # 但是实际效果很慢，不如一个个取数
            # 最后输入到tensorflow中
            # (N+N*W, 1)
            all_item_neis_id_sample=[item_neighbors[i][ranindex % len(item_neighbors[i])] for i,ranindex in zip(all_item_ids,all_item_neis_id_ranindex)]
            # assert len(all_item_neis_id_sample)==all_item_ids.shape[0]
            # print("sample all item nei done",datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
            # assert len(all_item_neis_id_sample[0])==1
            
            # (N+N*W, 1)
            user_ids=numpy.concatenate((user_pos[:,0],numpy.tile(numpy.expand_dims(user_pos[:,0], 1), (1, args.num_neg)).flatten()), axis=0)
            user_neis_id_ranindex=numpy.random.randint(1000000,size=user_ids.shape[0])
            # (N+N*W, 1)
            user_neis_id_sample=[user_neighbors[i][ranindex % len(user_neighbors[i])] for i,ranindex in zip(all_item_ids,user_neis_id_ranindex)]
            # print("sample user nei done",datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
            # assert len(user_neis_id_sample[0])==1
            # raise NotImplementedError
            
            _, loss= sess.run((model.optimize, model.loss),
                               {model.user_positive_items_pairs: user_pos,
                                model.negative_samples: neg,
                                model.all_item_neis_id_sample: all_item_neis_id_sample,
                                model.user_neis_id_sample: user_neis_id_sample,})
            losses.append(loss)
        # 训练中的日志同时记录到标准输出和文件中
        print("\nTrain loss: {}".format(numpy.mean(losses)),file=outputfile)
        print("\nTrain loss: {}".format(numpy.mean(losses)))

        # 在验证集进行评估,仅使用recall
        # 其他指标为maml遗留,在评估过程中直接置零,不再计算,减少计算用时
        valid_recalls, valid_ndcg, valid_hr, valid_pr = [], [], [], []
        for user_chunk in toolz.partition_all(100, valid_users):
            recalls, ndcgs, hit_ratios, precisions = validresult.eval(sess, user_chunk, item_neighbors, user_neighbors)
            valid_recalls.extend(recalls)
            valid_ndcg.extend(ndcgs)
            valid_hr.extend(hit_ratios)
            valid_pr.extend(precisions)
        ndcg_mean=numpy.mean(valid_ndcg)
        recall_mean=numpy.mean(valid_recalls)
        hr_mean=numpy.mean(valid_hr)
        pr_mean=numpy.mean(valid_pr)
        print("\nresult on valid set: recall:{}".format(recall_mean),file=outputfile)
        print("\nresult on valid set: recall:{}".format(recall_mean))

        # 看是否在验证集上过拟合,如果在指定轮数后
        # 在验证集仍未有提升,则触发提前终止
        # 实验中指定的是10轮
        # 每次取得最佳效果后保存模型
        # 供后续在测试集上还原
        if recall_mean<=best_recall:
            fail_cnt+=1
        else:
            # best_ndcg=ndcg_mean
            best_recall=recall_mean
            # best_hr=hr_mean
            # best_pr=pr_mean
            fail_cnt=0
            saver.save(sess,os.path.join(os.getcwd(),"models_{:%Y%m%d_%H%M%S}/".format(nowdate),Filename+"_model.ckpt"))
            print("Best result!",file=outputfile)
            print("Best result!")
            # print(saver.last_checkpoints[-1])
        outputfile.flush()
        if fail_cnt>=early_stopping_n:
            break

    # 还原最佳的模型
    # 在测试集进行评估
    # saver.restore(sess, saver.last_checkpoints[-1])
    ckpt_state=tf.train.get_checkpoint_state("./models_20200604_000541")
    with open("test_pred.txt",'w') as ftest:
        saver.restore(sess, ckpt_state.model_checkpoint_path)
        test_recalls, test_ndcg, test_hr, test_pr = [], [], [], []
        for user_chunk in toolz.partition_all(100, test_users):
            recalls, ndcgs, hit_ratios, precisions = testresult.eval(sess, user_chunk, item_neighbors, user_neighbors,ftest=ftest)
            test_recalls.extend(recalls)
            test_ndcg.extend(ndcgs)
            test_hr.extend(hit_ratios)
            test_pr.extend(precisions)
        ndcg_mean=numpy.mean(test_ndcg)
        recall_mean=numpy.mean(test_recalls)
        hr_mean=numpy.mean(test_hr)
        pr_mean=numpy.mean(test_pr)
        print("\nresult on test set: recall:{}".format(recall_mean),file=outputfile)
        print("\nresult on test set: recall:{}".format(recall_mean))


def parse_args():
    parser = argparse.ArgumentParser(description='Run NDML.')
    parser.add_argument('--dataset', nargs='?',default='industrial', help='Choose a dataset.')
    parser.add_argument('--eva_batches', type=int,default=100, help = 'evaluation every n bathes.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--num_neg', type=int, default=4, help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--hidden_layer_dim', type=int, default=256, help='Hidden layer dim.')
    parser.add_argument('--margin', type=float, default=1.0, help='margin.' )
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout.' )
    parser.add_argument('--feature_l2_reg', type=float, default=1.0, help='feature_l2_reg')
    parser.add_argument('--embed_dim', type=int, default=100, help='embed_dim')
    parser.add_argument('--clip_norm', type=float, default=1.1, help='clip_norm')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning_rate')
    parser.add_argument('--feature_projection_scaling_factor', type=float, default=1.0, help='feature_projection_scaling_factor')
    parser.add_argument('--cov_loss_weight', type=float, default=1.0, help='cov_loss_weight')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # get user-item matrix
    # make feature as dense matrix
    # print("parse begin",datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
    args = parse_args()
    # print("parse args done",datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
    Filename = args.dataset
    Filepath = 'Data/'+ Filename
    
    # 这里读取数据集的数据并处理
    # 会耗费较多时间
    # 小的数据集能在1分钟内完成
    # 大的需要一两分钟
    dataset = Dataset.Dataset(Filepath)
    # print("dataset construct done",datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
    train, valid, test = dataset.trainMatrix, dataset.validMatrix, dataset.testRatings
    textualfeatures=dataset.textualfeatures
    print(dataset.allMatrix.shape)
    item_neighbors=dataset.item_neighborhood
    user_neighbors=dataset.user_neighborhood

    n_users, n_items = max(train.shape[0],test.shape[0]),max(train.shape[1],test.shape[1])
    print(n_users,n_items)

    # 记录日志
    # 同时直接记下模型参数,便于后续查看日志
    nowdate=datetime.now()
    outputfile=open("NDMLoutput-{}-{:%Y%m%d_%H%M%S}.txt".format(Filename,nowdate),mode='w')
    hyper=dict(vars(args))
    print(hyper)
    print(hyper,file=outputfile)

    # create warp sampler
    sampler = WarpSampler(train, batch_size=args.batch_size, n_negative=args.num_neg, check_negative=True)
    try:
        # 使用try语句预防出现异常
        # 在模型的编写调试过程中
        # 经常会程序崩溃
        # 延续cml所使用的多进程并
        # 行取训练数据
        # 突然崩溃极易导致新开的
        # 子进程因为失去父进程
        # 成为僵尸进程
        # 僵尸进程无法终止,并占满
        # CPU的一个线程,浪费硬件资源
        # 所以无论是否出现异常
        # 最后都关闭子进程

        model = NDML(n_users,
                    n_items,
                    # enable feature projection
                    textualfeatures=textualfeatures,
                    embed_dim=args.embed_dim,
                    batch_size=args.batch_size,
                    # N_negatvie
                    n_negative=args.num_neg,
                    margin=args.margin,
                    clip_norm=args.clip_norm,
                    master_learning_rate=args.learning_rate,
                    # the size of the hidden layer in the feature projector NN
                    hidden_layer_dim=args.hidden_layer_dim,
                    # dropout rate between hidden layer and output layer in the feature projector NN
                    dropout_rate=args.dropout,
                    # scale the output of the NN so that the magnitude of the NN output is closer to the item embedding
                    feature_projection_scaling_factor=args.feature_projection_scaling_factor,
                    # the penalty to the distance between projection and item's actual location in the embedding
                    # tune this to adjust how much the embedding should be biased towards the item features.
                    feature_l2_reg=args.feature_l2_reg,
                    # whether to enable rank weight. If True, the loss will be scaled by the estimated
                    # log-rank of the positive items. If False, no weight will be applied.
                    # This is particularly useful to speed up the training for large item set.
                    # Weston, Jason, Samy Bengio, and Nicolas Usunier.
                    # "Wsabie: Scaling up to large vocabulary image annotation." IJCAI. Vol. 11. 2011.
                    use_rank_weight=True,
                    # whether to enable covariance regularization to encourage efficient use of the vector space.
                    # More useful when the size of embedding is smaller (e.g. < 20 ).
                    use_cov_loss=True,
                    # weight of the cov_loss
                    cov_loss_weight=args.cov_loss_weight
                    )
        optimize(model, sampler, train, valid, test,args, item_neighbors, user_neighbors, early_stopping_n=10)
    except KeyboardInterrupt as keyb:
        print("I caught ctrl c.")
        # sampler.close()
        raise keyb
    except Exception as e:
        print("I caught it.")
        # sampler.close()
        raise e
    else:
        print("Everything is ok!")
        # sampler.close()
    finally:
        # 便于查看实验数据,再次输出模型参数
        # 关闭日志文件
        # 最重要的是关闭子进程
        # 预防出现僵尸进程
        # 加入这一步后,代码运行过程中再也
        # 没有出现过僵尸进程
        print(hyper)
        print(hyper,file=outputfile)
        outputfile.close()
        sampler.close()
