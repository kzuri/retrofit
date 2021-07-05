'''Model for holding TF parts. etc.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import pickle
import h5py
import sys
sys.path.append('../bilm_tf')
from bilm import BidirectionalLanguageModel, weight_layers


# Orthogonal Initializer from
# https://github.com/OlavHN/bnlstm

e = 10**-8

class TFParts(object):
    '''TensorFlow-related things.
    This is to keep TensorFlow-related components in a neat shell.
    '''
    def __init__(self, m1, m2, a1, a2, a3, bilm_model = None, length = 30, dim = 128, token_embedding_file = None, batch_sizeK=1024):
        self._dim = dim  # dimension of both relation and ontology. 
        self._length = length
        self._batch_sizeK = batch_sizeK
        self._epoch_loss = 0
        self._bilm = bilm_model
        self.token_embedding_file = token_embedding_file
        # margins
        self._m1 = m1
        self._m2 = m2
        # hyperparameter
        self._a1 = a1 #coefficient on the constraint loss
        self._a2 = a2
        self._a3 = a3
        self.build()

    @property
    def dim(self):
        return self._dim

    @property
    def length(self):
        return self._length

    @property
    def batch_size(self):
        return self._batch_sizeK   

    def build(self):
        # load weights
        with h5py.File(self.token_embedding_file, 'r') as fin:
            _n_tokens_vocab = fin['embedding'].shape[0] + 1
            embed_weights = fin['embedding'][...]
            weights = np.zeros((embed_weights.shape[0] + 1, embed_weights.shape[1]), dtype='float32')
            weights[1:, :] = embed_weights
            print("model-load: weights")
        
        tf.reset_default_graph()
        # with tf.variable_scope("graph"):
            # Variables (matrix of embeddings/transformations)
        self._M1 = M1 = tf.get_variable(
            name='M1',
            dtype=tf.float32,
            initializer = tf.eye(self._dim, dtype="float32"),
            trainable = True,
            # shape=[self.dim, self.dim],
            # initializer=tf.initializers.orthogonal(dtype="float32")
        )
        self._M2 = M2 = tf.get_variable(# This is another transformation to be added after ELMo.
            name='M2',
            dtype=tf.float32,
            initializer= tf.eye(self._dim, dtype="float32")
        )

        # self._b1 = b1 = tf.get_variable(
        #     name='b1',
        #     dtype=tf.float32,
        #     shape=[1, self.dim]
        #     # initializer=tf.initializers.orthogonal(dtype="float32")
        # )
        print("_n_tokens_vocab",_n_tokens_vocab)
        self.embedding_table = embedding_table = tf.get_variable(
            name='embedding_table',
            shape=[_n_tokens_vocab, self.dim],
            initializer=tf.constant_initializer(weights),
            trainable=False,
            dtype=tf.float32)

        self._sent1 = sent1 = tf.placeholder(
            dtype=tf.int64,
            shape=[self._batch_sizeK, self._length],
            name='sent1')
        self._sent2 = sent2 = tf.placeholder(
            dtype=tf.int64,
            shape=[self._batch_sizeK, self._length],
            name='sent2')
        self._nsent1 = nsent1 = tf.placeholder(
            dtype=tf.int64,
            shape=[self._batch_sizeK, self._length],
            name='nsent1')
        self._nsent2 = nsent2 = tf.placeholder(
            dtype=tf.int64,
            shape=[self._batch_sizeK, self._length],
            name='nsent2')

        self._token1 = token1 = tf.placeholder(
            dtype=tf.int64,
            shape=[self._batch_sizeK],
            name='token1')
        self._token2 = token2 = tf.placeholder(
            dtype=tf.int64,
            shape=[self._batch_sizeK],
            name='token2')
        self._ntoken1 = ntoken1 = tf.placeholder(
            dtype=tf.int64,
            shape=[self._batch_sizeK],
            name='ntoken1')
        self._ntoken2 = ntoken2 = tf.placeholder(
            dtype=tf.int64,
            shape=[self._batch_sizeK],
            name='ntoken2')
        nindex = tf.range(self._batch_sizeK, dtype=tf.int64)


        # 1. Look up embeddings of training cases and negative cases, go through M1, go through bilm, and calculate hinge loss.
        sent1_emb = tf.nn.embedding_lookup(embedding_table, sent1)  # [s1:[[w1],[w2]...],s2:[[w1],[w2]...]]
        sent2_emb = tf.nn.embedding_lookup(embedding_table, sent2)
        nsent1_emb = tf.nn.embedding_lookup(embedding_table, nsent1) #[s1:[[w1],[w2]...],s2:[[w1],[w2]...]]
        nsent2_emb = tf.nn.embedding_lookup(embedding_table, nsent2)
        print("sent1_emb", sent1_emb.shape)
        # transformation:
        sent1_emb_transform = tf.matmul(tf.reshape(sent1_emb, [-1, self._dim]), self._M1)
        #sent1_emb_transform = tf.add(sent1_emb_transform, b1)
        sent1_emb_transform = tf.reshape(sent1_emb_transform, [self._batch_sizeK, self._length, self._dim])
        sent2_emb_transform = tf.matmul(tf.reshape(sent2_emb, [-1, self._dim]), self._M1)
        #sent2_emb_transform = tf.add(sent2_emb_transform, b1)
        sent2_emb_transform = tf.reshape(sent2_emb_transform, [self._batch_sizeK, self._length, self._dim])

        nsent1_emb_transform = tf.matmul(tf.reshape(nsent1_emb, [-1,self._dim]),self._M1)
        #nsent1_emb_transform = tf.add(nsent1_emb_transform, b1)
        nsent1_emb_transform = tf.reshape(nsent1_emb_transform, [self._batch_sizeK,self._length,self._dim])
        nsent2_emb_transform = tf.matmul(tf.reshape(nsent2_emb, [-1, self._dim]), self._M1)
        #nsent2_emb_transform = tf.add(nsent2_emb_transform, b1)
        nsent2_emb_transform = tf.reshape(nsent2_emb_transform, [self._batch_sizeK,self._length,self._dim])

        print("----")
        # run through bilm
        sent1_emb_context = self._bilm(sent1, sent1_emb_transform)
        with tf.variable_scope('', reuse=True):
            sent2_emb_context = self._bilm(sent2, sent2_emb_transform)
        # dim x 2
        self.sent1_emb_context_output = sent1_emb_context_output = weight_layers('output1', sent1_emb_context, l2_coef=0.0)
        # print("sent1_emb_context_output",sent1_emb_context_output)
        with tf.variable_scope('', reuse=True):
            self.sent2_emb_context_output = sent2_emb_context_output = weight_layers('output1', sent2_emb_context, l2_coef=0.0)['weighted_op']
        # print("sent2_emb_context_output",sent2_emb_context_output)

        with tf.variable_scope('', reuse=True):
            nsent1_emb_context = self._bilm(nsent1, nsent1_emb_transform)
            nsent2_emb_context = self._bilm(nsent2, nsent2_emb_transform)
        print("----")
        # self.nsent1_emb_context_output = nsent1_emb_context_output = weight_layers('output3', nsent1_emb_context, l2_coef=0.0)['weighted_op']
        # self.nsent2_emb_context_output = nsent2_emb_context_output = weight_layers('output4', nsent2_emb_context, l2_coef=0.0)['weighted_op']

        with tf.variable_scope('', reuse=True):
            self.nsent1_emb_context_output = nsent1_emb_context_output = weight_layers('output1', nsent1_emb_context, l2_coef=0.0)['weighted_op']
            self.nsent2_emb_context_output = nsent2_emb_context_output = weight_layers('output1', nsent2_emb_context, l2_coef=0.0)['weighted_op']

        # DEBUg: check if sentence embedding close
        sent_1_avg = tf.reduce_mean(sent1_emb_context_output, -2)
        sent_2_avg = tf.reduce_mean(sent2_emb_context_output, -2)
        self.para_dis = para_dis = tf.norm(tf.add(tf.subtract(sent_1_avg, sent_2_avg), e), axis=-1, ord=2)
        print("nsent1_emb_context_output", nsent1_emb_context_output.shape)
        nsent_1_avg = tf.reduce_mean(nsent1_emb_context_output, -2)
        nsent_2_avg = tf.reduce_mean(nsent2_emb_context_output, -2)
        self.npara_dis = npara_dis = tf.norm(tf.add(tf.subtract(nsent_1_avg, nsent_2_avg), e), axis=-1, ord=2)

        # extract word pair embedding
        print('token1.shape', token1.shape)
        print('nindex.shape', nindex.shape)
        token1 = tf.stack([nindex, token1], axis=1)  # [[0, index],[1,index],.....]
        print('token1.shape', token1.shape)
        token2 = tf.stack([nindex, token2], axis=1)
        token1_emb = tf.gather_nd(sent1_emb_context_output, token1)  # [[emb1], [emb2], ..... ]
        print('token1_emb', token1_emb.shape)
        token2_emb = tf.gather_nd(sent2_emb_context_output, token2)

        ntoken1 = tf.stack([nindex,ntoken1],axis = 1) #[[0, index],[1,index],.....]
        ntoken2 = tf.stack([nindex,ntoken2],axis = 1)
        print("nsent1_emb_context_output", nsent1_emb_context_output.shape)
        ntoken1_emb = tf.gather_nd(nsent1_emb_context_output,ntoken1) #[[emb1], [emb2], ..... ]
        ntoken2_emb = tf.gather_nd(nsent2_emb_context_output,ntoken2)

        self.token1_emb = token1_emb
        self.sent1_emb_context_output = sent1_emb_context_output

        self.pos_matrix = pos_matrix = tf.norm(tf.add(tf.subtract(token1_emb, token2_emb), e), axis=-1, ord=2)
        self.neg_matrix = neg_matrix = tf.norm(tf.add(tf.subtract(ntoken1_emb, ntoken2_emb), e), axis=-1, ord=2)
        # self.pos_matrix = pos_matrix = tf.math.squared_difference(token1_emb, token2_emb)
        # self.neg_matrix = neg_matrix = tf.math.squared_difference(ntoken1_emb, ntoken2_emb)
        self.A_loss_matrix = A_loss_matrix = tf.maximum(tf.subtract(tf.add(pos_matrix, self._m1), neg_matrix), 0.)
        self.A_loss_aux = A_loss_aux = self._a3 * tf.reduce_mean(tf.maximum(tf.subtract(pos_matrix, self._m2), 0.))
        self.A_loss = A_loss = tf.reduce_mean(A_loss_matrix) + A_loss_aux
        # print("self._m1", self._m1)

        # 2. Constraint loss @Muhao
        # identity = tf.Variable(tf.eye(self._dim, dtype=tf.float32), trainable = False)
        identity = tf.eye(self._dim, dtype=tf.float32)
        # self.norm = tf.norm(self.det_M1_sq_sub)
        self.B_loss = B_loss = self._a1 * tf.norm(tf.add(tf.subtract(identity, tf.matmul(self._M1, self._M1, transpose_b=True)), e), ord=2)

        self.C_loss = C_loss = self._a2 * tf.reduce_mean(tf.maximum(tf.subtract(tf.add(para_dis, self._m1), npara_dis), 0.))
        #self.C_loss = C_loss = self._a2 * tf.add(tf.reduce_mean(tf.maximum(tf.subtract(tf.add(para_dis, self._m1), npara_dis), 0.)), tf.reduce_mean(tf.maximum(tf.subtract(para_dis, self._m2), 0.)))

        self._loss = _loss = self.A_loss + self.B_loss + self.C_loss



        # Optimizer
        self._lr = lr = tf.placeholder(tf.float32)
        self._opt = opt = tf.train.AdamOptimizer(lr) #AdagradOptimizer(lr)#GradientDescentOptimizer(lr) #AdamOptimizer(lr)
        # self._train_op_A = train_op_A = opt.minimize(A_loss)
        # self._train_op_B = train_op_B = opt.minimize(B_loss)
        self._train_op = opt.minimize(_loss)






