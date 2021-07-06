''' Module for training TF parts.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time
import random
import pickle

 
from data import Data
import model as model
import sys
sys.path.append('../bilm_tf')
from bilm import BidirectionalLanguageModel, weight_layers
import logging

logger = logging.getLogger(__name__)

def neg_fun(dpkl, para_tuple):
        s1 = para_tuple[0]
        s1_index = para_tuple[2]
        s2 = para_tuple[1]
        s2_index = para_tuple[3]
        s1_token = dpkl['id2sent'][s1][s1_index]
        s2_token = dpkl['id2sent'][s2][s2_index]
        if((s1_token, s2_token) in dpkl['token_pair2neg_tuples']):
            neg_tuple_id = random.choice(list(dpkl['token_pair2neg_tuples'][(s1_token, s2_token)]))
            neg_tuple = dpkl['neg_tuples'][neg_tuple_id]
            return neg_tuple
        else:
            return None

def is_paraphrase(dpkl, sent_id1, sent_id2):
        if((sent_id1,sent_id2) in dpkl['paraphrases']):
            return True
        else:
            return False

def corrupt(dpkl, para_tuple, tar=None):
        # corrupt para tuple into a negative sample. Return (sent_id, sent_id, index_of_an_overlapping/synonym_token, index_of_an_overlapping/synonym_token) for a negative sample.
        if tar == None:
            tar = random.randint(0,1)
        s1 = para_tuple[0]
        s1_index = para_tuple[2]
        s2 = para_tuple[1]
        s2_index = para_tuple[3]

        if(tar == 0):
            token = dpkl['id2sent'][s1][s1_index]
            sents_list = dpkl['token2sents'][token]

            if ((s1, s1_index) in sents_list):
                sents_list.remove((s1, s1_index))
            if ((s2, s2_index) in sents_list):
                sents_list.remove((s2, s2_index))
            if (len(sents_list) == 0):
                return random.choice(dpkl['neg_tuples'])
            else:
                corrupt_s = random.choice(list(sents_list))
            ind = 0
            while is_paraphrase(dpkl, corrupt_s[0], s1):
                corrupt_s = random.choice(list(sents_list))
                ind += 1
                if ind > 10:
                    # print("ind", ind)
                    random.choice(dpkl['neg_tuples'])
                    break
            return (corrupt_s[0],s1,corrupt_s[1], s1_index)

        if(tar == 1):
            token = dpkl['id2sent'][s2][s2_index]
            sents_list = dpkl['token2sents'][token]

            if ((s1, s1_index) in sents_list):
                sents_list.remove((s1, s1_index))
            if ((s2, s2_index) in sents_list):
                sents_list.remove((s2, s2_index))
            if (len(sents_list) < 2):
                return random.choice(dpkl['neg_tuples'])
            else:
                corrupt_s = random.choice(list(sents_list))
            ind = 0
            while is_paraphrase(dpkl,corrupt_s[0], s2):
                corrupt_s = random.choice(list(sents_list))
                ind += 1
                if ind > 10:
                    # print("ind", ind)
                    random.choice(dpkl['neg_tuples'])
                    break
            c_tuple = (corrupt_s[0],s2,corrupt_s[1],s2_index)
            return c_tuple

class Trainer(object):
    def __init__(self):
        self.batch_sizeK=1024
        # self.dim=64
        self.length=20
        self._m1 = 0.5
        self._a1 = 0.5
        self.d = None
        self.tf_parts = None
        self.save_path = 'this-model.ckpt'
        self.data_save_path = 'this-data.bin'
        self.M1_path = None
        self.r1 = 0.5
        self.sess = None

    def build(self, data1, options_file, weight_file, token_embedding_file, m1, m2, a1, a2, a3, length=20, dim=128, batch_sizeK=1024, save_path = 'this-model.ckpt', data_save_path = 'this-data.bin', M1_path = None):
        self.data = Data
        dpkl = pickle.load(open(data1,'rb'))
        self.dpkl = dpkl
        self.para_tuples = dpkl['para_tuples']
  
        print(self.data)
        self.dim = dim
        self.length  = length
        self.batch_sizeK = batch_sizeK
        self.data_save_path = data_save_path
        self.save_path = save_path
        self.M1_path = M1_path
        self.bilm = BidirectionalLanguageModel(
            options_file,
            weight_file,
            use_character_inputs=False,
            embedding_weight_file=token_embedding_file,
            max_batch_size = 512)

        self.tf_parts = model.TFParts(m1, m2, a1, a2, a3, self.bilm, length, dim, token_embedding_file, batch_sizeK)



        # self.sess = sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5)))
        # self.sess.run(tf.initialize_all_variables())

    def gen_batch(self, r1, forever=False, shuffle=True):
        data = self.data
        l = len(self.para_tuples)
        while True:
            para_tuples = self.para_tuples
            if shuffle:
                np.random.shuffle(para_tuples)
            for i in range(0, l, self.batch_sizeK):
                batch = para_tuples[i: i+self.batch_sizeK, :]
                if batch.shape[0] < self.batch_sizeK:
                    batch = np.concatenate((batch, self.para_tuples[:self.batch_sizeK - batch.shape[0]]), axis=0)
                    assert batch.shape[0] == self.batch_sizeK
                #get negative samples, or corrupt a batch of positive cases data.corrupt_batch(batch)
                neg_batch = []
                for tup in batch:
                    # # for testing: no corrupt, all neg tuples:
                    # neg_tuple = random.choice(self.data.neg_tuples)
                    # neg_batch.append(neg_tuple)
                    neg = neg_fun(self.dpkl,tup)
                    if (random.uniform(0, 1) < r1) and (neg is not None):
                        # append a given negative case, or if None, corrupt
                        neg_batch.append(neg)
                    else:
                        # append a corrupted negative case
                        neg_batch.append(corrupt(self.dpkl,tup))

                neg_batch = np.array(neg_batch)
                id2sent = self.dpkl['id2sent']
                sent1_batch = np.take(id2sent,batch[:,0], axis = 0)
                sent2_batch = np.take(id2sent,batch[:,1], axis = 0)
                nsent1_batch = np.take(id2sent,neg_batch[:,0], axis = 0)
                nsent2_batch = np.take(id2sent,neg_batch[:,1], axis = 0)
                token1_batch = batch[:,2]
                token2_batch = batch[:,3]
                ntoken1_batch = neg_batch[:,2]
                ntoken2_batch = neg_batch[:,3]
                assert not np.any(np.isnan(sent1_batch))
                assert not np.any(np.isnan(sent2_batch))
                assert not np.any(np.isnan(nsent1_batch))
                assert not np.any(np.isnan(nsent2_batch))
                assert not np.any(np.isnan(token1_batch))
                assert not np.any(np.isnan(token2_batch))
                assert not np.any(np.isnan(ntoken1_batch))
                assert not np.any(np.isnan(ntoken2_batch))
                yield sent1_batch.astype(np.int64), sent2_batch.astype(np.int64), nsent1_batch.astype(np.int64), nsent2_batch.astype(np.int64), \
                      token1_batch.astype(np.int64), token2_batch.astype(np.int64),ntoken1_batch.astype(np.int64),ntoken2_batch.astype(np.int64)

            if not forever:
                break

    def train1epoch(self, sess, num_A_batch,lr, r1, epoch):
        this_gen_batch = self.gen_batch(r1, forever=True)
        this_loss = []
        this_pos = []
        this_neg = []

        for batch_id in range(num_A_batch):
            # call train op
            # sent1_batch, sent2_batch, nsent1_batch,nsent2_batch,token1_batch,token2_batch,ntoken1_batch,ntoken2_batch = next(this_gen_batch)
            # _, loss, A_loss, B_loss = sess.run([self.tf_parts._train_op, self.tf_parts._loss, self.tf_parts.A_loss,self.tf_parts.B_loss],
            #                    feed_dict={self.tf_parts._sent1: sent1_batch,
            #                    self.tf_parts._sent2: sent2_batch,
            #                    self.tf_parts._nsent1: nsent1_batch,
            #                    self.tf_parts._nsent2: nsent2_batch,
            #                    self.tf_parts._token1: token1_batch,
            #                    self.tf_parts._token2: token2_batch,
            #                    self.tf_parts._ntoken1: ntoken1_batch,
            #                    self.tf_parts._ntoken2: ntoken2_batch,
            #                    self.tf_parts._lr: lr})

            sent1_batch, sent2_batch, nsent1_batch, nsent2_batch, token1_batch, token2_batch, ntoken1_batch, ntoken2_batch = next(
                this_gen_batch)
            # swj
            _, loss, = sess.run([self.tf_parts._train_op, self.tf_parts._loss], \
                           feed_dict={self.tf_parts._sent1: sent1_batch,
                                      self.tf_parts._sent2: sent2_batch,
                                      self.tf_parts._nsent1: nsent1_batch,
                                      self.tf_parts._nsent2: nsent2_batch,
                                      self.tf_parts._token1: token1_batch,
                                      self.tf_parts._token2: token2_batch,
                                      self.tf_parts._ntoken1: ntoken1_batch,
                                      self.tf_parts._ntoken2: ntoken2_batch,
                                      self.tf_parts._lr: lr})

            if np.isnan(loss):
                print("loss", loss)
                print("Training collapsed.")
                sys.exit(1)
            this_loss.append(np.array(loss))

            if ((batch_id + 1) % 500 == 0 or batch_id == num_A_batch - 1):
                print('\rprocess pair: %d / %d. Epoch %d' % (batch_id+1, num_A_batch+1, epoch))
        print("finish one epoch")
        this_total_loss = np.mean(this_loss)

        logging.info("Loss of epoch %d = %s" % (epoch, np.mean(this_loss)))

        # print("Loss of epoch %d = %s, A_loss: %s, B_loss: %s" % (epoch, np.sum(this_total_loss), np.sum(this_total_A_loss), np.sum(this_total_B_loss)))
        return this_total_loss

    # Call train1epoch #epoch times
    def train(self, epochs=20, save_every_epoch=10, lr=0.001, r1 = 0.5, restore=False, restore_path=None):
        sess = tf.Session()
        if (restore == True):
            self.tf_parts._saver.restore(sess, restore_path)  # load it
        else:
            sess.run(tf.global_variables_initializer())
        for v in tf.trainable_variables():
            print(v)
        # 1/0
        num_A_batch = int(len(self.para_tuples)/self.batch_sizeK) + 1
        # num_A_batch = len(list(self.gen_batch(r1)))
        # print(len(self.data.para_tuples), self.batch_sizeK)
        print('batch =', num_A_batch)
        # num_A_batch = 1

        t0 = time.time()
        for epoch in range(epochs):
            epoch_loss = self.train1epoch(sess, num_A_batch, lr, r1, epoch + 1)
            print("Time use: %d" % (time.time() - t0))
            if np.isnan(epoch_loss):
                print("Training collapsed.")
                return
            # save + evaluate
            if (epoch + 1) % save_every_epoch == 0:
                this_save_path = self.tf_parts._saver.save(sess, self.save_path)
                self.data.save(self.data_save_path)
                print("Model saved in file: %s. Data saved in file: %s" % (this_save_path, self.data_save_path))



        this_save_path = self.tf_parts._saver.save(sess, self.save_path)
        self.data.save(self.data_save_path)
        print("Model saved in file: %s. Data saved in file: %s" % (this_save_path, self.data_save_path))
        sess.close()
        print("Done")



# A safer loading is available in Tester, with parameters like batch_size and dim recorded in the corresponding Data component
def load_tfparts(data, dim=64, batch_sizeK=1024,save_path = 'this-model.ckpt'):
    tf_parts = model.TFParts("""Add agruments""")
    #with tf.Session() as sess:
    sess = tf.Session()
    tf_parts._saver.restore(sess, save_path)