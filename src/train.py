import numpy as np
import os
import tensorflow as tf
import sys
sys.path.append('../src')

from data import Data
from trainer import Trainer
import logging
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001, help="set learning rate")
parser.add_argument('--batch_size', type=int, default=256, help="set batch size")
parser.add_argument('--epochs', type=int, default=100, help="set epoch")
parser.add_argument('--r1', type=int, default=0.5, help="negative samples ratio: 0-1")
parser.add_argument('--save_freq', type=int, default=2)
parser.add_argument('--sentence_length', type=int, default=40)
parser.add_argument('--dim', type=int, default=128, help="set dimension")
parser.add_argument('--model_sav_path', type=str, default = "./save_model/model.bin", help = "model save path")
parser.add_argument('--data_sav_path', type=str, default = "./save_model/data.bin", help = "data save path")
parser.add_argument('--margin', type=int, default = 4)
parser.add_argument('--lambda1', type=int, default = 1, help = "orthogonal matrix")
parser.add_argument('--lambda2', type=int, default = 0)
parser.add_argument('--lambda3', type=int, default = 0)
parser.add_argument('--gpu', type=str, default = "1")
parser.add_argument('--data_path', type=str, help = "data as a pickle")
parser.add_argument('--options_file', type=str, help = "model options path")
parser.add_argument('--weight_file', type=str, help = "model weight path")
parser.add_argument('--vocab_file', type=str, help = "model vocab path")
parser.add_argument('--token_emb_file', type=str, help = "token embedding file")
#parser.add_argument('--restore', bool=False, help = "restore any model")
parser.add_argument('--restore_path', type=str, help = "model restored path")
parser.add_argument('--log_path', type=str, default = "./save_model", help ="log path")



args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler("{0}/{1}.log".format(args.log_path, "result")),
        logging.StreamHandler()
    ])

this_data = Data(args.sentence_length)
this_data.load(args.data_path)
m_train = Trainer()
m_train.build(args.data_path, args.options_file, args.weight_file, args.token_emb_file, m1 = args.margin, m2 = 0, a1 =args.lambda1, a2 = args.lambda2, a3 = args.lambda3, length=args.sentence_length, dim=args.dim, batch_sizeK=args.batch_size, save_path=args.model_sav_path, data_save_path=args.data_sav_path)
m_train.train(epochs=args.epochs, save_every_epoch=args.save_freq, lr=args.lr, r1 =args.r1, restore=args.restore, restore_path=args.restore_path)


