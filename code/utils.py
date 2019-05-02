import os
import errno
import numpy as np
import glob
import pickle

from copy import deepcopy
from config import cfg

from torch.nn import init
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F
import torchvision.utils as vutils


def save_model(model, optim, iter, model_dir, max_to_keep=None, model_name=""):
    checkpoint = {
        'iter': iter,
        'model': model.state_dict(),
        'optim': optim.state_dict() if optim is not None else None}
    if model_name == "":
        torch.save(checkpoint, "{}/checkpoint_{:06}.pth".format(model_dir, iter))
    else:
        torch.save(checkpoint, "{}/{}_checkpoint_{:06}.pth".format(model_dir, model_name, iter))

    if max_to_keep is not None and max_to_keep > 0:
        checkpoint_list = sorted([ckpt for ckpt in glob.glob(model_dir + "/" + '*.pth')])
        while len(checkpoint_list) > max_to_keep:
            os.remove(checkpoint_list[0])
            checkpoint_list = checkpoint_list[1:]


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def init_modules(modules, w_init='kaiming_uniform'):
    if w_init == "normal":
        _init = init.normal_
    elif w_init == "xavier_normal":
        _init = init.xavier_normal_
    elif w_init == "xavier_uniform":
        _init = init.xavier_uniform_
    elif w_init == "kaiming_normal":
        _init = init.kaiming_normal_
    elif w_init == "kaiming_uniform":
        _init = init.kaiming_uniform_
    elif w_init == "orthogonal":
        _init = init.orthogonal_
    else:
        raise NotImplementedError
    for m in modules:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            _init(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        if isinstance(m, (nn.LSTM, nn.GRU)):
            for name, param in m.named_parameters():
                if 'bias' in name:
                    nn.init.zeros_(param)
                elif 'weight' in name:
                    _init(param)


def load_vocab(cfg):
    def invert_dict(d):
        return {v: k for k, v in d.items()}

    with open(os.path.join(cfg.DATASET.DATA_DIR, 'dic.pkl'), 'rb') as f:
        dictionaries = pickle.load(f)
    vocab = {}
    vocab['question_token_to_idx'] = dictionaries["word_dic"]
    vocab['answer_token_to_idx'] = dictionaries["answer_dic"]
    vocab['question_token_to_idx']['pad'] = 0
    vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
    vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])

    return vocab


def generateVarDpMask(shape, keepProb):
    randomTensor = torch.tensor(keepProb).cuda().expand(shape)
    randomTensor += nn.init.uniform_(torch.cuda.FloatTensor(shape[0], shape[1]))
    binaryTensor = torch.floor(randomTensor)
    mask = torch.cuda.FloatTensor(binaryTensor)
    return mask


def applyVarDpMask(inp, mask, keepProb):
    ret = (torch.div(inp, torch.tensor(keepProb).cuda())) * mask
    return ret