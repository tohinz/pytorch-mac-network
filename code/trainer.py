from __future__ import print_function

import sys
import os
import shutil
from six.moves import range
import pprint
from tqdm import tqdm

from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import mkdir_p, save_model, load_vocab
from datasets import ClevrDataset, collate_fn
import mac


class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        sys.stdout.flush()
        self.log.flush()

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass


class Trainer():
    def __init__(self, log_dir, cfg):

        self.path = log_dir
        self.cfg = cfg

        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(self.path, 'Model')
            self.log_dir = os.path.join(self.path, 'Log')
            mkdir_p(self.model_dir)
            mkdir_p(self.log_dir)
            self.writer = SummaryWriter(log_dir=self.log_dir)
            sys.stdout = Logger(logfile=os.path.join(self.path, "logfile.log"))

        self.data_dir = cfg.DATASET.DATA_DIR
        self.max_epochs = cfg.TRAIN.MAX_EPOCHS
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.lr = cfg.TRAIN.LEARNING_RATE

        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True

        # load dataset
        self.dataset = ClevrDataset(data_dir=self.data_dir, split="train")
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                                       num_workers=cfg.WORKERS, drop_last=True, collate_fn=collate_fn)

        self.dataset_val = ClevrDataset(data_dir=self.data_dir, split="val")
        self.dataloader_val = DataLoader(dataset=self.dataset_val, batch_size=200, drop_last=True,
                                         shuffle=False, num_workers=cfg.WORKERS, collate_fn=collate_fn)

        # load model
        self.vocab = load_vocab(cfg)
        self.model, self.model_ema = mac.load_MAC(cfg, self.vocab)
        self.weight_moving_average(alpha=0)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.previous_best_acc = 0.0
        self.previous_best_epoch = 0

        self.total_epoch_loss = 0
        self.prior_epoch_loss = 10

        self.print_info()
        self.loss_fn = torch.nn.CrossEntropyLoss().cuda()

    def print_info(self):
        print('Using config:')
        pprint.pprint(self.cfg)
        print("\n")

        pprint.pprint("Size of dataset: {}".format(len(self.dataset)))
        print("\n")

        print("Using MAC-Model:")
        pprint.pprint(self.model)
        print("\n")

    def weight_moving_average(self, alpha=0.999):
        for param1, param2 in zip(self.model_ema.parameters(), self.model.parameters()):
            param1.data *= alpha
            param1.data += (1.0 - alpha) * param2.data

    def set_mode(self, mode="train"):
        if mode == "train":
            self.model.train()
            self.model_ema.train()
        else:
            self.model.eval()
            self.model_ema.eval()

    def reduce_lr(self):
        epoch_loss = self.total_epoch_loss / float(len(self.dataset) // self.batch_size)
        lossDiff = self.prior_epoch_loss - epoch_loss
        if ((lossDiff < 0.015 and self.prior_epoch_loss < 0.5 and self.lr > 0.00002) or \
            (lossDiff < 0.008 and self.prior_epoch_loss < 0.15 and self.lr > 0.00001) or \
            (lossDiff < 0.003 and self.prior_epoch_loss < 0.10 and self.lr > 0.000005)):
            self.lr *= 0.5
            print("Reduced learning rate to {}".format(self.lr))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
        self.prior_epoch_loss = epoch_loss
        self.total_epoch_loss = 0

    def save_models(self, iteration):
        save_model(self.model, self.optimizer, iteration, self.model_dir, model_name="model")
        save_model(self.model_ema, None, iteration, self.model_dir, model_name="model_ema")

    def train_epoch(self, epoch):
        cfg = self.cfg
        avg_loss = 0
        train_accuracy = 0

        self.labeled_data = iter(self.dataloader)
        self.set_mode("train")

        dataset = tqdm(self.labeled_data)

        for data in dataset:
            ######################################################
            # (1) Prepare training data
            ######################################################
            image, question, question_len, answer = data['image'], data['question'], data['question_length'], data['answer']
            answer = answer.long()
            question = Variable(question)
            answer = Variable(answer)

            if cfg.CUDA:
                image = image.cuda()
                question = question.cuda()
                answer = answer.cuda().squeeze()
            else:
                question = question
                image = image
                answer = answer.squeeze()

            ############################
            # (2) Train Model
            ############################
            self.optimizer.zero_grad()

            scores = self.model(image, question, question_len)
            loss = self.loss_fn(scores, answer)
            loss.backward()

            if self.cfg.TRAIN.CLIP_GRADS:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.TRAIN.CLIP)

            self.optimizer.step()
            self.weight_moving_average()

            ############################
            # (3) Log Progress
            ############################
            correct = scores.detach().argmax(1) == answer
            accuracy = correct.sum().cpu().numpy() / answer.shape[0]

            if avg_loss == 0:
                avg_loss = loss.item()
                train_accuracy = accuracy
            else:
                avg_loss = 0.99 * avg_loss + 0.01 * loss.item()
                train_accuracy = 0.99 * train_accuracy + 0.01 * accuracy
            self.total_epoch_loss += loss.item()

            dataset.set_description(
                'Epoch: {}; Avg Loss: {:.5f}; Avg Train Acc: {:.5f}'.format(epoch + 1, avg_loss, train_accuracy)
            )

        dict = {
            "avg_loss": avg_loss,
            "train_accuracy": train_accuracy
        }
        return dict

    def train(self):
        cfg = self.cfg
        print("Start Training")
        for epoch in range(self.max_epochs):
            dict = self.train_epoch(epoch)
            self.reduce_lr()
            self.log_results(epoch, dict)
            if cfg.TRAIN.EALRY_STOPPING:
                if epoch - cfg.TRAIN.PATIENCE == self.previous_best_epoch:
                    break

        self.save_models(self.max_epochs)
        self.writer.close()
        print("Finished Training")
        print("Highest validation accuracy: {} at epoch {}")

    def log_results(self, epoch, dict, max_eval_samples=None):
        epoch += 1
        self.writer.add_scalar("avg_loss", dict["avg_loss"], epoch)
        self.writer.add_scalar("train_accuracy", dict["train_accuracy"], epoch)

        val_accuracy, val_accuracy_ema = self.calc_accuracy("validation", max_samples=max_eval_samples)
        self.writer.add_scalar("val_accuracy_ema", val_accuracy_ema, epoch)
        self.writer.add_scalar("val_accuracy", val_accuracy, epoch)

        print("Epoch: {}\tVal Acc: {},\tVal Acc EMA: {},\tAvg Loss: {},\tLR: {}".
              format(epoch, val_accuracy, val_accuracy_ema, dict["avg_loss"], self.lr))

        if val_accuracy > self.previous_best_acc:
            self.previous_best_acc = val_accuracy
            self.previous_best_epoch = epoch

        if epoch % self.snapshot_interval == 0:
            self.save_models(epoch)

    def calc_accuracy(self, mode="train", max_samples=None):
        self.set_mode("validation")

        if mode == "train":
            eval_data = iter(self.dataloader)
            num_imgs = len(self.dataset)
        elif mode == "validation":
            eval_data = iter(self.dataloader_val)
            num_imgs = len(self.dataset_val)

        batch_size = 200
        total_iters = num_imgs // batch_size
        if max_samples is not None:
            max_iter = max_samples // batch_size
        else:
            max_iter = None

        all_accuracies = []
        all_accuracies_ema = []

        for _iteration in range(total_iters):
            try:
                data = next(eval_data)
            except StopIteration:
                break
            if max_iter is not None and _iteration == max_iter:
                break

            image, question, question_len, answer = data['image'], data['question'], data['question_length'], data['answer']
            answer = answer.long()
            question = Variable(question)
            answer = Variable(answer)

            if self.cfg.CUDA:
                image = image.cuda()
                question = question.cuda()
                answer = answer.cuda().squeeze()

            with torch.no_grad():
                scores = self.model(image, question, question_len)
                scores_ema = self.model_ema(image, question, question_len)

            correct_ema = scores_ema.detach().argmax(1) == answer
            accuracy_ema = correct_ema.sum().cpu().numpy() / answer.shape[0]
            all_accuracies_ema.append(accuracy_ema)

            correct = scores.detach().argmax(1) == answer
            accuracy = correct.sum().cpu().numpy() / answer.shape[0]
            all_accuracies.append(accuracy)

        accuracy_ema = sum(all_accuracies_ema) / float(len(all_accuracies_ema))
        accuracy = sum(all_accuracies) / float(len(all_accuracies))

        return accuracy, accuracy_ema
