#!/usr/bin/env python
from __future__ import print_function

import argparse
import inspect
import os
import pickle
import random
import shutil
import time
from collections import OrderedDict
import numpy as np
# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
from model.agcn import feature_discriminator, Model_back, Target_back
import ipdb
class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epoch, after_scheduler=None):
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = -1
        super().__init__(optimizer)

    def get_lr(self):
        return [base_lr * (self.last_epoch + 1) / self.total_epoch for base_lr in self.base_lrs]

    def step(self, epoch=None, metric=None):
        if self.last_epoch >= self.total_epoch - 1:
            if metric is None:
                return self.after_scheduler.step(epoch)
            else:
                return self.after_scheduler.step(metric, epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


def init_seed(_):
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')
    
    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=2,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=32,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument('--target_encoder', default=None)
    parser.add_argument(
        '--target-encoder-args',
        type=dict,
        default=dict())
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument('--only_train_part', default=False)
    parser.add_argument('--only_train_epoch', default=0)
    parser.add_argument('--warm_up_epoch', default=0)
    parser.add_argument('--intersection_action', default=[0])
    return parser


class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0
        self.load_model()
        self.load_optimizer()
        self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model)
        self.model = Model(**self.arg.model_args).cuda(output_device)
        print(self.model)
        self.model_back = Model_back(**self.arg.model_args).cuda(output_device)
        self.klloss = nn.KLDivLoss(reduction='batchmean').cuda(output_device)
        self.loss = nn.CrossEntropyLoss().cuda(output_device)
        target_Encoder = import_class(self.arg.target_encoder)
        print(target_Encoder)
        self.target_encoder = target_Encoder(**self.arg.target_encoder_args).cuda(output_device)
        self.target_back = Target_back(**self.arg.target_encoder_args).cuda(output_device)
        self.feature_discriminator = feature_discriminator().cuda(output_device)
        
        dir_name = self.arg.work_dir.split('/')[-1] 
        if self.arg.weights:
            self.global_step = int(arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))
            
            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                pretrain_dict = { k: v for k, v in weights.items() if k in state }

                state.update(pretrain_dict)
                self.model.load_state_dict(state)
                '''freeze = False
                for child in self.model.children():
                    if child == self.model.l1:
                        freeze = True
                    if freeze:    
                        for para in child.parameters():
                            para.requires_grad = False'''
                        
            weights = torch.load(self.arg.weights)
            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])
            try:
                self.model_back.load_state_dict(weights)
            except:
                state = self.model_back.state_dict()
                pretrain_dict = { k: v for k, v in weights.items() if k in state }
                state.update(pretrain_dict)
                self.model_back.load_state_dict(state)

            #weights = torch.load('weights/tb_epoch_191.pt')
            #weights = OrderedDict(
            #    [[k.split('module.')[-1],
            #      v.cuda(output_device)] for k, v in weights.items()])
            #weights = torch.load('use/{}_tb_na.pt'.format(dir_name))

            #weights = OrderedDict(
            #    [[k.split('module.')[-1],
            #      v.cuda(output_device)] for k, v in weights.items()])
            '''try:
                self.target_back.load_state_dict(weights)
            except:
                state = self.target_back.state_dict()
                pretrain_dict = { k: v for k, v in weights.items() if k in state }
                state.update(pretrain_dict)
                self.target_back.load_state_dict(state)'''
            
            #weights = torch.load('runs/msr3d-191-4416.pt')

            #weights = OrderedDict(
            #    [[k.split('module.')[-1],
            #      v.cuda(output_device)] for k, v in weights.items()])

            #weights = torch.load('use/{}_te.pt'.format(dir_name))
            
            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])
            try:
                self.target_encoder.load_state_dict(weights)
            except:
                state = self.target_back.state_dict()
                pretrain_dict = { k: v for k, v in weights.items() if k in state }
                state.update(pretrain_dict)

            weights = torch.load('use/{}_dis.pt'.format(dir_name))
            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])
            try:
                self.feature_discriminator.load_state_dict(weights)
            except:
                state = self.feature_discriminator.state_dict()
                pretrain_dict = { k: v for k, v in weights.items() if k in state }
                state.update(pretrain_dict)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=output_device)
            if len(self.arg.device) > 1:
                self.model_back = nn.DataParallel(
                    self.model_back,
                    device_ids=self.arg.device,
                    output_device=output_device)
            if len(self.arg.device) > 1:
                self.target_encoder = nn.DataParallel(
                    self.target_encoder,
                    device_ids=self.arg.device,
                    output_device=output_device)
            if len(self.arg.device) > 1:
                self.target_back = nn.DataParallel(
                    self.target_back,
                    device_ids=self.arg.device,
                    output_device=output_device)
            if len(self.arg.device) > 1:
                self.feature_discriminator = nn.DataParallel(
                    self.feature_discriminator,
                    device_ids=self.arg.device,
                    output_device=output_device)
    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                list(self.model.parameters()),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()
        
        self.te_opt = optim.SGD(
            list(self.target_encoder.parameters())+list(self.target_back.parameters()),
            lr=self.arg.base_lr,
            momentum=0.9,
            weight_decay=self.arg.weight_decay)
        
        self.te_dis_opt = optim.SGD(
            list(self.target_encoder.parameters()),
            lr=self.arg.base_lr,
            momentum=0.9,
            weight_decay=self.arg.weight_decay)
        self.ntu_dis_opt = optim.SGD(
            list(self.model.parameters()),
            lr=self.arg.base_lr,
            momentum=0.9,
            weight_decay=self.arg.weight_decay)
        self.dis_opt = optim.SGD(
            list(self.feature_discriminator.parameters()),
            lr=self.arg.base_lr,
            momentum=0.9,
            weight_decay=self.arg.weight_decay)
        lr_scheduler_pre = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.arg.step, gamma=0.1)

        self.lr_scheduler = GradualWarmupScheduler(self.optimizer, total_epoch=self.arg.warm_up_epoch,
                                                   after_scheduler=lr_scheduler_pre)
        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        0.1 ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        self.model.train()
        self.target_encoder.train()
        self.model_back.train()
        self.target_back.train()
        self.feature_discriminator.train()
        if epoch >= 0:
            for child in self.model.children():
                for p in child.parameters():
                    p.requires_grad = True
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)
        # for name, param in self.model.named_parameters():
        #     self.train_writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
        loss_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader)
        if self.arg.only_train_part:
            if epoch > self.arg.only_train_epoch:
                print('only train part, require grad')
                for key, value in self.model.named_parameters():
                    if 'PA' in key:
                        value.requires_grad = True
                        # print(key + '-require grad')
            else:
                print('only train part, do not require grad')
                for key, value in self.model.named_parameters():
                    if 'PA' in key:
                        value.requires_grad = False
                        # print(key + '-not require grad')
        #citi
        inter_act = np.array(self.arg.intersection_action)
        #sa = np.array([1,4,6,9,10,11,12,13,14])
        #pku
        #sa = np.array([3,8,10,2,9,22,41,33,34,6,30,51,40,28,48,37,49,36,35,5,13,19,29,15,17,20,23,46,25,39,4,31,1,50,32,7,11,44,43,42,45,26,18,27,21,24,16,12,14])
        #ucla
        #sa = np.array([1,2,3,4,5,6,8,9,11])
        #msr3d
        d_hit = 0 
        d_all = 0
        kloss=[]
        for batch_idx, (data, label, index, ntu_data, ntu_label) in enumerate(process):
            self.global_step += 1
            # get data
            sa_idx=[]
            nn_idx=[]
            for idx,ll in enumerate(label):
                if ll.numpy() in inter_act:
                    sa_idx.append(idx)
                else:
                    nn_idx.append(idx)
            sa_idx = np.array(sa_idx)
            #s_data = Variable(data[sa_idx].float().cuda(self.output_device), requires_grad=False)
            data = Variable(data.float().cuda(self.output_device), requires_grad=False)
            label = Variable(label.long().cuda(self.output_device), requires_grad=False)
            #s_label = Variable(label[sa_idx].long().cuda(self.output_device), requires_grad=False)
            ntu_data = Variable(ntu_data.float().cuda(self.output_device), requires_grad=False)
            ntu_label = Variable(ntu_label.long().cuda(self.output_device), requires_grad=False)
            timer['dataloader'] += self.split_time()
            #ipdb.set_trace()
            # forward
            t_output, t_f = self.target_encoder(data)
            s_output, s_f = self.model(ntu_data)
            ipdb.set_trace()
            try:#if data.shape[0]%2 != 0:
                output1 = self.model_back(s_output[:-1])
                output2 = self.model_back(s_output[-4:])
                s_out = torch.cat((output1, output2[-1].view(1,-1)))
            except:
                s_out = self.model_back(s_output)
            d_label = torch.ones(ntu_data.shape[0]+len(sa_idx))
            #d_label[:data.shape[0]] = 0
            d_label[:len(sa_idx)] = 0
            d_label = Variable(d_label.long().cuda(self.output_device),requires_grad=False)
            predict_d = self.feature_discriminator(torch.cat((s_f[sa_idx], t_f), 0))
            dis_loss = self.loss(predict_d, d_label)
            try: #data.shape[0]%2 !=0: 
                output1 = self.target_back(t_output[:-1])
                output2 = self.target_back(t_output[-4:])
                t_out = torch.cat((output1, output2[-1].view(1,-1)))
            except:    
                t_out = self.target_back(t_output)

            '''if data.shape[0]%2 !=0: 
                output1 = self.model_back(t_output[:-1])
                output2 = self.model_back(t_output[-4:])
                t_out = torch.cat((output1, output2[-1].view(1,-1)))
            else:    '''
            #t_out = self.model_back(t_output)
            #t_out = self.target_back(t_output)
            #s_out = self.model_back(s_output)
            #if batch_idx == 0 and epoch == 0:
            #    self.train_writer.add_graph(self.model, output)
            
            ntu_loss = self.loss(s_out, ntu_label) - 1*dis_loss 
            target_loss = self.loss(t_out,label)# - 0.1*dis_loss
            kl = self.klloss(s_out,t_out)
            kloss.append(kl.detach().cpu().numpy())
            # backward
            self.optimizer.zero_grad()
            self.te_opt.zero_grad()
            self.te_dis_opt.zero_grad()
            self.ntu_dis_opt.zero_grad()
            self.dis_opt.zero_grad()

            #ntu_loss.backward(retain_graph=True)
            #self.optimizer.step()
            target_loss.backward(retain_graph=True)
            self.te_opt.step()
            dis_loss.backward()
            self.dis_opt.step()
            d_hit += torch.sum((torch.max(predict_d.data,1)[1]==d_label.data).float())
            d_all += data.shape[0]+len(sa_idx)
            d_acc = d_hit/d_all

            loss_value.append(target_loss.data.item())
            timer['model'] += self.split_time()
            process.set_description(f't_loss:{target_loss.item():.4f} ntu_loss:{ntu_loss.item():.4f} dis_loss:{dis_loss.item():.4f} d_acc:{d_acc:.4f}' )
            
            value, predict_label = torch.max(t_out.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            self.train_writer.add_scalar('acc', acc, self.global_step)
            self.train_writer.add_scalar('loss', target_loss.data.item(), self.global_step)
            self.train_writer.add_scalar('dis_loss', dis_loss.data.item(), self.global_step)
            self.train_writer.add_scalar('klloss', kl.data.item(),self.global_step)
            #self.train_writer.add_scalar('loss_l1', l1, self.global_step)
            # self.train_writer.add_scalar('batch_time', process.iterable.last_duration, self.global_step)

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            # if self.global_step % self.arg.log_interval == 0:
            #     self.print_log(
            #         '\tBatch({}/{}) done. Loss: {:.4f}  lr:{:.6f}'.format(
            #             batch_idx, len(loader), loss.data[0], lr))
            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.'.format(np.mean(loss_value)))
        self.print_log(
            '\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(
                **proportion))

        if save_model:
            state_dict = self.target_encoder.state_dict()
            weights = OrderedDict([[k.split('module.')[-1],
                                    v.cpu()] for k, v in state_dict.items()])

            dir_name = self.arg.work_dir.split('/')[-1] 
            torch.save(weights, 'weights/{}_te_'.format(dir_name) + str(epoch) + '.pt')# + '-' + str(int(self.global_step)) + '.pt')
            s_d = self.feature_discriminator.state_dict()
            w = OrderedDict([[k.split('module.')[-1],v.cpu()] for k,v in s_d.items()])
            torch.save(w , 'weights/{}_dis_{}.pt'.format(dir_name,epoch))
            t_b = self.target_back.state_dict()
            w = OrderedDict([[k.split('module.')[-1],v.cpu()] for k,v in t_b.items()])
            torch.save(w , 'weights/{}_tb_{}.pt'.format(dir_name,epoch))
                    
    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        self.target_encoder.eval()
        self.target_back.eval()
        self.model_back.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        kldisloss=[]
        for ln in loader_name:
            loss_value = []
            score_frag = []
            right_num_total = 0
            total_num = 0
            loss_total = 0
            step = 0
            process = tqdm(self.data_loader[ln])
            feature_l = []
            labels=[]
            n_labels=[]
            for batch_idx, (data, label, index, ntu_data, ntu_label) in enumerate(process):
                with torch.no_grad():
                    data = Variable(
                        data.float().cuda(self.output_device),
                        requires_grad=False)
                    label = Variable(
                        label.long().cuda(self.output_device),
                        requires_grad=False)
                    ntu_data = Variable(
                        ntu_data.float().cuda(self.output_device),
                        requires_grad=False)
                    ntu_label = Variable(
                        ntu_label.long().cuda(self.output_device),
                        requires_grad=False)
                    if data.shape[0]/2%2 != 0:
                        #ipdb.set_trace()
                        output1 = self.model_back(self.model(ntu_data[:-2])[0])
                        output2 = self.model_back(self.model(ntu_data[-4:])[0])
                        output = torch.cat((output1, output2[-2:].view(2,-1)))
                    else:
                        output = self.model_back(self.model(ntu_data)[0])
                    feature_l.append(output.data.cpu().numpy())
                    if isinstance(output, tuple):
                        output, l1 = output
                        l1 = l1.mean()
                    else:
                        l1 = 0
                    '''if data.shape[0]/2%2 !=0:
                        #ipdb.set_trace() 
                        output1 = self.target_back(self.target_encoder(data[:-1])[0])
                        output2 = self.target_back(self.target_encoder(data[-4:])[0])
                        t_out = torch.cat((output1, output2[-1:].view(1,-1)))
                    else:'''
                    #t_out = self.target_back(self.target_encoder(data)[0])
                    if data.shape[0]/2%2 !=0: 
                        output1 = self.model_back(self.target_encoder(data[:-2])[0])
                        output2 = self.model_back(self.target_encoder(data[-4:])[0])
                        t_out = torch.cat((output1, output2[-2:].view(2,-1)))
                    else:
                        t_out = self.model_back(self.target_encoder(data)[0])
                    target_loss = self.loss(t_out, label)
                    score_frag.append(t_out.data.cpu().numpy())
                    loss_value.append(target_loss.data.item())
                    #ipdb.set_trace()
                    _, predict_label = torch.max(t_out.data, 1)
                    step += 1
                    labels.append(label.data.cpu().numpy())
                    n_labels.append(ntu_label.data.cpu().numpy())
                    #ipdb.set_trace()
                    kloss = self.klloss(t_out, output)
                    kldisloss.append(kloss.cpu().numpy())
                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')
            labels = np.concatenate(labels)
            n_labels=np.concatenate(n_labels)
            f = np.concatenate(feature_l)
            score = np.concatenate(score_frag)
            #print(np.argmax(score,1))
            from sklearn.manifold import TSNE
            fff = np.vstack((f,score))
            f_em = TSNE(n_components=2).fit_transform(fff)
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches 
            cmap = plt.get_cmap('gnuplot')
            n_act=np.array([7,8,5,13,14,27,0,22,9])#citi
            act_name=['sit','stand','pick','put on','take off','phone call','drink','wave','clap']
            '''n_act = np.array([5,5,4,58,7,8,13,20,6])#msr3d#np.array([3,1,2,4,5,6,7,8,9,10,11,12,13,20,14,15,18,19,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,37,38,40,43,44,45,46,47,50,51,52,53,54,55,56,58])#pku
            '''
            colors = [cmap(i) for i in np.linspace(0,1,len(n_act))]
            #ax = plt.subplots()
            '''plt.figure()
            patch=[]
            for iid,i in enumerate(colors):
                patch.append(mpatches.Patch(color=i,label=act_name[iid]))
            dir_name = self.arg.work_dir.split('/')[-1] 
            for i in range(len(n_labels)):
                #act_n = act_name[np.argwhere(n_act==n_labels[i])[0][0]]
                plt.scatter(f_em[i,0], f_em[i,1],c=colors[np.argwhere(n_act==n_labels[i])[0][0]])
            for i in range(len(labels)):
                plt.scatter(f_em[i+len(n_labels),0], f_em[i+len(n_labels),1],c=colors[np.argwhere(n_act==labels[i])[0][0]])
            plt.legend(handles=patch, loc='upper left')
            #plt.scatter(f_em[len(f):,0], f_em[len(f):,1],c='r')
            #plt.savefig("tsne_{}_act/feature_tsne_epoch_{}.png".format(dir_name,epoch))
            plt.close()
            plt.figure()
            plt.scatter(f_em[:len(f),0],f_em[:len(f),1],c='b')
            plt.scatter(f_em[len(f):,0],f_em[len(f):,1],c='r')
            #plt.savefig('tsne_{}/feature_{}.png'.format(dir_name, epoch))
            plt.close()'''
            #confusion_matrix 
            cm = np.zeros((len(labels),len(labels)))

            for i in range(len(labels)):
                cm[labels[i], np.argmax(score,1)[i]]+=1
            np.save('/scratch2/users/tom/cm/cm_{}.npy'.format(epoch),cm)
            loss = np.mean(loss_value)
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
            # self.lr_scheduler.step(loss)
            print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)
            #ipdb.set_trace()
            kl = sum(kldisloss)/len(kldisloss)
            if self.arg.phase == 'train':
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('loss_l1', l1, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)
                self.val_writer.add_scalar('klloss', kl, self.global_step)
            
            score_dict = dict(
                zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.top_k(score, k)))
            #param = self.target_encoder.state_dict()
            #for par in param.items():
            #    ipdb.set_trace()
            
            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)
            
    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                if self.lr < 1e-3:
                    break
                save_model = 10#((epoch + 1) % self.arg.save_interval == 0) or (
                        #epoch + 1 == self.arg.num_epoch)

                self.train(epoch, save_model=save_model)

                self.eval(
                    epoch,
                    save_score=self.arg.save_score,
                    loader_name=['test'])

            print('best accuracy: ', self.best_acc, ' model_name: ', self.arg.model_saved_name)

        elif self.arg.phase == 'test':
            if not self.arg.test_feeder_args['debug']:
                wf = self.arg.model_saved_name + '_wrong.txt'
                rf = self.arg.model_saved_name + '_right.txt'
            else:
                wf = rf = None
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(0)
    processor = Processor(arg)
    processor.start()
