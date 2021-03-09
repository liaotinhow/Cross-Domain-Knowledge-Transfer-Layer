import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import sys
import ipdb
sys.path.extend(['../'])
from feeders import tools
import random

citi_ntu_map = np.array([5,9,21,2,13,17,1,6,10,7,11,8,12,14,18,15,19,16,20])

class Feeder(Dataset):
    def __init__(self, data_path, label_path, ntu_action, target_action, p_label_path=None,ntu_data_path=None, ntu_label_path=None,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True):
        """
        
        :param data_path: 
        :param label_path: 
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move: 
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.p_label_path = p_label_path
        self.ntu_data_path = ntu_data_path
        self.ntu_label_path = ntu_label_path
        self.ntu_action = ntu_action
        self.target_action = target_action
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            try:
                with open(self.label_path, 'rb') as f:
                    self.sample_name, self.label = pickle.load(f, encoding='latin1')
                    self.person_label = np.zeros(len(self.sample_name))
                    for i in range(len(self.sample_name)):
                        self.person_label[i] = int(self.sample_name[i][self.sample_name[i].find('P')+1: self.sample_name[i].find('P')+4])-1
            except:
                with open(self.label_path, 'rb') as f:
                    self.sample_name, self.label = pickle.load(f)
                if self.p_label_path is not None:   
                    self.person_label = np.load(self.p_label_path)
        
        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)


        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        try:
            with open(self.ntu_label_path) as f:
                self.ntu_sample_name, self.ntu_label = pickle.load(f)
        except:
            # for pickle file from python2
            try:
                with open(self.ntu_label_path, 'rb') as f:
                    self.ntu_sample_name, self.ntu_label = pickle.load(f, encoding='latin1')
                    self.ntu_person_label = np.zeros(len(self.ntu_sample_name))
                    for i in range(len(self.ntu_sample_name)):
                        self.ntu_person_label[i] = int(self.ntu_sample_name[i][self.ntu_sample_name[i].find('P')+1: self.ntu_sample_name[i].find('P')+4])-1
            except:
                with open(self.ntu_label_path, 'rb') as f:
                    self.ntu_sample_name, self.ntu_label = pickle.load(f)
                if self.p_label_path is not None:   
                    self.ntu_person_label = np.load(self.p_label_path)
        if self.use_mmap:
            self.ntu_data = np.load(self.ntu_data_path, mmap_mode='r')
        else:
            self.ntu_data = np.load(self.ntu_data_path)
        target_action=np.array(self.target_action)
        ntu_action=np.array(self.ntu_action)
        #citi
        #t_action=np.array([1,4,6,9,10,11,12,13,14])
        #n_action=np.array([7,8,5,13,14,27,0,22,9])
        #pku
        #t_action = np.array([3,8,10,2,9,22,41,33,34,6,30,51,40,28,48,37,49,36,35,5,13,19,29,15,17,20,23,46,25,39,4,31,1,50,32,7,11,44,43,42,45,26,18,27,21,24,16,12,14])
        #n_action = np.array([3,1,2,4,5,6,7,8,9,10,11,12,13,20,14,15,18,19,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,37,38,40,43,44,45,46,47,50,51,52,53,54,55,56,58])-1
        #ucla
        #t_action=np.array([1,2,3,4,5,6,8,9,11])
        #n_action =np.array([5,5,4,58,7,8,13,20,6])
        #msrd
        #_action = 
        #msr3d
        #t_action=np.array([1,2,11,5,6,10,14,20])
        #n_action =np.array([22,22,22,49,6,9,23,5])
        #t_action=np.array([1])
        #n_action =np.array([22])
        p_idx=[]
        n_idx=[]
        #ipdb.set_trace()
        for i in range(len(self.label)):
            if self.label[i] in target_action:
                p_idx.append(i)
        for i in range(len(self.ntu_label)):
            if self.ntu_label[i] in ntu_action:
                n_idx.append(i)
        p_idx = np.array(p_idx)
        n_idx = np.array(n_idx)
        self.data=self.data[p_idx]
        self.label=np.array(self.label)[p_idx].tolist()
        self.ntu_data=self.ntu_data[n_idx]
        self.ntu_label=np.array(self.ntu_label)[n_idx].tolist()
        for i in range(len(self.label)):
            self.label[i] = ntu_action[np.argwhere(target_action==self.label[i])[0][0]]
        num_of_source = self.ntu_data.shape[0]
        num_of_target = self.data.shape[0]
        choice_from_source = np.random.choice(num_of_source, num_of_target, replace=False)
        self.ntu_data = self.ntu_data[choice_from_source]
        self.ntu_label = np.array(self.ntu_label)[choice_from_source].tolist()
        # source domain label: 0 target:1
        #self.domain_label = np.zeros(2*num_of_target)
        #self.domain_label[num_of_target:] = 1
        # self.data = np.vstack((self.data, self.ntu_data))
        # self.label = self.label+self.ntu_label

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        #p_label = self.person_label[index]
        data_numpy = np.array(data_numpy)
        ntu_data_numpy = self.ntu_data[index]
        ntu_label = self.ntu_label[index]
        # choose one same action data
        #same_label_idx = np.argwhere(self.label==label)[0]
        #chosen_idx = np.random.choice(same_label_idx)
        #c_data = np.array(self.data[chosen_idx])
        #c_label = self.label[chosen_idx]
        #c_p_label = self.person_label[chosen_idx]
        
        # h_label = self.h_label[index]
        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)
        
        return data_numpy, label, index, ntu_data_numpy, ntu_label #, p_label, c_data, c_label, c_p_label
    
    #def few_shot_data(self):
        # data = 
        #return data

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        #ipdb.set_trace()
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
        
    def top_h(self, score):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -1:] for i, l in enumerate(self.h_label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def test(data_path, label_path, vid=None, graph=None, is_3d=False):
    '''
    vis the samples using matplotlib
    :param data_path: 
    :param label_path: 
    :param vid: the id of sample
    :param graph: 
    :param is_3d: when vis NTU, set it True
    :return: 
    '''
    import matplotlib.pyplot as plt
    loader = torch.utils.data.DataLoader(
        dataset=Feeder(data_path, label_path),
        batch_size=64,
        shuffle=False,
        num_workers=2)

    if vid is not None:
        sample_name = loader.dataset.sample_name
        sample_id = [name.split('.')[0] for name in sample_name]
        index = sample_id.index(vid)
        data, label, index = loader.dataset[index]
        data = data.reshape((1,) + data.shape)

        # for batch_idx, (data, label) in enumerate(loader):
        N, C, T, V, M = data.shape

        plt.ion()
        fig = plt.figure()
        if is_3d:
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        if graph is None:
            p_type = ['b.', 'g.', 'r.', 'c.', 'm.', 'y.', 'k.', 'k.', 'k.', 'k.']
            pose = [
                ax.plot(np.zeros(V), np.zeros(V), p_type[m])[0] for m in range(M)
            ]
            ax.axis([-1, 1, -1, 1])
            for t in range(T):
                for m in range(M):
                    pose[m].set_xdata(data[0, 0, t, :, m])
                    pose[m].set_ydata(data[0, 1, t, :, m])
                fig.canvas.draw()
                plt.pause(0.001)
        else:
            p_type = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'k-', 'k-', 'k-']
            import sys
            from os import path
            sys.path.append(
                path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
            G = import_class(graph)()
            edge = G.inward
            pose = []
            for m in range(M):
                a = []
                for i in range(len(edge)):
                    if is_3d:
                        a.append(ax.plot(np.zeros(3), np.zeros(3), p_type[m])[0])
                    else:
                        a.append(ax.plot(np.zeros(2), np.zeros(2), p_type[m])[0])
                pose.append(a)
            ax.axis([-1, 1, -1, 1])
            if is_3d:
                ax.set_zlim3d(-1, 1)
            for t in range(T):
                for m in range(M):
                    for i, (v1, v2) in enumerate(edge):
                        x1 = data[0, :2, t, v1, m]
                        x2 = data[0, :2, t, v2, m]
                        if (x1.sum() != 0 and x2.sum() != 0) or v1 == 1 or v2 == 1:
                            pose[m][i].set_xdata(data[0, 0, t, [v1, v2], m])
                            pose[m][i].set_ydata(data[0, 1, t, [v1, v2], m])
                            if is_3d:
                                pose[m][i].set_3d_properties(data[0, 2, t, [v1, v2], m])
                fig.canvas.draw()
                # plt.savefig('/home/lshi/Desktop/skeleton_sequence/' + str(t) + '.jpg')
                plt.pause(0.01)


if __name__ == '__main__':
    import os

    os.environ['DISPLAY'] = 'localhost:10.0'
    data_path = "../data/ntu/xview/val_data_joint.npy"
    label_path = "../data/ntu/xview/val_label.pkl"
    graph = 'graph.ntu_rgb_d.Graph'
    test(data_path, label_path, vid='S004C001P003R001A032', graph=graph, is_3d=True)
    # data_path = "../data/kinetics/val_data.npy"
    # label_path = "../data/kinetics/val_label.pkl"
    # graph = 'graph.Kinetics'
    # test(data_path, label_path, vid='UOD7oll3Kqo', graph=graph)
