import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import ipdb

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

class Adapt_layer(nn.Module):
    def __init__(self):
        super(Adapt_layer, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        #self.A = A
        #self.pre_A = self.graph.A
        #self.pre_A = nn.Parameter(torch.from_numpy(self.pre_A))
        #nn.init.constant_(self.pre_A, 1e-6)
        #self.decon_l = nn.ModuleList()
        #for i in range(3):
        self.decon_l.append(nn.ConvTranspose2d(20, 25, 1))

    def forward(self, x):
        N, C, V, T = x.size()
        # A = self.A.cuda(x.get_device())
        #PA = self.PA.cuda(x.get_device())
        x = self.decon_l[i](x.permute(0,3,1,2)).permute(0,2,3,1)
        return x




class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x

class adapt_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, p_A, coff_embedding=4, num_subset=3):
        super(adapt_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(p_A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(p_A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset
        #in_channels = A.shape[1]
        #inter_channels = p_A.shape[1]
        #out_channels = p_A.shape[1]
        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        self.conv_e = nn.ConvTranspose2d(A.shape[1],p_A.shape[1],1)
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))
            #self.conv_e.append(nn.Conv2d(A.shape[1], p_A.shape[1],1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A.cuda(x.get_device())
        A = A + self.PA
        y = None
        V_p = self.A.shape[1] 
        x = self.conv_e(x.permute(0,3,1,2).contiguous()).permute(0,2,3,1).contiguous()
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0,3,1,2).contiguous().view(N, V_p, self.inter_c*T)
            A2 = self.conv_b[i](x).view(N, self.inter_c*T, V_p)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V_p)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V_p))
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)

class Adapt_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, p_A, stride=1, residual=True):
        super(Adapt_unit, self).__init__()
        self.gcn1 = adapt_gcn(in_channels, out_channels, A, p_A)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) #+ self.residual(x)
        return self.relu(x)

class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A.cuda(x.get_device())
        A = A + self.PA
        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


class Model(nn.Module):
    def __init__(self, num_class=15, num_point=20, num_person=2, graph=None, graph_args=dict(), pre_graph=None,pre_graph_args=dict(), in_channels=3):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
        if pre_graph is None:
            a=0
        else:
            pre_Graph = import_class(pre_graph)
            self.pre_graph = pre_Graph(**pre_graph_args)
            p_A = self.pre_graph.A
        A = self.graph.A
        self.data_bn_ = nn.BatchNorm1d(num_person * in_channels * num_point)
        
        self.l1 = TCN_GCN_unit(3, 64, A, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A)
        self.l3 = TCN_GCN_unit(64, 64, A)
        #self.decon_l = nn.Conv2d(A.shape[1], 25, 1)
        # self.decon_fc = nn.Linear()
        
        self.l4 = TCN_GCN_unit(64, 64, A)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2)
        self.l6 = TCN_GCN_unit(128, 128, A)
        self.l7 = TCN_GCN_unit(128, 128, A)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2)
        self.l9 = TCN_GCN_unit(256, 256, A)
        self.l10 = TCN_GCN_unit(256, 256, A)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn_, 1)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn_(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l2(x)
        #x = self.ll3(x)
        #x = self.decon_l(x.permute(0,3,1,2).contiguous()).permute(0,2,3,1).contiguous()
        x = self.l3(x)
        '''x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)'''

        # N*M,C,T,V
        c_new = x.size(1)
        x1 = x.view(N, M, c_new, -1)
        x1 = x1.mean(3).mean(1)
        return x, x1 # self.fc(x)


class target_encoder(nn.Module):
    def __init__(self, num_class=None, num_point=None, num_person=2, graph=None, graph_args=dict(), pre_graph=None, pre_graph_args=dict(), in_channels=3):
        super(target_encoder, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
        
        pre_Graph = import_class(pre_graph)
        self.pre_graph = pre_Graph(**pre_graph_args)
        p_A = self.pre_graph.A
        A = self.graph.A
        #ipdb.set_trace()
        self.data_bn_ = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.l1 = TCN_GCN_unit(3, 64, A, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A)
        #self.decon_l = nn.ConvTranspose2d(A.shape[1], 25, 1)
        self.adapt_l = Adapt_unit(64,64,A,p_A)
        self.l3 = TCN_GCN_unit(64, 64, A)
   
        self.l4 = TCN_GCN_unit(64, 64, p_A)
        self.l5 = TCN_GCN_unit(64, 128, p_A, stride=2)
        self.l6 = TCN_GCN_unit(128, 128, p_A)
        self.l7 = TCN_GCN_unit(128, 128, p_A)
        self.l8 = TCN_GCN_unit(128, 256, p_A, stride=2)
        self.l9 = TCN_GCN_unit(256, 256, p_A)
        self.l10 = TCN_GCN_unit(256, 256, p_A)
        self.adp_l = nn.Linear(300*num_point, 300*25)  
        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn_, 1)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn_(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
         
        #x = self.decon_l(x.permute(0,3,1,2).contiguous()).permute(0,2,3,1).contiguous()
        x = self.l1(x)
        x = self.l2(x)
        #x = self.ll3(x)
        #x = self.decon_l(x.permute(0,3,1,2).contiguous()).permute(0,2,3,1).contiguous()
        x = self.l3(x)
        b,c,f,n = x.shape
        #x = self.adp_l(x.reshape(b,c,-1))
        #x = x.reshape(b,c,f,25)
        x = self.adapt_l(x)
        '''x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
        '''
        # N*M,C,T,V
        c_new = x.size(1)
        x1 = x.view(N, M, c_new, -1)
        x1 = x1.mean(3).mean(1)
        return x,x1 # self.fc(x)


class feature_discriminator(nn.Module):
    def __init__(self):
        super(feature_discriminator, self).__init__()
        self.fc1 = nn.Linear(64,2)
        self.sf = nn.Softmax(dim=1)
    def forward(self,x):
        return self.sf(self.fc1(x))
        
class Model_back(nn.Module):
    def __init__(self, num_class=15, num_point=20, num_person=2, graph=None, graph_args=dict(), pre_graph=None,pre_graph_args=dict(), in_channels=3):
        super(Model_back, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
        if pre_graph is None:
            a=0
        else:
            pre_Graph = import_class(pre_graph)
            self.pre_graph = pre_Graph(**pre_graph_args)
            p_A = self.pre_graph.A
        A = self.graph.A
        self.data_bn_ = nn.BatchNorm1d(num_person * in_channels * num_point)
        
        self.l1 = TCN_GCN_unit(3, 64, A, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A)
        self.l3 = TCN_GCN_unit(64, 64, A)
        self.l4 = TCN_GCN_unit(64, 64, A)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2)
        self.l6 = TCN_GCN_unit(128, 128, A)
        self.l7 = TCN_GCN_unit(128, 128, A)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2)
        self.l9 = TCN_GCN_unit(256, 256, A)
        self.l10 = TCN_GCN_unit(256, 256, A)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn_, 1)

    def forward(self, x):
        #N, C, T, V, M = x.size()
        #x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        #x = self.data_bn_(x)
        #x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        '''x = self.l1(x)
        x = self.l2(x)
        x = self.ll3(x)
        x = self.l3(x)
        '''
        M=int(2)
        N=int(x.size(0)/M)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
        # N*M,C,T,V
        c_new = x.size(1)
        x1 = x.view(N, M, c_new, -1)
        x1 = x1.mean(3).mean(1)
        return self.fc(x1)

class Target_back(nn.Module):
    def __init__(self, num_class=15, num_point=20, num_person=2, graph=None, graph_args=dict(), pre_graph=None,pre_graph_args=dict(), in_channels=3):
        super(Target_back, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
        if pre_graph is None:
            a=0
        else:
            pre_Graph = import_class(pre_graph)
            self.pre_graph = pre_Graph(**pre_graph_args)
            p_A = self.pre_graph.A
        A = self.graph.A
        self.data_bn_ = nn.BatchNorm1d(num_person * in_channels * num_point)
        
        #self.l1 = TCN_GCN_unit(3, 64, A, residual=False)
        #self.l2 = TCN_GCN_unit(64, 64, A)
        #self.l3 = TCN_GCN_unit(64, 64, A)
        self.l4 = TCN_GCN_unit(64, 64, p_A)
        self.l5 = TCN_GCN_unit(64, 128, p_A, stride=2)
        self.l6 = TCN_GCN_unit(128, 128, p_A)
        self.l7 = TCN_GCN_unit(128, 128, p_A)
        self.l8 = TCN_GCN_unit(128, 256, p_A, stride=2)
        self.l9 = TCN_GCN_unit(256, 256, p_A)
        self.l10 = TCN_GCN_unit(256, 256, p_A)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn_, 1)

    def forward(self, x):
        #N, C, T, V, M = x.size()
        #x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        #x = self.data_bn_(x)
        #x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        '''x = self.l1(x)
        x = self.l2(x)
        x = self.ll3(x)
        x = self.l3(x)
        '''
        M = int(2)
        N = int(x.size(0)/M)

        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x1 = x.view(N, M, c_new, -1)
        x1 = x1.mean(3).mean(1)
        return self.fc(x1)
