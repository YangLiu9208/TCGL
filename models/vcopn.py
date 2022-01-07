"""VCOPN"""
import math
from collections import OrderedDict
import random
import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.utils import dropout_adj
import torch.nn.functional as F
import numpy as np

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x

class SE_Fusion(nn.Module):
    def __init__(self,channel_1=128,channel_2=256,channel_3=256,reduction=6):
        super(SE_Fusion, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_1 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3), (channel_1+channel_2+channel_3) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3) // reduction, channel_1, bias=True),
            nn.Sigmoid()
        )
        self.fc_2 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3), (channel_1+channel_2+channel_3) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3) //reduction, channel_2, bias=True),
            nn.Sigmoid()
        )
        self.fc_3 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3), (channel_1+channel_2+channel_3) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3) //reduction, channel_3, bias=True),
            nn.Sigmoid()
        )

    def forward(self, fea1, fea2, fea3):

        #squeeze
        c_1= fea1.size()
        y_1 = fea1

        c_2= fea2.size()
        y_2 = fea2

        c_3= fea3.size()
        y_3 = fea3

        z=torch.cat((y_1,y_2,y_3),0)

        y_1 =self.fc_1(z).view(c_1)
        y_2 = self.fc_2(z).view(c_2)  
        y_3 = self.fc_3(z).view(c_3) 
        
        return torch.mul((F.relu(y_1)).expand_as(fea1),fea1), torch.mul((F.relu(y_2)).expand_as(fea2),fea2), torch.mul((F.relu(y_3)).expand_as(fea3),fea3) 

class SE_Fusion_Four(nn.Module):
    def __init__(self,channel_1=128,channel_2=256,channel_3=256,channel_4=256,reduction=8):
        super(SE_Fusion_Four, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_1 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3+channel_4), (channel_1+channel_2+channel_3+channel_4) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3+channel_4) // reduction, channel_1, bias=True),
            nn.Sigmoid()
        )
        self.fc_2 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3+channel_4), (channel_1+channel_2+channel_3+channel_4) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3+channel_4) //reduction, channel_2, bias=True),
            nn.Sigmoid()
        )
        self.fc_3 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3+channel_4), (channel_1+channel_2+channel_3+channel_4) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3+channel_4) //reduction, channel_3, bias=True),
            nn.Sigmoid()
        )
        self.fc_4 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3+channel_4), (channel_1+channel_2+channel_3+channel_4) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3+channel_4) //reduction, channel_4, bias=True),
            nn.Sigmoid()
        )

    def forward(self, fea1, fea2, fea3, fea4):

        #squeeze
        c_1= fea1.size()
        y_1 = fea1

        c_2= fea2.size()
        y_2 = fea2

        c_3= fea3.size()
        y_3 = fea3

        c_4= fea4.size()
        y_4 = fea4

        z=torch.cat((y_1,y_2,y_3,y_4),0)

        y_1 =self.fc_1(z).view(c_1)
        y_2 = self.fc_2(z).view(c_2)  
        y_3 = self.fc_3(z).view(c_3) 
        y_4 = self.fc_4(z).view(c_4) 
        
        return torch.mul((F.relu(y_1)).expand_as(fea1),fea1), torch.mul((F.relu(y_2)).expand_as(fea2),fea2), torch.mul((F.relu(y_3)).expand_as(fea3),fea3), torch.mul((F.relu(y_4)).expand_as(fea4),fea4)


class SE_Fusion_batch(nn.Module):
    def __init__(self,channel_1=128,channel_2=256,channel_3=256,reduction=6):
        super(SE_Fusion_batch, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_1 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3), (channel_1+channel_2+channel_3) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3) // reduction, channel_1, bias=True),
            nn.Sigmoid()
        )
        self.fc_2 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3), (channel_1+channel_2+channel_3) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3) //reduction, channel_2, bias=True),
            nn.Sigmoid()
        )
        self.fc_3 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3), (channel_1+channel_2+channel_3) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3) //reduction, channel_3, bias=True),
            nn.Sigmoid()
        )

    def forward(self, fea1, fea2, fea3):

        #squeeze
        b_1,c_1= fea1.size()
        y_1 = fea1

        b_2,c_2= fea2.size()
        y_2 = fea2

        b_3,c_3= fea3.size()
        y_3 = fea3

        z=torch.cat((y_1,y_2,y_3),1)

        y_1 =self.fc_1(z).view(b_1,c_1)
        y_2 = self.fc_2(z).view(b_2,c_2)  
        y_3 = self.fc_3(z).view(b_3,c_3) 
        
        return torch.mul((F.relu(y_1)).expand_as(fea1),fea1), torch.mul((F.relu(y_2)).expand_as(fea2),fea2), torch.mul((F.relu(y_3)).expand_as(fea3),fea3) 

class SE_Fusion_Four_batch(nn.Module):
    def __init__(self,channel_1=128,channel_2=256,channel_3=256,channel_4=256,reduction=8):
        super(SE_Fusion_Four, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_1 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3+channel_4), (channel_1+channel_2+channel_3+channel_4) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3+channel_4) // reduction, channel_1, bias=True),
            nn.Sigmoid()
        )
        self.fc_2 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3+channel_4), (channel_1+channel_2+channel_3+channel_4) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3+channel_4) //reduction, channel_2, bias=True),
            nn.Sigmoid()
        )
        self.fc_3 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3+channel_4), (channel_1+channel_2+channel_3+channel_4) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3+channel_4) //reduction, channel_3, bias=True),
            nn.Sigmoid()
        )
        self.fc_4 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3+channel_4), (channel_1+channel_2+channel_3+channel_4) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3+channel_4) //reduction, channel_4, bias=True),
            nn.Sigmoid()
        )

    def forward(self, fea1, fea2, fea3, fea4):

        #squeeze
        b_1,c_1= fea1.size()
        y_1 = fea1

        b_2,c_2= fea2.size()
        y_2 = fea2

        b_3,c_3= fea3.size()
        y_3 = fea3

        b_4,c_4= fea4.size()
        y_4 = fea4

        z=torch.cat((y_1,y_2,y_3,y_4),0)

        y_1 =self.fc_1(z).view(b_1,c_1)
        y_2 = self.fc_2(z).view(b_2,c_2)  
        y_3 = self.fc_3(z).view(b_3,c_3) 
        y_4 = self.fc_4(z).view(b_4,c_4) 
        
        return torch.mul((F.relu(y_1)).expand_as(fea1),fea1), torch.mul((F.relu(y_2)).expand_as(fea2),fea2), torch.mul((F.relu(y_3)).expand_as(fea3),fea3), torch.mul((F.relu(y_4)).expand_as(fea4),fea4)


class VCOPN(nn.Module):
    """Video clip order prediction with PFE (Pairwire Feature Extraction), the same as OPN."""
    def __init__(self, base_network, feature_size, tuple_len):
        """
        Args:
            feature_size (int): 512
        """
        super(VCOPN, self).__init__()

        self.base_network = base_network
        self.feature_size = feature_size
        self.tuple_len = tuple_len
        self.class_num = math.factorial(tuple_len)

        self.fc7 = nn.Linear(self.feature_size*2, 512)
        pair_num = int(tuple_len*(tuple_len-1)/2)
        self.fc8 = nn.Linear(512*pair_num, self.class_num)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, tuple):
        f = []  # clip features
        for i in range(self.tuple_len):
            clip = tuple[:, i, :, :, :, :]
            f.append(self.base_network(clip))

        pf = []  # pairwise concat
        for i in range(self.tuple_len):
            for j in range(i+1, self.tuple_len):
                pf.append(torch.cat([f[i], f[j]], dim=1))

        pf = [self.fc7(i) for i in pf]
        pf = [self.relu(i) for i in pf]
        h = torch.cat(pf, dim=1)
        h = self.dropout(h)
        h = self.fc8(h)  # logits

        return f,h


class VCOPN_RNN(nn.Module):
    """Video clip order prediction with RNN."""
    def __init__(self, base_network, feature_size, tuple_len, hidden_size, rnn_type='LSTM'):
        """
        Args:
            feature_size (int): 1024
        """
        super(VCOPN_RNN, self).__init__()

        self.base_network = base_network
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.tuple_len = tuple_len
        self.class_num = math.factorial(tuple_len)
        self.rnn_type = rnn_type

        if self.rnn_type == 'LSTM':
            self.lstm = nn.LSTM(self.feature_size, self.hidden_size)
        elif self.rnn_type == 'GRU':
            self.gru = nn.GRU(self.feature_size, self.hidden_size)
        
        self.fc = nn.Linear(self.hidden_size, self.class_num)

    def forward(self, tuple):
        f = []  # clip features
        for i in range(self.tuple_len):
            clip = tuple[:, i, :, :, :, :]
            f.append(self.base_network(clip))

        inputs = torch.stack(f)
        if self.rnn_type == 'LSTM':
            outputs, (hn, cn) = self.lstm(inputs)
        elif self.rnn_type == 'GRU':
            outputs, hn = self.gru(inputs)

        h = self.fc(hn.squeeze(dim=0))  # logits

        return h

class VCOPN_GCN(nn.Module):
    """Video clip order prediction with PFE (Pairwire Feature Extraction), the same as OPN."""
    def __init__(self, base_network, feature_size, tuple_len):
        """
        Args:
            feature_size (int): 512
        """
        super(VCOPN_GCN, self).__init__()

        self.base_network = base_network
        self.feature_size = feature_size
        self.tuple_len = tuple_len
        self.class_num = math.factorial(tuple_len)

        self.gcn_f = GCNConv(self.feature_size, 512)
        #self.gcn_f2 = GCNConv(self.feature_size, 512)
        #self.gcn_f3 = GCNConv(self.feature_size, 512)
        self.gcn1 = GCNConv(self.feature_size, 512)
        #self.gcn2 = GCNConv(512*tuple_len, self.class_num)
        #pair_num = int(tuple_len*(tuple_len-1)/2)
        self.fc7 = nn.Linear(512*tuple_len, self.class_num)
        #self.fc8 = nn.Linear(512, self.class_num)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = torch.nn.Linear(512, 256)
        self.fc2 = torch.nn.Linear(256, 256)
    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)
        #return z

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / 0.5)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / 0.5)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret
    def repeating(self, x):
        _,c, t, h, w = x.shape
        x_new=[]
        for ind in range(t):
            one_frame = x[:,:,ind,:,:] # c, h, w
            one_frame = torch.unsqueeze(one_frame, 2)# -> c, 1, h, w
            x_new.append(one_frame.repeat(1,1,2,1,1))
        x_repeat=torch.cat([torch.cat([torch.cat([x_new[0],x_new[1]],2),x_new[2]],2),x_new[3]],2)
        return x_repeat

    def adjacent_shuffle(self, x):
        # (C X T x H x W)
        tmp = torch.chunk(x, 4, dim=1)
        order = [0,1,2,3]
        ind1 = random.randint(0,3)
        ind2 = (ind1 + random.randint(0,2) + 1) % 4
        order[ind1], order[ind2] = order[ind2], order[ind1]
        x_new = torch.cat((tmp[order[0]], tmp[order[1]], tmp[order[2]], tmp[order[3]]),1)
        return x_new

    def forward(self, tuple, tuple_orders):

        #edge_index_clip_frames=torch.tensor([[0,0,1,2,1,1,1,0,2,3,2,2,2,2,0,1,3,4,3,3,3,3,1,2,4,5,4,4,4,4,2,3,5,6,5,5,5,5,3,4,6,7,6,6,6,6,4,5,7,8,7,7,7,7,5,6,8,9,8,8,8,8,6,7,9,10,9,9,9,9,7,8,9,10,10,10,10,10,8,9,11,12,11,11,11,11,9,10,12,13,12,12,12,12,10,11,13,14,13,13,13,13,11,12,14,15,14,14,14,14,12,13,15,16,15,15,13,14],\
        #    [1,2,0,0,0,2,3,1,1,1,0,1,3,4,2,2,2,2,1,2,4,5,3,3,3,3,2,3,5,6,4,4,4,4,3,4,6,7,5,5,5,5,4,5,7,8,6,6,6,6,5,6,8,9,7,7,7,7,6,7,9,10,8,8,8,8,7,8,9,10,9,9,9,9,8,9,11,12,10,10,10,10,9,10,12,13,11,11,11,11,10,11,13,14,12,12,12,12,11,12,14,15,13,13,13,13,12,13,15,16,14,14,14,14,13,14,15,15]],dtype=torch.long).cuda()
        #edge_index_clip=torch.tensor([[0,1,1,2],[1,0,2,1]],dtype=torch.long).cuda()
        edge_index_clip=[]
        edge_index_clip_1=torch.tensor([[0,1,1,2],[1,0,2,1]],dtype=torch.long).cuda()
        edge_index_clip_2=torch.tensor([[0,2,1,2],[2,0,2,1]],dtype=torch.long).cuda()
        edge_index_clip_3=torch.tensor([[0,1,2,0],[1,0,0,2]],dtype=torch.long).cuda()
        edge_index_clip_4=torch.tensor([[0,1,2,0],[1,0,0,2]],dtype=torch.long).cuda()
        edge_index_clip_5=torch.tensor([[1,2,2,0],[2,1,0,2]],dtype=torch.long).cuda()
        edge_index_clip_6=torch.tensor([[0,1,1,2],[1,0,2,1]],dtype=torch.long).cuda()
        for i in range(tuple_orders.size(0)):
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([0,1,2])):
               edge_index_clip.append(edge_index_clip_1)
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([0,2,1])):
               edge_index_clip.append(edge_index_clip_2)
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([1,0,2])):
                edge_index_clip.append(edge_index_clip_3)
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([1,2,0])):
                edge_index_clip.append(edge_index_clip_4)
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([2,0,1])):
                edge_index_clip.append(edge_index_clip_5)
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([2,1,0])):
                edge_index_clip.append(edge_index_clip_6)
        #edge_index_clip=torch.tensor(np.kron(np.eye(2), edge_index_clip.numpy())).cuda()
        #edge_index_frames=torch.tensor([[0,0,0,1,2,3,1,1,1,0,2,3,2,2,2,0,1,3,3,3,3,0,1,2],\
        #    [1,2,3,0,0,0,0,2,3,1,1,1,0,1,3,2,2,2,0,1,2,3,3,3]],dtype=torch.long).cuda()
        edge_index_frames=torch.tensor([[0,1,1,2,2,3],\
            [1,0,2,1,3,2]],dtype=torch.long).cuda()
        sub_f = []  # clip sub-frames features
        cl= []  # clip features
        for i in range(self.tuple_len):
            cl.append(self.base_network(tuple[:, i, :, :, :, :]))
            for j in range(4):
                sub_f.append(self.base_network(self.repeating(tuple[:, i, :, 4*j:4*(j+1), :, :])))

        sub_f1=sub_f[0:4]
        sub_f2=sub_f[4:8]
        sub_f3=sub_f[8:12]
        sub_f1=torch.stack(sub_f1).permute([1,0,2])
        sub_f2=torch.stack(sub_f2).permute([1,0,2])
        sub_f3=torch.stack(sub_f3).permute([1,0,2])
        sub_f1_drop=[]
        sub_f2_drop=[]
        sub_f3_drop=[]
        edge_index_frames_drop1=dropout_adj(edge_index_frames, p=0.2)[0]
        edge_index_frames_drop2=dropout_adj(edge_index_frames, p=0.2)[0]
        edge_index_frames_drop3=dropout_adj(edge_index_frames, p=0.2)[0]
        sub_f1_gcn=[]
        sub_f2_gcn=[]
        sub_f3_gcn=[]
        sub_f1_gcn_drop=[]
        sub_f2_gcn_drop=[]
        sub_f3_gcn_drop=[]
        cl=torch.stack(cl).permute([1,0,2])
        gf1=[]
        gf1_relu=[]
        gf1_drop=[]
        gf2=[]
        h=[]
        contrast_loss_1=0.0
        contrast_loss_2=0.0
        contrast_loss_3=0.0
        for j in range(cl.size(0)):
            sub_f1_drop.append(drop_feature(sub_f1[j,:,:], 0.1))
            sub_f2_drop.append(drop_feature(sub_f2[j,:,:], 0.1))
            sub_f3_drop.append(drop_feature(sub_f3[j,:,:], 0.1))
            sub_f1_gcn.append(self.gcn_f(sub_f1[j], edge_index_frames)) 
            sub_f2_gcn.append(self.gcn_f(sub_f2[j], edge_index_frames)) 
            sub_f3_gcn.append(self.gcn_f(sub_f3[j], edge_index_frames)) 
            sub_f1_gcn_drop.append(self.gcn_f(sub_f1_drop[j], edge_index_frames_drop1)) 
            sub_f2_gcn_drop.append(self.gcn_f(sub_f2_drop[j], edge_index_frames_drop2)) 
            sub_f3_gcn_drop.append(self.gcn_f(sub_f3_drop[j], edge_index_frames_drop3)) 
            gf1.append(self.gcn1(cl[j,:,:], edge_index_clip[j])) 
            #temp=torch.stack(gf1_relu).squeeze(0)
            h.append(torch.cat((gf1[j][0],gf1[j][1],gf1[j][2]), dim=0))
            gf1_relu.append(self.relu(h[j])) 
            gf1_drop.append(self.dropout(gf1_relu[j]))
            gf2.append(self.fc7(gf1_drop[j]))
            contrast_loss_1 = contrast_loss_1+self.loss(sub_f1_gcn[j], sub_f1_gcn_drop[j], batch_size=cl.size(0))
            contrast_loss_2 = contrast_loss_2+self.loss(sub_f2_gcn[j], sub_f2_gcn_drop[j], batch_size=cl.size(0))
            contrast_loss_3 = contrast_loss_3+self.loss(sub_f3_gcn[j], sub_f3_gcn_drop[j], batch_size=cl.size(0))
            #gf2.append(self.gcn2(gf1_drop[j], edge_index_clip))
        #return f,gf2
        gf2=torch.stack(gf2)
        #sub_f1_gcn=torch.stack(sub_f1_gcn)
        #sub_f2_gcn=torch.stack(sub_f2_gcn)
        #sub_f3_gcn=torch.stack(sub_f3_gcn)
        #sub_f1_gcn_drop=torch.stack(sub_f1_gcn_drop)
        #sub_f2_gcn_drop=torch.stack(sub_f2_gcn_drop)
        #sub_f3_gcn_drop=torch.stack(sub_f3_gcn_drop)

        return contrast_loss_1, contrast_loss_2, contrast_loss_3, gf2

class VCOPN_GCN_R(nn.Module):
    """Video clip order prediction with PFE (Pairwire Feature Extraction), the same as OPN."""
    def __init__(self, base_network, feature_size, tuple_len):
        """
        Args:
            feature_size (int): 512
        """
        super(VCOPN_GCN_R, self).__init__()

        self.base_network = base_network
        self.feature_size = feature_size
        self.tuple_len = tuple_len
        self.class_num = math.factorial(tuple_len)
        self.gcn_f = GCNConv(self.feature_size, 512)
        self.gcn_f1 = GCNConv(self.feature_size, 512)
        self.gcn_f2 = GCNConv(self.feature_size, 512)
        self.gcn_f3 = GCNConv(self.feature_size, 512)
        self.gcn1 = GCNConv(self.feature_size, 512)
        self.gcn2 = GCNConv(512, 512)

        #self.hid = 256
        #self.in_head = 8
        #self.out_head = 1
        #self.gatn1 = GATConv(self.feature_size, 256, heads=self.in_head, dropout=0.6)
        #self.gatn2 = GATConv(self.hid*self.in_head, 256, heads=self.out_head, dropout=0.6)

        #self.fc7 = nn.Linear(512*tuple_len, 512)
        #self.fc8 = nn.Linear(512, self.class_num)
        self.fc7 = nn.Linear(self.feature_size*2, 512)
        pair_num = int(tuple_len*(tuple_len-1)/2)
        self.fc8 = nn.Linear(512*pair_num, self.class_num)
        #self.fc8 = nn.Linear(512, self.class_num)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = torch.nn.Linear(512, 256)
        self.fc2 = torch.nn.Linear(256, 256)

        self.fusion=SE_Fusion(512,512,512,6)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)
        #return z

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / 0.5)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / 0.5)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret
    def repeating(self, x):
        _,c, t, h, w = x.shape
        x_new=[]
        for ind in range(t):
            one_frame = x[:,:,ind,:,:] # c, h, w
            one_frame = torch.unsqueeze(one_frame, 2)# -> c, 1, h, w
            x_new.append(one_frame.repeat(1,1,2,1,1))
        x_repeat=torch.cat([x_new[0],x_new[1],x_new[2],x_new[3]],2)
        return x_repeat

    def repeating2(self, x):
        _,c, t, h, w = x.shape
        x_repeat=torch.cat([x,x],2)
        return x_repeat

    def adjacent_shuffle(self, x):
        # (C X T x H x W)
        tmp = torch.chunk(x, 4, dim=2)
        order = [0,1,2,3]
        ind1 = random.randint(0,3)
        ind2 = (ind1 + random.randint(0,2) + 1) % 4
        order[ind1], order[ind2] = order[ind2], order[ind1]
        x_new = torch.cat((tmp[order[0]], tmp[order[1]], tmp[order[2]], tmp[order[3]]),2)
        return x_new

    def adjacent_shuffle_clip(self, x):
        # (C X T x H x W)
        tmp = torch.chunk(x, 16, dim=2)
        ind = torch.randperm(16)
        x_new=[]
        for i in range(16):
            x_new.append(x[:,:,ind[i],:,:])
        x_new= torch.stack(x_new,2)
        return x_new

    def forward(self, tuple, tuple_orders):

        #edge_index_clip_frames=torch.tensor([[0,0,1,2,1,1,1,0,2,3,2,2,2,2,0,1,3,4,3,3,3,3,1,2,4,5,4,4,4,4,2,3,5,6,5,5,5,5,3,4,6,7,6,6,6,6,4,5,7,8,7,7,7,7,5,6,8,9,8,8,8,8,6,7,9,10,9,9,9,9,7,8,9,10,10,10,10,10,8,9,11,12,11,11,11,11,9,10,12,13,12,12,12,12,10,11,13,14,13,13,13,13,11,12,14,15,14,14,14,14,12,13,15,16,15,15,13,14],\
        #    [1,2,0,0,0,2,3,1,1,1,0,1,3,4,2,2,2,2,1,2,4,5,3,3,3,3,2,3,5,6,4,4,4,4,3,4,6,7,5,5,5,5,4,5,7,8,6,6,6,6,5,6,8,9,7,7,7,7,6,7,9,10,8,8,8,8,7,8,9,10,9,9,9,9,8,9,11,12,10,10,10,10,9,10,12,13,11,11,11,11,10,11,13,14,12,12,12,12,11,12,14,15,13,13,13,13,12,13,15,16,14,14,14,14,13,14,15,15]],dtype=torch.long).cuda()
        #edge_index_clip=torch.tensor([[0,1,1,2],[1,0,2,1]],dtype=torch.long).cuda()
        edge_index_clip=[]
        edge_index_clip_drop=[]
        edge_index_clip_1=torch.tensor([[0,1,1,2],[1,0,2,1]],dtype=torch.long).cuda()
        edge_index_clip_2=torch.tensor([[0,2,1,2],[2,0,2,1]],dtype=torch.long).cuda()
        edge_index_clip_3=torch.tensor([[0,1,2,0],[1,0,0,2]],dtype=torch.long).cuda()
        edge_index_clip_4=torch.tensor([[0,1,2,0],[1,0,0,2]],dtype=torch.long).cuda()
        edge_index_clip_5=torch.tensor([[1,2,2,0],[2,1,0,2]],dtype=torch.long).cuda()
        edge_index_clip_6=torch.tensor([[0,1,1,2],[1,0,2,1]],dtype=torch.long).cuda()
        for i in range(tuple_orders.size(0)):
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([0,1,2])):
                edge_index_clip.append(edge_index_clip_1)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_1, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([0,2,1])):
                edge_index_clip.append(edge_index_clip_2)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_2, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([1,0,2])):
                edge_index_clip.append(edge_index_clip_3)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_3, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([1,2,0])):
                edge_index_clip.append(edge_index_clip_4)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_4, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([2,0,1])):
                edge_index_clip.append(edge_index_clip_5)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_5, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([2,1,0])):
                edge_index_clip.append(edge_index_clip_6)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_6, p=0.2)[0])
        #edge_index_clip=torch.tensor(np.kron(np.eye(2), edge_index_clip.numpy())).cuda()
        #edge_index_frames=torch.tensor([[0,0,0,1,2,3,1,1,1,0,2,3,2,2,2,0,1,3,3,3,3,0,1,2],\
        #    [1,2,3,0,0,0,0,2,3,1,1,1,0,1,3,2,2,2,0,1,2,3,3,3]],dtype=torch.long).cuda()
        edge_index_frames=torch.tensor([[0,1,1,2,2,3],\
            [1,0,2,1,3,2]],dtype=torch.long).cuda()
        sub_f = []  # clip sub-frames features
        sub_shuffle_f = []
        cl= []  # clip features
        cl_shuffle= []  # clip features
        for i in range(self.tuple_len):
            cl.append(self.base_network(tuple[:, i, :, :, :, :]))
            #cl_shuffle.append(self.base_network(self.adjacent_shuffle_clip(tuple[:, i, :, :, :, :])))
            for j in range(4):
                sub_f.append(self.base_network(self.repeating(tuple[:, i, :, 4*j:4*(j+1), :, :])))
                #sub_f.append(self.base_network(tuple[:, i, :, 8*j:8*(j+1), :, :]))
                #sub_f.append(self.base_network(self.repeating(self.adjacent_shuffle(tuple[:, i, :, 4*j:4*(j+1), :, :]))))

        sub_f1=sub_f[0:4]
        sub_f2=sub_f[4:8]
        sub_f3=sub_f[8:12]
        #sub_shuffle_f1=sub_shuffle_f[0:4]
        #sub_shuffle_f2=sub_shuffle_f[4:8]
        #sub_shuffle_f3=sub_shuffle_f[8:12]
        sub_f1=torch.stack(sub_f1).permute([1,0,2])
        sub_f2=torch.stack(sub_f2).permute([1,0,2])
        sub_f3=torch.stack(sub_f3).permute([1,0,2])
        #sub_shuffle_f1=torch.stack(sub_shuffle_f1).permute([1,0,2])
        #sub_shuffle_f2=torch.stack(sub_shuffle_f2).permute([1,0,2])
        #sub_shuffle_f3=torch.stack(sub_shuffle_f3).permute([1,0,2])
        sub_f1_drop=[]
        sub_f2_drop=[]
        sub_f3_drop=[]
        sub_f1_shuffle_drop=[]
        sub_f2_shuffle_drop=[]
        sub_f3_shuffle_drop=[]
        edge_index_frames_drop1=dropout_adj(edge_index_frames, p=0.2)[0]
        edge_index_frames_drop2=dropout_adj(edge_index_frames, p=0.2)[0]
        edge_index_frames_drop3=dropout_adj(edge_index_frames, p=0.2)[0]
        edge_index_frames_shuffle_drop1=dropout_adj(edge_index_frames, p=0.2)[0]
        edge_index_frames_shuffle_drop2=dropout_adj(edge_index_frames, p=0.2)[0]
        edge_index_frames_shuffle_drop3=dropout_adj(edge_index_frames, p=0.2)[0]
        sub_f1_gcn=[]
        sub_f2_gcn=[]
        sub_f3_gcn=[]
        sub_f1_gcn_shuffle=[]
        sub_f2_gcn_shuffle=[]
        sub_f3_gcn_shuffle=[]
        cl=torch.stack(cl).permute([1,0,2])
        #cl_shuffle=torch.stack(cl_shuffle).permute([1,0,2])
        gf1=[]
        gf2_relu=[]
        gf2_drop=[]
        gf1_shuffle=[]
        gf2=[]
        gf2_shuffle=[]
        gf3=[]
        h=[]
        contrast_loss_1=0.0
        contrast_loss_2=0.0
        contrast_loss_3=0.0
        contrast_loss_4=0.0
        contrast_loss_5=0.0
        contrast_loss_6=0.0
        contrast_loss_clip=0.0
        for j in range(cl.size(0)):
            #sub_f1_drop.append(drop_feature(sub_f1[j,:,:], 0.2))
            #sub_f2_drop.append(drop_feature(sub_f2[j,:,:], 0.2))
            #sub_f3_drop.append(drop_feature(sub_f3[j,:,:], 0.2))
            sub_f1_shuffle_drop.append(drop_feature(sub_f1[j,:,:], 0.1))
            sub_f2_shuffle_drop.append(drop_feature(sub_f2[j,:,:], 0.1))
            sub_f3_shuffle_drop.append(drop_feature(sub_f3[j,:,:], 0.1))
            sub_f1_gcn.append(self.gcn_f1(sub_f1[j], edge_index_frames)) 
            sub_f2_gcn.append(self.gcn_f2(sub_f2[j], edge_index_frames)) 
            sub_f3_gcn.append(self.gcn_f3(sub_f3[j], edge_index_frames)) 
            sub_f1_gcn_shuffle.append(self.gcn_f1(sub_f1_shuffle_drop[j], edge_index_frames_shuffle_drop1)) 
            sub_f2_gcn_shuffle.append(self.gcn_f2(sub_f2_shuffle_drop[j], edge_index_frames_shuffle_drop2)) 
            sub_f3_gcn_shuffle.append(self.gcn_f3(sub_f3_shuffle_drop[j], edge_index_frames_shuffle_drop3)) 

            
            gf1.append(self.gcn1(cl[j,:,:], edge_index_clip[j]))
            #gf2.append(self.gcn2(gf1[j], edge_index_clip[j]))  
            gf1_shuffle.append(self.gcn1(drop_feature(cl[j,:,:], 0.1), edge_index_clip_drop[j]))
            #gf2_shuffle.append(self.gcn2(gf1_shuffle[j], edge_index_clip_drop[j]))  
 
            #gf1.append(self.gatn1(cl[j,:,:], edge_index_clip[j]))
            #gf1 = [F.elu(k) for k in gf1]
            #gf1 = [self.dropout(p) for p in gf1]
            #gf2.append(self.gatn2(gf1[j], edge_index_clip[j]))

            #pf = []  # pairwise concat
            #for m in range(self.tuple_len):
            #    for n in range(m+1, self.tuple_len):
            #        pf.append(torch.cat([gf1[j][m], gf1[j][n]], dim=0))
            #pf = [self.fc7(k) for k in pf]
            #pf = [self.relu(p) for p in pf]
            #h = torch.cat(pf, dim=0)
            #h = self.dropout(h)
            #gf2.append(self.fc8(h))#

            fea1,fea2,fea3=self.fusion(gf1[j][0],gf1[j][1],gf1[j][2])
            pf=[]
            pf.append(torch.cat([fea1, fea2], dim=0))
            pf.append(torch.cat([fea1, fea3], dim=0))
            pf.append(torch.cat([fea2, fea3], dim=0))
            pf = [self.fc7(k) for k in pf]
            pf = [self.relu(p) for p in pf]
            h = torch.cat(pf, dim=0)
            h = self.dropout(h)
            gf2.append(self.fc8(h))

            #h.append(torch.cat((fea1,fea2,fea3), dim=0))
            #gf2.append(self.fc7(h[j]))
            #gf2_relu.append(self.relu(gf2[j])) 
            #gf2_drop.append(self.dropout(gf2_relu[j]))
            #gf3.append(self.fc8(gf2_drop[j]))

            #contrast_loss_1 = contrast_loss_1+self.loss(sub_f1_gcn[j], sub_f2_gcn_shuffle[j], batch_size=0)
            #contrast_loss_2 = contrast_loss_2+self.loss(sub_f1_gcn[j], sub_f3_gcn_shuffle[j], batch_size=0)
            #contrast_loss_3 = contrast_loss_3+self.loss(sub_f2_gcn[j], sub_f3_gcn_shuffle[j], batch_size=0)
            contrast_loss_1 = contrast_loss_1+self.loss(sub_f1_gcn[j], sub_f1_gcn_shuffle[j], batch_size=0)
            contrast_loss_2 = contrast_loss_2+self.loss(sub_f2_gcn[j], sub_f2_gcn_shuffle[j], batch_size=0)
            contrast_loss_3 = contrast_loss_3+self.loss(sub_f3_gcn[j], sub_f3_gcn_shuffle[j], batch_size=0)
            contrast_loss_clip = contrast_loss_4+self.loss(gf1[j], gf1_shuffle[j], batch_size=0)
            #gf2.append(self.gcn2(gf1_drop[j], edge_index_clip))
        #return f,gf2
        gf2=torch.stack(gf2)
        #sub_f1_gcn=torch.stack(sub_f1_gcn)
        #sub_f2_gcn=torch.stack(sub_f2_gcn)
        #sub_f3_gcn=torch.stack(sub_f3_gcn)
        #sub_f1_gcn_drop=torch.stack(sub_f1_gcn_drop)
        #sub_f2_gcn_drop=torch.stack(sub_f2_gcn_drop)
        #sub_f3_gcn_drop=torch.stack(sub_f3_gcn_drop)

        return contrast_loss_1, contrast_loss_2, contrast_loss_3, contrast_loss_clip, gf2        

class VCOPN_GCN_R_Eight(nn.Module):
    """Video clip order prediction with PFE (Pairwire Feature Extraction), the same as OPN."""
    def __init__(self, base_network, feature_size, tuple_len):
        """
        Args:
            feature_size (int): 512
        """
        super(VCOPN_GCN_R_Eight, self).__init__()

        self.base_network = base_network
        self.feature_size = feature_size
        self.tuple_len = tuple_len
        self.class_num = math.factorial(tuple_len)

        self.gcn_f = GCNConv(self.feature_size, 512)
        #self.gcn_f2 = GCNConv(self.feature_size, 512)
        #self.gcn_f3 = GCNConv(self.feature_size, 512)
        self.gcn1 = GCNConv(self.feature_size, 512)
        #self.gcn2 = GCNConv(512*tuple_len, self.class_num)
        #pair_num = int(tuple_len*(tuple_len-1)/2)
        #self.fc7 = nn.Linear(512*tuple_len, self.class_num)
        self.fc7 = nn.Linear(self.feature_size*2, 512)
        pair_num = int(tuple_len*(tuple_len-1)/2)
        self.fc8 = nn.Linear(512*pair_num, self.class_num)
        #self.fc8 = nn.Linear(512, self.class_num)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = torch.nn.Linear(512, 256)
        self.fc2 = torch.nn.Linear(256, 256)
    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)
        #return z

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / 0.5)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / 0.5)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret
    def repeating(self, x):
        _,c, t, h, w = x.shape
        x_new=[]
        for ind in range(t):
            one_frame = x[:,:,ind,:,:] # c, h, w
            one_frame = torch.unsqueeze(one_frame, 2)# -> c, 1, h, w
            x_new.append(one_frame.repeat(1,1,4,1,1))
        x_repeat=torch.cat([x_new[0],x_new[1]],2)
        return x_repeat

    def adjacent_shuffle(self, x):
        # (C X T x H x W)
        tmp = torch.chunk(x, 2, dim=2)
        order = [0,1]
        ind1 = random.randint(0,1)
        ind2 = (ind1 + random.randint(0,0) + 1) % 2
        order[ind1], order[ind2] = order[ind2], order[ind1]
        x_new = torch.cat((tmp[order[0]], tmp[order[1]]),2)
        return x_new

    def forward(self, tuple, tuple_orders):

        #edge_index_clip_frames=torch.tensor([[0,0,1,2,1,1,1,0,2,3,2,2,2,2,0,1,3,4,3,3,3,3,1,2,4,5,4,4,4,4,2,3,5,6,5,5,5,5,3,4,6,7,6,6,6,6,4,5,7,8,7,7,7,7,5,6,8,9,8,8,8,8,6,7,9,10,9,9,9,9,7,8,9,10,10,10,10,10,8,9,11,12,11,11,11,11,9,10,12,13,12,12,12,12,10,11,13,14,13,13,13,13,11,12,14,15,14,14,14,14,12,13,15,16,15,15,13,14],\
        #    [1,2,0,0,0,2,3,1,1,1,0,1,3,4,2,2,2,2,1,2,4,5,3,3,3,3,2,3,5,6,4,4,4,4,3,4,6,7,5,5,5,5,4,5,7,8,6,6,6,6,5,6,8,9,7,7,7,7,6,7,9,10,8,8,8,8,7,8,9,10,9,9,9,9,8,9,11,12,10,10,10,10,9,10,12,13,11,11,11,11,10,11,13,14,12,12,12,12,11,12,14,15,13,13,13,13,12,13,15,16,14,14,14,14,13,14,15,15]],dtype=torch.long).cuda()
        #edge_index_clip=torch.tensor([[0,1,1,2],[1,0,2,1]],dtype=torch.long).cuda()
        edge_index_clip=[]
        edge_index_clip_1=torch.tensor([[0,1,1,2],[1,0,2,1]],dtype=torch.long).cuda()
        edge_index_clip_2=torch.tensor([[0,2,1,2],[2,0,2,1]],dtype=torch.long).cuda()
        edge_index_clip_3=torch.tensor([[0,1,2,0],[1,0,0,2]],dtype=torch.long).cuda()
        edge_index_clip_4=torch.tensor([[0,1,2,0],[1,0,0,2]],dtype=torch.long).cuda()
        edge_index_clip_5=torch.tensor([[1,2,2,0],[2,1,0,2]],dtype=torch.long).cuda()
        edge_index_clip_6=torch.tensor([[0,1,1,2],[1,0,2,1]],dtype=torch.long).cuda()
        for i in range(tuple_orders.size(0)):
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([0,1,2])):
               edge_index_clip.append(edge_index_clip_1)
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([0,2,1])):
               edge_index_clip.append(edge_index_clip_2)
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([1,0,2])):
                edge_index_clip.append(edge_index_clip_3)
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([1,2,0])):
                edge_index_clip.append(edge_index_clip_4)
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([2,0,1])):
                edge_index_clip.append(edge_index_clip_5)
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([2,1,0])):
                edge_index_clip.append(edge_index_clip_6)
        #edge_index_clip=torch.tensor(np.kron(np.eye(2), edge_index_clip.numpy())).cuda()
        #edge_index_frames=torch.tensor([[0,0,0,1,2,3,1,1,1,0,2,3,2,2,2,0,1,3,3,3,3,0,1,2],\
        #    [1,2,3,0,0,0,0,2,3,1,1,1,0,1,3,2,2,2,0,1,2,3,3,3]],dtype=torch.long).cuda()
        #edge_index_frames=torch.tensor([[0,1,1,2,2,3],\
        #    [1,0,2,1,3,2]],dtype=torch.long).cuda()
        edge_index_frames=torch.tensor([[0,1,1,2,2,3,3,4,4,5,5,6,6,7],\
            [1,0,2,1,3,2,4,3,5,4,6,5,7,6]],dtype=torch.long).cuda()
        sub_f = []  # clip sub-frames features
        sub_shuffle_f = []
        cl= []  # clip features
        for i in range(self.tuple_len):
            cl.append(self.base_network(tuple[:, i, :, :, :, :]))
            for j in range(8):
                #sub_f.append(self.base_network(self.repeating(tuple[:, i, :, 2*j:2*(j+1), :, :])))
                sub_shuffle_f.append(self.base_network(self.repeating(self.adjacent_shuffle(tuple[:, i, :, 2*j:2*(j+1), :, :]))))

        #sub_f1=sub_f[0:8]
        #sub_f2=sub_f[8:16]
        #sub_f3=sub_f[16:24]
        sub_shuffle_f1=sub_shuffle_f[0:8]
        sub_shuffle_f2=sub_shuffle_f[8:16]
        sub_shuffle_f3=sub_shuffle_f[16:24]
        #sub_f1=torch.stack(sub_f1).permute([1,0,2])
        #sub_f2=torch.stack(sub_f2).permute([1,0,2])
        #sub_f3=torch.stack(sub_f3).permute([1,0,2])
        sub_shuffle_f1=torch.stack(sub_shuffle_f1).permute([1,0,2])
        sub_shuffle_f2=torch.stack(sub_shuffle_f2).permute([1,0,2])
        sub_shuffle_f3=torch.stack(sub_shuffle_f3).permute([1,0,2])
        sub_f1_drop=[]
        sub_f2_drop=[]
        sub_f3_drop=[]
        sub_f1_shuffle_drop=[]
        sub_f2_shuffle_drop=[]
        sub_f3_shuffle_drop=[]
        edge_index_frames_drop1=dropout_adj(edge_index_frames, p=0.5)[0]
        edge_index_frames_drop2=dropout_adj(edge_index_frames, p=0.5)[0]
        edge_index_frames_drop3=dropout_adj(edge_index_frames, p=0.5)[0]
        edge_index_frames_shuffle_drop1=dropout_adj(edge_index_frames, p=0.5)[0]
        edge_index_frames_shuffle_drop2=dropout_adj(edge_index_frames, p=0.5)[0]
        edge_index_frames_shuffle_drop3=dropout_adj(edge_index_frames, p=0.5)[0]
        sub_f1_gcn=[]
        sub_f2_gcn=[]
        sub_f3_gcn=[]
        sub_f1_gcn_shuffle=[]
        sub_f2_gcn_shuffle=[]
        sub_f3_gcn_shuffle=[]
        cl=torch.stack(cl).permute([1,0,2])
        gf1=[]
        gf1_relu=[]
        gf1_drop=[]
        gf2=[]
        h=[]
        contrast_loss_1=0.0
        contrast_loss_2=0.0
        contrast_loss_3=0.0
        contrast_loss_4=0.0
        contrast_loss_5=0.0
        contrast_loss_6=0.0
        for j in range(cl.size(0)):
            sub_f1_drop.append(drop_feature(sub_shuffle_f1[j,:,:], 0.5))
            sub_f2_drop.append(drop_feature(sub_shuffle_f2[j,:,:], 0.5))
            sub_f3_drop.append(drop_feature(sub_shuffle_f3[j,:,:], 0.5))
            sub_f1_shuffle_drop.append(drop_feature(sub_shuffle_f1[j,:,:], 0.5))
            sub_f2_shuffle_drop.append(drop_feature(sub_shuffle_f2[j,:,:], 0.5))
            sub_f3_shuffle_drop.append(drop_feature(sub_shuffle_f3[j,:,:], 0.5))
            sub_f1_gcn.append(self.gcn_f(sub_f1_drop[j], edge_index_frames_drop1)) 
            sub_f2_gcn.append(self.gcn_f(sub_f2_drop[j], edge_index_frames_drop2)) 
            sub_f3_gcn.append(self.gcn_f(sub_f3_drop[j], edge_index_frames_drop3)) 
            sub_f1_gcn_shuffle.append(self.gcn_f(sub_f1_shuffle_drop[j], edge_index_frames_shuffle_drop1)) 
            sub_f2_gcn_shuffle.append(self.gcn_f(sub_f2_shuffle_drop[j], edge_index_frames_shuffle_drop2)) 
            sub_f3_gcn_shuffle.append(self.gcn_f(sub_f3_shuffle_drop[j], edge_index_frames_shuffle_drop3)) 
            gf1.append(self.gcn1(cl[j,:,:], edge_index_clip[j])) 
            #temp=torch.stack(gf1_relu).squeeze(0)
            pf = []  # pairwise concat
            for m in range(self.tuple_len):
                for n in range(m+1, self.tuple_len):
                    pf.append(torch.cat([gf1[j][m], gf1[j][n]], dim=0))
            pf = [self.fc7(k) for k in pf]
            #pf = [self.relu(p) for p in pf]
            h = torch.cat(pf, dim=0)
            #h = self.dropout(h)
            gf2.append(self.fc8(h))

            #h.append(torch.cat((gf1[j][0],gf1[j][1],gf1[j][2]), dim=0))
            #gf1_relu.append(self.relu(h[j])) 
            #gf1_drop.append(self.dropout(gf1_relu[j]))
            #gf2.append(self.fc7(h[j]))
            contrast_loss_1 = contrast_loss_1+self.loss(sub_f1_gcn[j], sub_f2_gcn_shuffle[j], batch_size=0)
            contrast_loss_2 = contrast_loss_2+self.loss(sub_f1_gcn[j], sub_f3_gcn_shuffle[j], batch_size=0)
            contrast_loss_3 = contrast_loss_3+self.loss(sub_f2_gcn[j], sub_f3_gcn_shuffle[j], batch_size=0)
            contrast_loss_4 = contrast_loss_4+self.loss(sub_f1_gcn[j], sub_f1_gcn_shuffle[j], batch_size=0)
            contrast_loss_5 = contrast_loss_5+self.loss(sub_f2_gcn[j], sub_f2_gcn_shuffle[j], batch_size=0)
            contrast_loss_6 = contrast_loss_6+self.loss(sub_f3_gcn[j], sub_f3_gcn_shuffle[j], batch_size=0)
            #gf2.append(self.gcn2(gf1_drop[j], edge_index_clip))
        #return f,gf2
        gf2=torch.stack(gf2)
        #sub_f1_gcn=torch.stack(sub_f1_gcn)
        #sub_f2_gcn=torch.stack(sub_f2_gcn)
        #sub_f3_gcn=torch.stack(sub_f3_gcn)
        #sub_f1_gcn_drop=torch.stack(sub_f1_gcn_drop)
        #sub_f2_gcn_drop=torch.stack(sub_f2_gcn_drop)
        #sub_f3_gcn_drop=torch.stack(sub_f3_gcn_drop)

        return contrast_loss_1, contrast_loss_2, contrast_loss_3, contrast_loss_4, contrast_loss_5, contrast_loss_6, gf2         

class VCOPN_GATN_R(nn.Module):
    """Video clip order prediction with PFE (Pairwire Feature Extraction), the same as OPN."""
    def __init__(self, base_network, feature_size, tuple_len):
        """
        Args:
            feature_size (int): 512
        """
        super(VCOPN_GATN_R, self).__init__()

        self.base_network = base_network
        self.feature_size = feature_size
        self.tuple_len = tuple_len
        self.class_num = math.factorial(tuple_len)

        self.gcn_f1 = GATConv(self.feature_size, 512, heads=1)
        self.gcn_f2 = GATConv(self.feature_size, 512, heads=1)
        self.gcn_f3 = GATConv(self.feature_size, 512, heads=1)

        #self.gcn_f1 = GCNConv(self.feature_size, 256)
        #self.gcn_f2 = GCNConv(self.feature_size, 256)
        #self.gcn_f3 = GCNConv(self.feature_size, 256)

        self.gcn1 = GATConv(self.feature_size, 512, heads=1)
        self.gcn2 = GATConv(512, 512, heads=1)

        self.hid = 512
        self.in_head = 4
        self.out_head = 1
        self.gatn1 = GATConv(self.feature_size, 512, heads=self.in_head)
        self.gatn2 = GATConv(self.hid*self.in_head, 512, concat=False, heads=self.out_head)

        self.fc7 = nn.Linear(512*tuple_len, self.class_num)
        #self.fc7 = nn.Linear(self.feature_size*2, 512)
        #pair_num = int(tuple_len*(tuple_len-1)/2)
        #self.fc8 = nn.Linear(512*pair_num, self.class_num)
        #self.fc8 = nn.Linear(512, self.class_num)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = torch.nn.Linear(512, 256)
        self.fc2 = torch.nn.Linear(256, 256)
    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)
        #return z

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / 0.5)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / 0.5)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret
    def repeating(self, x):
        _,c, t, h, w = x.shape
        x_new=[]
        for ind in range(t):
            one_frame = x[:,:,ind,:,:] # c, h, w
            one_frame = torch.unsqueeze(one_frame, 2)# -> c, 1, h, w
            x_new.append(one_frame.repeat(1,1,2,1,1))
        x_repeat=torch.cat([x_new[0],x_new[1],x_new[2],x_new[3]],2)
        return x_repeat

    def adjacent_shuffle(self, x):
        # (C X T x H x W)
        tmp = torch.chunk(x, 4, dim=2)
        order = [0,1,2,3]
        ind1 = random.randint(0,3)
        ind2 = (ind1 + random.randint(0,2) + 1) % 4
        order[ind1], order[ind2] = order[ind2], order[ind1]
        x_new = torch.cat((tmp[order[0]], tmp[order[1]], tmp[order[2]], tmp[order[3]]),2)
        return x_new

    def forward(self, tuple, tuple_orders):

        #edge_index_clip_frames=torch.tensor([[0,0,1,2,1,1,1,0,2,3,2,2,2,2,0,1,3,4,3,3,3,3,1,2,4,5,4,4,4,4,2,3,5,6,5,5,5,5,3,4,6,7,6,6,6,6,4,5,7,8,7,7,7,7,5,6,8,9,8,8,8,8,6,7,9,10,9,9,9,9,7,8,9,10,10,10,10,10,8,9,11,12,11,11,11,11,9,10,12,13,12,12,12,12,10,11,13,14,13,13,13,13,11,12,14,15,14,14,14,14,12,13,15,16,15,15,13,14],\
        #    [1,2,0,0,0,2,3,1,1,1,0,1,3,4,2,2,2,2,1,2,4,5,3,3,3,3,2,3,5,6,4,4,4,4,3,4,6,7,5,5,5,5,4,5,7,8,6,6,6,6,5,6,8,9,7,7,7,7,6,7,9,10,8,8,8,8,7,8,9,10,9,9,9,9,8,9,11,12,10,10,10,10,9,10,12,13,11,11,11,11,10,11,13,14,12,12,12,12,11,12,14,15,13,13,13,13,12,13,15,16,14,14,14,14,13,14,15,15]],dtype=torch.long).cuda()
        #edge_index_clip=torch.tensor([[0,1,1,2],[1,0,2,1]],dtype=torch.long).cuda()
        edge_index_clip=[]
        edge_index_clip_1=torch.tensor([[0,1,1,2],[1,0,2,1]],dtype=torch.long).cuda()
        edge_index_clip_2=torch.tensor([[0,2,1,2],[2,0,2,1]],dtype=torch.long).cuda()
        edge_index_clip_3=torch.tensor([[0,1,2,0],[1,0,0,2]],dtype=torch.long).cuda()
        edge_index_clip_4=torch.tensor([[0,1,2,0],[1,0,0,2]],dtype=torch.long).cuda()
        edge_index_clip_5=torch.tensor([[1,2,2,0],[2,1,0,2]],dtype=torch.long).cuda()
        edge_index_clip_6=torch.tensor([[0,1,1,2],[1,0,2,1]],dtype=torch.long).cuda()
        for i in range(tuple_orders.size(0)):
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([0,1,2])):
               edge_index_clip.append(edge_index_clip_1)
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([0,2,1])):
               edge_index_clip.append(edge_index_clip_2)
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([1,0,2])):
                edge_index_clip.append(edge_index_clip_3)
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([1,2,0])):
                edge_index_clip.append(edge_index_clip_4)
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([2,0,1])):
                edge_index_clip.append(edge_index_clip_5)
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([2,1,0])):
                edge_index_clip.append(edge_index_clip_6)
        #edge_index_clip=torch.tensor(np.kron(np.eye(2), edge_index_clip.numpy())).cuda()
        edge_index_frames=torch.tensor([[0,0,0,1,2,3,1,1,1,0,2,3,2,2,2,0,1,3,3,3,3,0,1,2],\
            [1,2,3,0,0,0,0,2,3,1,1,1,0,1,3,2,2,2,0,1,2,3,3,3]],dtype=torch.long).cuda()
        #edge_index_frames=torch.tensor([[0,1,1,2,2,3],\
        #    [1,0,2,1,3,2]],dtype=torch.long).cuda()
        sub_f = []  # clip sub-frames features
        sub_shuffle_f = []
        cl= []  # clip features
        for i in range(self.tuple_len):
            cl.append(self.base_network(tuple[:, i, :, :, :, :]))
            for j in range(4):
                #sub_f.append(self.base_network(self.repeating(tuple[:, i, :, 4*j:4*(j+1), :, :])))
                sub_f.append(self.base_network(self.repeating(self.adjacent_shuffle(tuple[:, i, :, 4*j:4*(j+1), :, :]))))

        sub_f1=sub_f[0:4]
        sub_f2=sub_f[4:8]
        sub_f3=sub_f[8:12]
        #sub_shuffle_f1=sub_shuffle_f[0:4]
        #sub_shuffle_f2=sub_shuffle_f[4:8]
        #sub_shuffle_f3=sub_shuffle_f[8:12]
        sub_f1=torch.stack(sub_f1).permute([1,0,2])
        sub_f2=torch.stack(sub_f2).permute([1,0,2])
        sub_f3=torch.stack(sub_f3).permute([1,0,2])
        #sub_shuffle_f1=torch.stack(sub_shuffle_f1).permute([1,0,2])
        #sub_shuffle_f2=torch.stack(sub_shuffle_f2).permute([1,0,2])
        #sub_shuffle_f3=torch.stack(sub_shuffle_f3).permute([1,0,2])
        sub_f1_drop=[]
        sub_f2_drop=[]
        sub_f3_drop=[]
        sub_f1_shuffle_drop=[]
        sub_f2_shuffle_drop=[]
        sub_f3_shuffle_drop=[]
        edge_index_frames_drop1=dropout_adj(edge_index_frames, p=0.3)[0]
        edge_index_frames_drop2=dropout_adj(edge_index_frames, p=0.3)[0]
        edge_index_frames_drop3=dropout_adj(edge_index_frames, p=0.3)[0]
        edge_index_frames_shuffle_drop1=dropout_adj(edge_index_frames, p=0.4)[0]
        edge_index_frames_shuffle_drop2=dropout_adj(edge_index_frames, p=0.4)[0]
        edge_index_frames_shuffle_drop3=dropout_adj(edge_index_frames, p=0.4)[0]
        sub_f1_gcn=[]
        sub_f2_gcn=[]
        sub_f3_gcn=[]
        sub_f1_gcn_shuffle=[]
        sub_f2_gcn_shuffle=[]
        sub_f3_gcn_shuffle=[]
        cl=torch.stack(cl).permute([1,0,2])
        gf1=[]
        gf1_relu=[]
        gf1_drop=[]
        gf2=[]
        gf3=[]
        h=[]
        contrast_loss_1=0.0
        contrast_loss_2=0.0
        contrast_loss_3=0.0
        contrast_loss_4=0.0
        contrast_loss_5=0.0
        contrast_loss_6=0.0
        for j in range(cl.size(0)):
            sub_f1_drop.append(drop_feature(sub_f1[j,:,:], 0.2))
            sub_f2_drop.append(drop_feature(sub_f2[j,:,:], 0.2))
            sub_f3_drop.append(drop_feature(sub_f3[j,:,:], 0.2))
            sub_f1_shuffle_drop.append(drop_feature(sub_f1[j,:,:], 0.1))
            sub_f2_shuffle_drop.append(drop_feature(sub_f2[j,:,:], 0.1))
            sub_f3_shuffle_drop.append(drop_feature(sub_f3[j,:,:], 0.1))
            sub_f1_gcn.append(self.gcn_f1(sub_f1_drop[j], edge_index_frames_drop1)) 
            sub_f2_gcn.append(self.gcn_f2(sub_f2_drop[j], edge_index_frames_drop2)) 
            sub_f3_gcn.append(self.gcn_f3(sub_f3_drop[j], edge_index_frames_drop3)) 
            sub_f1_gcn_shuffle.append(self.gcn_f1(sub_f1_shuffle_drop[j], edge_index_frames_shuffle_drop1)) 
            sub_f2_gcn_shuffle.append(self.gcn_f2(sub_f2_shuffle_drop[j], edge_index_frames_shuffle_drop2)) 
            sub_f3_gcn_shuffle.append(self.gcn_f3(sub_f3_shuffle_drop[j], edge_index_frames_shuffle_drop3)) 

            
            #gf1.append(self.gcn1(cl[j,:,:], edge_index_clip[j]))
            #gf2.append(self.gcn2(gf1[j], edge_index_clip[j]))  
 
            gf1.append(self.gatn1(cl[j,:,:], edge_index_clip[j]))
            gf1 = [F.elu(k) for k in gf1]
            gf1 = [self.dropout(p) for p in gf1]
            gf2.append(self.gatn2(gf1[j], edge_index_clip[j]))

            #pf = []  # pairwise concat
            #for m in range(self.tuple_len):
            #    for n in range(m+1, self.tuple_len):
            #        pf.append(torch.cat([gf1[j][m], gf1[j][n]], dim=0))
            #pf = [self.fc7(k) for k in pf]
            #pf = [self.relu(p) for p in pf]
            #h = torch.cat(pf, dim=0)
            #h = self.dropout(h)
            #gf2.append(self.fc8(h))#

            h.append(torch.cat((gf2[j][0],gf2[j][1],gf2[j][2]), dim=0))
            #gf1_relu.append(self.relu(h[j])) 
            #gf1_drop.append(self.dropout(gf1_relu[j]))
            gf3.append(self.fc7(h[j]))

            #contrast_loss_1 = contrast_loss_1+self.loss(sub_f1_gcn[j], sub_f2_gcn_shuffle[j], batch_size=0)
            #contrast_loss_2 = contrast_loss_2+self.loss(sub_f1_gcn[j], sub_f3_gcn_shuffle[j], batch_size=0)
            #contrast_loss_3 = contrast_loss_3+self.loss(sub_f2_gcn[j], sub_f3_gcn_shuffle[j], batch_size=0)
            contrast_loss_1 = contrast_loss_1+self.loss(sub_f1_gcn[j], sub_f1_gcn_shuffle[j], batch_size=0)
            contrast_loss_2 = contrast_loss_2+self.loss(sub_f2_gcn[j], sub_f2_gcn_shuffle[j], batch_size=0)
            contrast_loss_3 = contrast_loss_3+self.loss(sub_f3_gcn[j], sub_f3_gcn_shuffle[j], batch_size=0)
            #gf2.append(self.gcn2(gf1_drop[j], edge_index_clip))
        #return f,gf2
        gf3=torch.stack(gf3)
        #sub_f1_gcn=torch.stack(sub_f1_gcn)
        #sub_f2_gcn=torch.stack(sub_f2_gcn)
        #sub_f3_gcn=torch.stack(sub_f3_gcn)
        #sub_f1_gcn_drop=torch.stack(sub_f1_gcn_drop)
        #sub_f2_gcn_drop=torch.stack(sub_f2_gcn_drop)
        #sub_f3_gcn_drop=torch.stack(sub_f3_gcn_drop)

        return contrast_loss_1, contrast_loss_2, contrast_loss_3, gf3  

class VCOPN_GCN_randomedge(nn.Module):
    """Video clip order prediction with PFE (Pairwire Feature Extraction), the same as OPN."""
    def __init__(self, base_network, feature_size, tuple_len):
        """
        Args:
            feature_size (int): 512
        """
        super(VCOPN_GCN_randomedge, self).__init__()

        self.base_network = base_network
        self.feature_size = feature_size
        self.tuple_len = tuple_len
        self.class_num = math.factorial(tuple_len)

        self.gcn_f1 = GCNConv(self.feature_size, 512)
        self.gcn_f2 = GCNConv(self.feature_size, 512)
        self.gcn_f3 = GCNConv(self.feature_size, 512)
        self.gcn1 = GCNConv(self.feature_size, 512)
        self.gcn2 = GCNConv(512, 512)

        #self.hid = 256
        #self.in_head = 8
        #self.out_head = 1
        #self.gatn1 = GATConv(self.feature_size, 256, heads=self.in_head, dropout=0.6)
        #self.gatn2 = GATConv(self.hid*self.in_head, 256, heads=self.out_head, dropout=0.6)

        self.fc7 = nn.Linear(512*tuple_len, self.class_num)
        #self.fc7 = nn.Linear(self.feature_size*2, 512)
        #pair_num = int(tuple_len*(tuple_len-1)/2)
        #self.fc8 = nn.Linear(512*pair_num, self.class_num)
        #self.fc8 = nn.Linear(512, self.class_num)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = torch.nn.Linear(512, 256)
        self.fc2 = torch.nn.Linear(256, 256)

        self.fusion=SE_Fusion(512,512,512,6)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)
        #return z

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / 0.5)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / 0.5)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret
    def repeating(self, x):
        _,c, t, h, w = x.shape
        x_new=[]
        for ind in range(t):
            one_frame = x[:,:,ind,:,:] # c, h, w
            one_frame = torch.unsqueeze(one_frame, 2)# -> c, 1, h, w
            x_new.append(one_frame.repeat(1,1,2,1,1))
        x_repeat=torch.cat([x_new[0],x_new[1],x_new[2],x_new[3]],2)
        return x_repeat

    def adjacent_shuffle(self, x):
        # (C X T x H x W)
        tmp = torch.chunk(x, 4, dim=2)
        order = [0,1,2,3]
        ind1 = random.randint(0,3)
        ind2 = (ind1 + random.randint(0,2) + 1) % 4
        order[ind1], order[ind2] = order[ind2], order[ind1]
        x_new = torch.cat((tmp[order[0]], tmp[order[1]], tmp[order[2]], tmp[order[3]]),2)
        return x_new

    def adjacent_shuffle_clip(self, x):
        # (C X T x H x W)
        tmp = torch.chunk(x, 16, dim=2)
        ind = torch.randperm(16)
        x_new=[]
        for i in range(16):
            x_new.append(x[:,:,ind[i],:,:])
        x_new= torch.stack(x_new,2)
        return x_new

    def forward(self, tuple, tuple_orders):

        #edge_index_clip_frames=torch.tensor([[0,0,1,2,1,1,1,0,2,3,2,2,2,2,0,1,3,4,3,3,3,3,1,2,4,5,4,4,4,4,2,3,5,6,5,5,5,5,3,4,6,7,6,6,6,6,4,5,7,8,7,7,7,7,5,6,8,9,8,8,8,8,6,7,9,10,9,9,9,9,7,8,9,10,10,10,10,10,8,9,11,12,11,11,11,11,9,10,12,13,12,12,12,12,10,11,13,14,13,13,13,13,11,12,14,15,14,14,14,14,12,13,15,16,15,15,13,14],\
        #    [1,2,0,0,0,2,3,1,1,1,0,1,3,4,2,2,2,2,1,2,4,5,3,3,3,3,2,3,5,6,4,4,4,4,3,4,6,7,5,5,5,5,4,5,7,8,6,6,6,6,5,6,8,9,7,7,7,7,6,7,9,10,8,8,8,8,7,8,9,10,9,9,9,9,8,9,11,12,10,10,10,10,9,10,12,13,11,11,11,11,10,11,13,14,12,12,12,12,11,12,14,15,13,13,13,13,12,13,15,16,14,14,14,14,13,14,15,15]],dtype=torch.long).cuda()
        edge_index_clip=torch.tensor([[0,1,1,2],[1,0,2,1]],dtype=torch.long).cuda()
        #edge_index_clip=[]
        #edge_index_clip_1=torch.tensor([[0,1,1,2],[1,0,2,1]],dtype=torch.long).cuda()
        #edge_index_clip_2=torch.tensor([[0,2,1,2],[2,0,2,1]],dtype=torch.long).cuda()
        #edge_index_clip_3=torch.tensor([[0,1,2,0],[1,0,0,2]],dtype=torch.long).cuda()
        #edge_index_clip_4=torch.tensor([[0,1,2,0],[1,0,0,2]],dtype=torch.long).cuda()
        #edge_index_clip_5=torch.tensor([[1,2,2,0],[2,1,0,2]],dtype=torch.long).cuda()
        #edge_index_clip_6=torch.tensor([[0,1,1,2],[1,0,2,1]],dtype=torch.long).cuda()
        #for i in range(tuple_orders.size(0)):
        #    if torch.equal(tuple_orders[i].cpu(),torch.tensor([0,1,2])):
        #       edge_index_clip.append(edge_index_clip_1)
        #    if torch.equal(tuple_orders[i].cpu(),torch.tensor([0,2,1])):
        #       edge_index_clip.append(edge_index_clip_2)
        #    if torch.equal(tuple_orders[i].cpu(),torch.tensor([1,0,2])):
        #        edge_index_clip.append(edge_index_clip_3)
        #    if torch.equal(tuple_orders[i].cpu(),torch.tensor([1,2,0])):
        #        edge_index_clip.append(edge_index_clip_4)
        #    if torch.equal(tuple_orders[i].cpu(),torch.tensor([2,0,1])):
        #        edge_index_clip.append(edge_index_clip_5)
        #    if torch.equal(tuple_orders[i].cpu(),torch.tensor([2,1,0])):
        #        edge_index_clip.append(edge_index_clip_6)
        #edge_index_clip=torch.tensor(np.kron(np.eye(2), edge_index_clip.numpy())).cuda()
        edge_index_frames=torch.tensor([[0,0,0,1,2,3,1,1,1,0,2,3,2,2,2,0,1,3,3,3,3,0,1,2],\
            [1,2,3,0,0,0,0,2,3,1,1,1,0,1,3,2,2,2,0,1,2,3,3,3]],dtype=torch.long).cuda()
        #edge_index_frames=torch.tensor([[0,1,1,2,2,3],\
        #    [1,0,2,1,3,2]],dtype=torch.long).cuda()
        sub_f = []  # clip sub-frames features
        sub_shuffle_f = []
        cl= []  # clip features
        cl_shuffle= []  # clip features
        for i in range(self.tuple_len):
            cl.append(self.base_network(tuple[:, i, :, :, :, :]))
            #cl_shuffle.append(self.base_network(self.adjacent_shuffle_clip(tuple[:, i, :, :, :, :])))
            for j in range(4):
                #sub_f.append(self.base_network(self.repeating(tuple[:, i, :, 4*j:4*(j+1), :, :])))
                sub_f.append(self.base_network(self.repeating(self.adjacent_shuffle(tuple[:, i, :, 4*j:4*(j+1), :, :]))))

        sub_f1=sub_f[0:4]
        sub_f2=sub_f[4:8]
        sub_f3=sub_f[8:12]
        #sub_shuffle_f1=sub_shuffle_f[0:4]
        #sub_shuffle_f2=sub_shuffle_f[4:8]
        #sub_shuffle_f3=sub_shuffle_f[8:12]
        sub_f1=torch.stack(sub_f1).permute([1,0,2])
        sub_f2=torch.stack(sub_f2).permute([1,0,2])
        sub_f3=torch.stack(sub_f3).permute([1,0,2])
        #sub_shuffle_f1=torch.stack(sub_shuffle_f1).permute([1,0,2])
        #sub_shuffle_f2=torch.stack(sub_shuffle_f2).permute([1,0,2])
        #sub_shuffle_f3=torch.stack(sub_shuffle_f3).permute([1,0,2])
        sub_f1_drop=[]
        sub_f2_drop=[]
        sub_f3_drop=[]
        sub_f1_shuffle_drop=[]
        sub_f2_shuffle_drop=[]
        sub_f3_shuffle_drop=[]
        edge_index_frames_drop1=dropout_adj(edge_index_frames, p=0.1)[0]
        edge_index_frames_drop2=dropout_adj(edge_index_frames, p=0.1)[0]
        edge_index_frames_drop3=dropout_adj(edge_index_frames, p=0.1)[0]
        edge_index_frames_shuffle_drop1=dropout_adj(edge_index_frames, p=0.1)[0]
        edge_index_frames_shuffle_drop2=dropout_adj(edge_index_frames, p=0.1)[0]
        edge_index_frames_shuffle_drop3=dropout_adj(edge_index_frames, p=0.1)[0]
        edge_index_clip_drop=dropout_adj(edge_index_clip, p=0.5)[0]
        sub_f1_gcn=[]
        sub_f2_gcn=[]
        sub_f3_gcn=[]
        sub_f1_gcn_shuffle=[]
        sub_f2_gcn_shuffle=[]
        sub_f3_gcn_shuffle=[]
        cl=torch.stack(cl).permute([1,0,2])
        #cl_shuffle=torch.stack(cl_shuffle).permute([1,0,2])
        gf1=[]
        gf1_relu=[]
        gf1_drop=[]
        gf1_shuffle=[]
        gf2=[]
        gf2_shuffle=[]
        gf3=[]
        h=[]
        contrast_loss_1=0.0
        contrast_loss_2=0.0
        contrast_loss_3=0.0
        contrast_loss_4=0.0
        contrast_loss_5=0.0
        contrast_loss_6=0.0
        contrast_loss_clip=0.0
        for j in range(cl.size(0)):
            sub_f1_drop.append(drop_feature(sub_f1[j,:,:], 0.4))
            sub_f2_drop.append(drop_feature(sub_f2[j,:,:], 0.4))
            sub_f3_drop.append(drop_feature(sub_f3[j,:,:], 0.4))
            sub_f1_shuffle_drop.append(drop_feature(sub_f1[j,:,:], 0.0))
            sub_f2_shuffle_drop.append(drop_feature(sub_f2[j,:,:], 0.0))
            sub_f3_shuffle_drop.append(drop_feature(sub_f3[j,:,:], 0.0))
            sub_f1_gcn.append(self.gcn_f1(sub_f1_drop[j], edge_index_frames_drop1)) 
            sub_f2_gcn.append(self.gcn_f2(sub_f2_drop[j], edge_index_frames_drop2)) 
            sub_f3_gcn.append(self.gcn_f3(sub_f3_drop[j], edge_index_frames_drop3)) 
            sub_f1_gcn_shuffle.append(self.gcn_f1(sub_f1_shuffle_drop[j], edge_index_frames_shuffle_drop1)) 
            sub_f2_gcn_shuffle.append(self.gcn_f2(sub_f2_shuffle_drop[j], edge_index_frames_shuffle_drop2)) 
            sub_f3_gcn_shuffle.append(self.gcn_f3(sub_f3_shuffle_drop[j], edge_index_frames_shuffle_drop3)) 

            
            gf1.append(self.gcn1(cl[j,:,:], edge_index_clip))
            #gf2.append(self.gcn2(gf1[j], edge_index_clip[j]))  
            gf1_drop.append(self.gcn1(drop_feature(cl[j,:,:],0.4), edge_index_clip_drop))
            #gf2_shuffle.append(self.gcn2(gf1_shuffle[j], edge_index_clip[j]))  
 
            #gf1.append(self.gatn1(cl[j,:,:], edge_index_clip[j]))
            #gf1 = [F.elu(k) for k in gf1]
            #gf1 = [self.dropout(p) for p in gf1]
            #gf2.append(self.gatn2(gf1[j], edge_index_clip[j]))

            #pf = []  # pairwise concat
            #for m in range(self.tuple_len):
            #    for n in range(m+1, self.tuple_len):
            #        pf.append(torch.cat([gf1[j][m], gf1[j][n]], dim=0))
            #pf = [self.fc7(k) for k in pf]
            #pf = [self.relu(p) for p in pf]
            #h = torch.cat(pf, dim=0)
            #h = self.dropout(h)
            #gf2.append(self.fc8(h))#

            fea1,fea2,fea3=self.fusion(gf1[j][0],gf1[j][1],gf1[j][2])
            h.append(torch.cat((fea1,fea2,fea3), dim=0))
            #gf1_relu.append(self.relu(h[j])) 
            #gf1_drop.append(self.dropout(gf1_relu[j]))
            gf3.append(self.fc7(h[j]))

            #contrast_loss_1 = contrast_loss_1+self.loss(sub_f1_gcn[j], sub_f2_gcn_shuffle[j], batch_size=0)
            #contrast_loss_2 = contrast_loss_2+self.loss(sub_f1_gcn[j], sub_f3_gcn_shuffle[j], batch_size=0)
            #contrast_loss_3 = contrast_loss_3+self.loss(sub_f2_gcn[j], sub_f3_gcn_shuffle[j], batch_size=0)
            contrast_loss_1 = contrast_loss_1+self.loss(sub_f1_gcn[j], sub_f1_gcn_shuffle[j], batch_size=0)
            contrast_loss_2 = contrast_loss_2+self.loss(sub_f2_gcn[j], sub_f2_gcn_shuffle[j], batch_size=0)
            contrast_loss_3 = contrast_loss_3+self.loss(sub_f3_gcn[j], sub_f3_gcn_shuffle[j], batch_size=0)
            contrast_loss_clip = contrast_loss_4+self.loss(gf1[j], gf1_drop[j], batch_size=0)
            #gf2.append(self.gcn2(gf1_drop[j], edge_index_clip))
        #return f,gf2
        gf3=torch.stack(gf3)
        #sub_f1_gcn=torch.stack(sub_f1_gcn)
        #sub_f2_gcn=torch.stack(sub_f2_gcn)
        #sub_f3_gcn=torch.stack(sub_f3_gcn)
        #sub_f1_gcn_drop=torch.stack(sub_f1_gcn_drop)
        #sub_f2_gcn_drop=torch.stack(sub_f2_gcn_drop)
        #sub_f3_gcn_drop=torch.stack(sub_f3_gcn_drop)

        return contrast_loss_1, contrast_loss_2, contrast_loss_3, contrast_loss_clip, gf3  

class VCOPN_GCN_R3D_R21D(nn.Module):
    """Video clip order prediction with PFE (Pairwire Feature Extraction), the same as OPN."""
    def __init__(self, base_network, feature_size, tuple_len):
        """
        Args:
            feature_size (int): 512
        """
        super(VCOPN_GCN_R3D_R21D, self).__init__()

        self.base_network = base_network
        self.feature_size = feature_size
        self.tuple_len = tuple_len
        self.class_num = math.factorial(tuple_len)
        self.gcn_f = GCNConv(self.feature_size, 512)
        self.gcn_f1 = GCNConv(self.feature_size, 512)
        self.gcn_f2 = GCNConv(self.feature_size, 512)
        self.gcn_f3 = GCNConv(self.feature_size, 512)
        self.gcn1 = GCNConv(self.feature_size, 512)
        self.gcn2 = GCNConv(512, 512)

        #self.hid = 256
        #self.in_head = 8
        #self.out_head = 1
        #self.gatn1 = GATConv(self.feature_size, 256, heads=self.in_head, dropout=0.6)
        #self.gatn2 = GATConv(self.hid*self.in_head, 256, heads=self.out_head, dropout=0.6)

        #self.fc7 = nn.Linear(512*tuple_len, 512)
        #self.fc8 = nn.Linear(512, self.class_num)
        self.fc7 = nn.Linear(self.feature_size*2, 512)
        pair_num = int(tuple_len*(tuple_len-1)/2)
        self.fc8 = nn.Linear(512*pair_num, self.class_num)
        #self.fc8 = nn.Linear(512, self.class_num)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = torch.nn.Linear(512, 512)
        self.fc2 = torch.nn.Linear(512, 512)

        self.fusion=SE_Fusion_batch(512,512,512,6)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)
        #return z

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / 0.5)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / 0.5)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret
    def repeating(self, x):
        _,c, t, h, w = x.shape
        x_new=[]
        for ind in range(t):
            one_frame = x[:,:,ind,:,:] # c, h, w
            one_frame = torch.unsqueeze(one_frame, 2)# -> c, 1, h, w
            x_new.append(one_frame.repeat(1,1,2,1,1))
        x_repeat=torch.cat([x_new[0],x_new[1],x_new[2],x_new[3]],2)
        return x_repeat

    def repeating2(self, x):
        _,c, t, h, w = x.shape
        x_repeat=torch.cat([x,x],2)
        return x_repeat

    def adjacent_shuffle(self, x):
        # (C X T x H x W)
        tmp = torch.chunk(x, 4, dim=2)
        order = [0,1,2,3]
        ind1 = random.randint(0,3)
        ind2 = (ind1 + random.randint(0,2) + 1) % 4
        order[ind1], order[ind2] = order[ind2], order[ind1]
        x_new = torch.cat((tmp[order[0]], tmp[order[1]], tmp[order[2]], tmp[order[3]]),2)
        return x_new

    def adjacent_shuffle_clip(self, x):
        # (C X T x H x W)
        tmp = torch.chunk(x, 16, dim=2)
        ind = torch.randperm(16)
        x_new=[]
        for i in range(16):
            x_new.append(x[:,:,ind[i],:,:])
        x_new= torch.stack(x_new,2)
        return x_new

    def forward(self, tuple, tuple_orders):

        #edge_index_clip_frames=torch.tensor([[0,0,1,2,1,1,1,0,2,3,2,2,2,2,0,1,3,4,3,3,3,3,1,2,4,5,4,4,4,4,2,3,5,6,5,5,5,5,3,4,6,7,6,6,6,6,4,5,7,8,7,7,7,7,5,6,8,9,8,8,8,8,6,7,9,10,9,9,9,9,7,8,9,10,10,10,10,10,8,9,11,12,11,11,11,11,9,10,12,13,12,12,12,12,10,11,13,14,13,13,13,13,11,12,14,15,14,14,14,14,12,13,15,16,15,15,13,14],\
        #    [1,2,0,0,0,2,3,1,1,1,0,1,3,4,2,2,2,2,1,2,4,5,3,3,3,3,2,3,5,6,4,4,4,4,3,4,6,7,5,5,5,5,4,5,7,8,6,6,6,6,5,6,8,9,7,7,7,7,6,7,9,10,8,8,8,8,7,8,9,10,9,9,9,9,8,9,11,12,10,10,10,10,9,10,12,13,11,11,11,11,10,11,13,14,12,12,12,12,11,12,14,15,13,13,13,13,12,13,15,16,14,14,14,14,13,14,15,15]],dtype=torch.long).cuda()
        #edge_index_clip=torch.tensor([[0,1,1,2],[1,0,2,1]],dtype=torch.long).cuda()
        edge_index_clip=[]
        edge_index_clip_drop=[]
        edge_index_clip_1=torch.tensor([[0,1,1,2],[1,0,2,1]],dtype=torch.long).cuda()
        edge_index_clip_2=torch.tensor([[0,2,1,2],[2,0,2,1]],dtype=torch.long).cuda()
        edge_index_clip_3=torch.tensor([[0,1,2,0],[1,0,0,2]],dtype=torch.long).cuda()
        edge_index_clip_4=torch.tensor([[0,1,2,0],[1,0,0,2]],dtype=torch.long).cuda()
        edge_index_clip_5=torch.tensor([[1,2,2,0],[2,1,0,2]],dtype=torch.long).cuda()
        edge_index_clip_6=torch.tensor([[0,1,1,2],[1,0,2,1]],dtype=torch.long).cuda()
        for i in range(tuple_orders.size(0)):
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([0,1,2])):
                edge_index_clip.append(edge_index_clip_1)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_1, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([0,2,1])):
                edge_index_clip.append(edge_index_clip_2)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_2, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([1,0,2])):
                edge_index_clip.append(edge_index_clip_3)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_3, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([1,2,0])):
                edge_index_clip.append(edge_index_clip_4)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_4, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([2,0,1])):
                edge_index_clip.append(edge_index_clip_5)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_5, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([2,1,0])):
                edge_index_clip.append(edge_index_clip_6)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_6, p=0.2)[0])
        #edge_index_clip=torch.tensor(np.kron(np.eye(2), edge_index_clip.numpy())).cuda()
        #edge_index_frames=torch.tensor([[0,0,0,1,2,3,1,1,1,0,2,3,2,2,2,0,1,3,3,3,3,0,1,2],\
        #    [1,2,3,0,0,0,0,2,3,1,1,1,0,1,3,2,2,2,0,1,2,3,3,3]],dtype=torch.long).cuda()
        edge_index_frames=torch.tensor([[0,1,1,2,2,3],\
            [1,0,2,1,3,2]],dtype=torch.long).cuda()
        sub_f = []  # clip sub-frames features
        sub_shuffle_f = []
        cl= []  # clip features
        cl_shuffle= []  # clip features
        for i in range(self.tuple_len):
            cl.append(self.base_network(tuple[:, i, :, :, :, :]))
            #cl_shuffle.append(self.base_network(self.adjacent_shuffle_clip(tuple[:, i, :, :, :, :])))
            for j in range(4):
                sub_f.append(self.base_network(tuple[:, i, :, 4*j:4*(j+1), :, :]))
                #sub_f.append(self.base_network(tuple[:, i, :, 8*j:8*(j+1), :, :]))
                #sub_f.append(self.base_network(self.repeating(self.adjacent_shuffle(tuple[:, i, :, 4*j:4*(j+1), :, :]))))

        sub_f1=sub_f[0:4]
        sub_f2=sub_f[4:8]
        sub_f3=sub_f[8:12]
        #sub_shuffle_f1=sub_shuffle_f[0:4]
        #sub_shuffle_f2=sub_shuffle_f[4:8]
        #sub_shuffle_f3=sub_shuffle_f[8:12]
        sub_f1=torch.stack(sub_f1).permute([1,0,2])
        sub_f2=torch.stack(sub_f2).permute([1,0,2])
        sub_f3=torch.stack(sub_f3).permute([1,0,2])
        #sub_shuffle_f1=torch.stack(sub_shuffle_f1).permute([1,0,2])
        #sub_shuffle_f2=torch.stack(sub_shuffle_f2).permute([1,0,2])
        #sub_shuffle_f3=torch.stack(sub_shuffle_f3).permute([1,0,2])
        sub_f1_drop=[]
        sub_f2_drop=[]
        sub_f3_drop=[]
        sub_f1_shuffle_drop=[]
        sub_f2_shuffle_drop=[]
        sub_f3_shuffle_drop=[]
        edge_index_frames_drop1=dropout_adj(edge_index_frames, p=0.2)[0]
        edge_index_frames_drop2=dropout_adj(edge_index_frames, p=0.2)[0]
        edge_index_frames_drop3=dropout_adj(edge_index_frames, p=0.2)[0]
        edge_index_frames_shuffle_drop1=dropout_adj(edge_index_frames, p=0.2)[0]
        edge_index_frames_shuffle_drop2=dropout_adj(edge_index_frames, p=0.2)[0]
        edge_index_frames_shuffle_drop3=dropout_adj(edge_index_frames, p=0.2)[0]
        sub_f1_gcn=[]
        sub_f2_gcn=[]
        sub_f3_gcn=[]
        sub_f1_gcn_shuffle=[]
        sub_f2_gcn_shuffle=[]
        sub_f3_gcn_shuffle=[]
        #gf2=[]
        #fea1,fea2,fea3=self.fusion(cl[0],cl[1],cl[2])
        pf=[]
        pf.append(torch.cat([cl[0], cl[1]], dim=1))
        pf.append(torch.cat([cl[0], cl[2]], dim=1))
        pf.append(torch.cat([cl[1], cl[2]], dim=1))
        pf = [self.fc7(k) for k in pf]
        pf = [self.relu(p) for p in pf]
        h = torch.cat(pf, dim=1)
        h = self.dropout(h)
        gf2=self.fc8(h)

        cl=torch.stack(cl).permute([1,0,2])
        #cl_shuffle=torch.stack(cl_shuffle).permute([1,0,2])
        gf1=[]
        gf2_relu=[]
        gf2_drop=[]
        gf1_shuffle=[]

        gf2_shuffle=[]
        gf3=[]
        h=[]
        contrast_loss_1=0.0
        contrast_loss_2=0.0
        contrast_loss_3=0.0
        contrast_loss_4=0.0
        contrast_loss_5=0.0
        contrast_loss_6=0.0
        contrast_loss_clip=0.0
        for j in range(cl.size(0)):
            #sub_f1_drop.append(drop_feature(sub_f1[j,:,:], 0.2))
            #sub_f2_drop.append(drop_feature(sub_f2[j,:,:], 0.2))
            #sub_f3_drop.append(drop_feature(sub_f3[j,:,:], 0.2))
            sub_f1_shuffle_drop.append(drop_feature(sub_f1[j,:,:], 0.1))
            sub_f2_shuffle_drop.append(drop_feature(sub_f2[j,:,:], 0.1))
            sub_f3_shuffle_drop.append(drop_feature(sub_f3[j,:,:], 0.1))
            sub_f1_gcn.append(self.gcn_f1(sub_f1[j], edge_index_frames)) 
            sub_f2_gcn.append(self.gcn_f2(sub_f2[j], edge_index_frames)) 
            sub_f3_gcn.append(self.gcn_f3(sub_f3[j], edge_index_frames)) 
            sub_f1_gcn_shuffle.append(self.gcn_f1(sub_f1_shuffle_drop[j], edge_index_frames_shuffle_drop1)) 
            sub_f2_gcn_shuffle.append(self.gcn_f2(sub_f2_shuffle_drop[j], edge_index_frames_shuffle_drop2)) 
            sub_f3_gcn_shuffle.append(self.gcn_f3(sub_f3_shuffle_drop[j], edge_index_frames_shuffle_drop3)) 

            
            gf1.append(self.gcn1(cl[j,:,:], edge_index_clip[j]))
            #gf2.append(self.gcn2(gf1[j], edge_index_clip[j]))  
            gf1_shuffle.append(self.gcn1(drop_feature(cl[j,:,:], 0.1), edge_index_clip_drop[j]))
            #gf2_shuffle.append(self.gcn2(gf1_shuffle[j], edge_index_clip_drop[j]))  
 
            #gf1.append(self.gatn1(cl[j,:,:], edge_index_clip[j]))
            #gf1 = [F.elu(k) for k in gf1]
            #gf1 = [self.dropout(p) for p in gf1]
            #gf2.append(self.gatn2(gf1[j], edge_index_clip[j]))

            #pf = []  # pairwise concat
            #for m in range(self.tuple_len):
            #    for n in range(m+1, self.tuple_len):
            #        pf.append(torch.cat([gf1[j][m], gf1[j][n]], dim=0))
            #pf = [self.fc7(k) for k in pf]
            #pf = [self.relu(p) for p in pf]
            #h = torch.cat(pf, dim=0)
            #h = self.dropout(h)
            #gf2.append(self.fc8(h))#

            #h.append(torch.cat((fea1,fea2,fea3), dim=0))
            #gf2.append(self.fc7(h[j]))
            #gf2_relu.append(self.relu(gf2[j])) 
            #gf2_drop.append(self.dropout(gf2_relu[j]))
            #gf3.append(self.fc8(gf2_drop[j]))

            #contrast_loss_1 = contrast_loss_1+self.loss(sub_f1_gcn[j], sub_f2_gcn_shuffle[j], batch_size=0)
            #contrast_loss_2 = contrast_loss_2+self.loss(sub_f1_gcn[j], sub_f3_gcn_shuffle[j], batch_size=0)
            #contrast_loss_3 = contrast_loss_3+self.loss(sub_f2_gcn[j], sub_f3_gcn_shuffle[j], batch_size=0)
            contrast_loss_1 = contrast_loss_1+self.loss(sub_f1_gcn[j], sub_f1_gcn_shuffle[j], batch_size=0)
            contrast_loss_2 = contrast_loss_2+self.loss(sub_f2_gcn[j], sub_f2_gcn_shuffle[j], batch_size=0)
            contrast_loss_3 = contrast_loss_3+self.loss(sub_f3_gcn[j], sub_f3_gcn_shuffle[j], batch_size=0)
            contrast_loss_clip = contrast_loss_4+self.loss(gf1[j], gf1_shuffle[j], batch_size=0)
            #gf2.append(self.gcn2(gf1_drop[j], edge_index_clip))
        #return f,gf2

        #sub_f1_gcn=torch.stack(sub_f1_gcn)
        #sub_f2_gcn=torch.stack(sub_f2_gcn)
        #sub_f3_gcn=torch.stack(sub_f3_gcn)
        #sub_f1_gcn_drop=torch.stack(sub_f1_gcn_drop)
        #sub_f2_gcn_drop=torch.stack(sub_f2_gcn_drop)
        #sub_f3_gcn_drop=torch.stack(sub_f3_gcn_drop)

        return contrast_loss_1, contrast_loss_2, contrast_loss_3, contrast_loss_clip, gf2        

class TCG_FourClip(nn.Module):
    """Video clip order prediction with PFE (Pairwire Feature Extraction), the same as OPN."""
    def __init__(self, base_network, feature_size, tuple_len):
        """
        Args:
            feature_size (int): 512
        """
        super(TCG_FourClip, self).__init__()

        self.base_network = base_network
        self.feature_size = feature_size
        self.tuple_len = tuple_len
        self.class_num = math.factorial(tuple_len)
        self.gcn_f = GCNConv(self.feature_size, 512)
        self.gcn_f1 = GCNConv(self.feature_size, 512)
        self.gcn_f2 = GCNConv(self.feature_size, 512)
        self.gcn_f3 = GCNConv(self.feature_size, 512)
        self.gcn_f4 = GCNConv(self.feature_size, 512)
        self.gcn_f5 = GCNConv(self.feature_size, 512)
        self.gcn_f6 = GCNConv(self.feature_size, 512)
        self.gcn_f7 = GCNConv(self.feature_size, 512)
        self.gcn_f8 = GCNConv(self.feature_size, 512)
        self.gcn1 = GCNConv(self.feature_size, 512)
        self.gcn1_shuffle = GCNConv(self.feature_size, 512)
        self.gcn2 = GCNConv(512, 512)

        #self.hid = 256
        #self.in_head = 8
        #self.out_head = 1
        #self.gatn1 = GATConv(self.feature_size, 256, heads=self.in_head, dropout=0.6)
        #self.gatn2 = GATConv(self.hid*self.in_head, 256, heads=self.out_head, dropout=0.6)

        self.fc7 = nn.Linear(512*tuple_len, 512)
        self.fc8 = nn.Linear(512, self.class_num)
        #self.fc7 = nn.Linear(self.feature_size*2, 512)
        #pair_num = int(tuple_len*(tuple_len-1)/2)
        #self.fc8 = nn.Linear(512*pair_num, self.class_num)
        #self.fc8 = nn.Linear(512, self.class_num)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = torch.nn.Linear(512, 256)
        self.fc2 = torch.nn.Linear(256, 256)

        self.fusion=SE_Fusion_Four(512,512,512,512,8)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)
        #return z

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / 0.5)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / 0.5)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret
    def repeating(self, x):
        _,c, t, h, w = x.shape
        x_new=[]
        for ind in range(t):
            one_frame = x[:,:,ind,:,:] # c, h, w
            one_frame = torch.unsqueeze(one_frame, 2)# -> c, 1, h, w
            x_new.append(one_frame.repeat(1,1,2,1,1))
        x_repeat=torch.cat([x_new[0],x_new[1],x_new[2],x_new[3]],2)
        return x_repeat

    def repeating2(self, x):
        _,c, t, h, w = x.shape
        x_repeat=torch.cat([x,x],2)
        return x_repeat

    def adjacent_shuffle(self, x):
        # (C X T x H x W)
        tmp = torch.chunk(x, 4, dim=2)
        order = [0,1,2,3]
        ind1 = random.randint(0,3)
        ind2 = (ind1 + random.randint(0,2) + 1) % 4
        order[ind1], order[ind2] = order[ind2], order[ind1]
        x_new = torch.cat((tmp[order[0]], tmp[order[1]], tmp[order[2]], tmp[order[3]]),2)
        return x_new

    def adjacent_shuffle_clip(self, x):
        # (C X T x H x W)
        tmp = torch.chunk(x, 16, dim=2)
        ind = torch.randperm(16)
        x_new=[]
        for i in range(16):
            x_new.append(x[:,:,ind[i],:,:])
        x_new= torch.stack(x_new,2)
        return x_new

    def forward(self, tuple, tuple_orders):

        #edge_index_clip_frames=torch.tensor([[0,0,1,2,1,1,1,0,2,3,2,2,2,2,0,1,3,4,3,3,3,3,1,2,4,5,4,4,4,4,2,3,5,6,5,5,5,5,3,4,6,7,6,6,6,6,4,5,7,8,7,7,7,7,5,6,8,9,8,8,8,8,6,7,9,10,9,9,9,9,7,8,9,10,10,10,10,10,8,9,11,12,11,11,11,11,9,10,12,13,12,12,12,12,10,11,13,14,13,13,13,13,11,12,14,15,14,14,14,14,12,13,15,16,15,15,13,14],\
        #    [1,2,0,0,0,2,3,1,1,1,0,1,3,4,2,2,2,2,1,2,4,5,3,3,3,3,2,3,5,6,4,4,4,4,3,4,6,7,5,5,5,5,4,5,7,8,6,6,6,6,5,6,8,9,7,7,7,7,6,7,9,10,8,8,8,8,7,8,9,10,9,9,9,9,8,9,11,12,10,10,10,10,9,10,12,13,11,11,11,11,10,11,13,14,12,12,12,12,11,12,14,15,13,13,13,13,12,13,15,16,14,14,14,14,13,14,15,15]],dtype=torch.long).cuda()
        #edge_index_clip=torch.tensor([[0,1,1,2],[1,0,2,1]],dtype=torch.long).cuda()
        edge_index_clip=[]
        edge_index_clip_drop=[]
        edge_index_clip_1=torch.tensor([[0,1,1,2,2,3],[1,0,2,1,3,2]],dtype=torch.long).cuda()
        edge_index_clip_2=torch.tensor([[0,1,1,3,2,3],[1,0,3,1,3,2]],dtype=torch.long).cuda()
        edge_index_clip_3=torch.tensor([[0,2,1,2,1,3],[2,0,2,1,3,1]],dtype=torch.long).cuda()
        edge_index_clip_4=torch.tensor([[0,3,1,2,1,3],[3,0,2,1,3,1]],dtype=torch.long).cuda()
        edge_index_clip_5=torch.tensor([[0,2,1,3,2,3],[2,0,3,1,3,2]],dtype=torch.long).cuda()
        edge_index_clip_6=torch.tensor([[0,3,1,2,2,3],[3,0,2,1,3,2]],dtype=torch.long).cuda()

        edge_index_clip_7=torch.tensor([[0,1,0,2,2,3],[1,0,2,0,3,2]],dtype=torch.long).cuda()
        edge_index_clip_8=torch.tensor([[0,1,0,3,2,3],[1,0,3,0,3,2]],dtype=torch.long).cuda()
        edge_index_clip_9=torch.tensor([[0,1,0,2,1,3],[1,0,2,0,3,1]],dtype=torch.long).cuda()
        edge_index_clip_10=torch.tensor([[0,1,0,3,1,2],[1,0,3,0,2,1]],dtype=torch.long).cuda()
        edge_index_clip_11=torch.tensor([[0,2,0,3,1,3],[2,0,3,0,3,1]],dtype=torch.long).cuda()
        edge_index_clip_12=torch.tensor([[0,3,0,3,1,2],[2,0,3,0,2,1]],dtype=torch.long).cuda()

        edge_index_clip_13=torch.tensor([[0,2,0,3,1,2],[2,0,3,0,2,1]],dtype=torch.long).cuda()
        edge_index_clip_14=torch.tensor([[0,2,0,3,1,3],[2,0,3,0,3,1]],dtype=torch.long).cuda()
        edge_index_clip_15=torch.tensor([[0,1,0,3,1,2],[1,0,3,0,2,1]],dtype=torch.long).cuda()
        edge_index_clip_16=torch.tensor([[0,1,0,2,1,3],[1,0,2,0,3,1]],dtype=torch.long).cuda()
        edge_index_clip_17=torch.tensor([[0,1,0,3,2,3],[1,0,3,0,3,2]],dtype=torch.long).cuda()
        edge_index_clip_18=torch.tensor([[0,1,0,2,2,3],[1,0,2,0,3,2]],dtype=torch.long).cuda()

        edge_index_clip_19=torch.tensor([[0,3,1,2,2,3],[3,0,2,1,3,2]],dtype=torch.long).cuda()
        edge_index_clip_20=torch.tensor([[0,2,1,3,2,3],[2,0,3,1,3,2]],dtype=torch.long).cuda()
        edge_index_clip_21=torch.tensor([[0,3,1,2,1,3],[3,0,2,1,3,1]],dtype=torch.long).cuda()
        edge_index_clip_22=torch.tensor([[0,2,1,2,1,3],[2,0,2,1,3,1]],dtype=torch.long).cuda()
        edge_index_clip_23=torch.tensor([[0,1,1,3,2,3],[1,0,3,1,3,2]],dtype=torch.long).cuda()
        edge_index_clip_24=torch.tensor([[0,1,1,2,2,3],[1,0,2,1,3,2]],dtype=torch.long).cuda()

        for i in range(tuple_orders.size(0)):
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([0,1,2,3])):
                edge_index_clip.append(edge_index_clip_1)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_1, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([0,1,3,2])):
                edge_index_clip.append(edge_index_clip_2)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_2, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([0,2,1,3])):
                edge_index_clip.append(edge_index_clip_3)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_3, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([0,2,3,1])):
                edge_index_clip.append(edge_index_clip_4)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_4, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([0,3,1,2])):
                edge_index_clip.append(edge_index_clip_5)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_5, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([0,3,2,1])):
                edge_index_clip.append(edge_index_clip_6)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_6, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([1,0,2,3])):
                edge_index_clip.append(edge_index_clip_7)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_7, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([1,0,3,2])):
                edge_index_clip.append(edge_index_clip_8)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_8, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([1,2,0,3])):
                edge_index_clip.append(edge_index_clip_9)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_9, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([1,2,3,0])):
                edge_index_clip.append(edge_index_clip_10)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_10, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([1,3,0,2])):
                edge_index_clip.append(edge_index_clip_11)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_11, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([1,3,2,0])):
                edge_index_clip.append(edge_index_clip_12)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_12, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([2,0,1,3])):
                edge_index_clip.append(edge_index_clip_13)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_13, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([2,0,3,1])):
                edge_index_clip.append(edge_index_clip_14)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_14, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([2,1,0,3])):
                edge_index_clip.append(edge_index_clip_15)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_15, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([2,1,3,0])):
                edge_index_clip.append(edge_index_clip_16)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_16, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([2,3,0,1])):
                edge_index_clip.append(edge_index_clip_17)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_17, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([2,3,1,0])):
                edge_index_clip.append(edge_index_clip_18)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_18, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([3,0,1,2])):
                edge_index_clip.append(edge_index_clip_19)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_19, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([3,0,2,1])):
                edge_index_clip.append(edge_index_clip_20)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_20, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([3,1,0,2])):
                edge_index_clip.append(edge_index_clip_21)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_21, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([3,1,2,0])):
                edge_index_clip.append(edge_index_clip_22)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_22, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([3,2,0,1])):
                edge_index_clip.append(edge_index_clip_23)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_23, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([3,2,1,0])):
                edge_index_clip.append(edge_index_clip_24)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_24, p=0.2)[0])
        #edge_index_clip=torch.tensor(np.kron(np.eye(2), edge_index_clip.numpy())).cuda()
        #edge_index_frames=torch.tensor([[0,0,0,1,2,3,1,1,1,0,2,3,2,2,2,0,1,3,3,3,3,0,1,2],\
        #    [1,2,3,0,0,0,0,2,3,1,1,1,0,1,3,2,2,2,0,1,2,3,3,3]],dtype=torch.long).cuda()
        edge_index_frames=torch.tensor([[0,1,1,2,2,3],\
            [1,0,2,1,3,2]],dtype=torch.long).cuda()
        sub_f = []  # clip sub-frames features
        sub_shuffle_f = []
        cl= []  # clip features
        cl_shuffle= []  # clip features
        for i in range(self.tuple_len):
            cl.append(self.base_network(tuple[:, i, :, :, :, :]))
            #cl_shuffle.append(self.base_network(self.adjacent_shuffle_clip(tuple[:, i, :, :, :, :])))
            for j in range(4):
                sub_f.append(self.base_network(self.repeating(tuple[:, i, :, 4*j:4*(j+1), :, :])))
                #sub_f.append(self.base_network(tuple[:, i, :, 8*j:8*(j+1), :, :]))
                #sub_f.append(self.base_network(self.repeating(self.adjacent_shuffle(tuple[:, i, :, 4*j:4*(j+1), :, :]))))

        sub_f1=sub_f[0:4]
        sub_f2=sub_f[4:8]
        sub_f3=sub_f[8:12]
        sub_f4=sub_f[12:16]
        #sub_shuffle_f1=sub_shuffle_f[0:4]
        #sub_shuffle_f2=sub_shuffle_f[4:8]
        #sub_shuffle_f3=sub_shuffle_f[8:12]
        sub_f1=torch.stack(sub_f1).permute([1,0,2])
        sub_f2=torch.stack(sub_f2).permute([1,0,2])
        sub_f3=torch.stack(sub_f3).permute([1,0,2])
        sub_f4=torch.stack(sub_f4).permute([1,0,2])
        #sub_shuffle_f1=torch.stack(sub_shuffle_f1).permute([1,0,2])
        #sub_shuffle_f2=torch.stack(sub_shuffle_f2).permute([1,0,2])
        #sub_shuffle_f3=torch.stack(sub_shuffle_f3).permute([1,0,2])
        sub_f1_drop=[]
        sub_f2_drop=[]
        sub_f3_drop=[]
        sub_f4_drop=[]
        sub_f1_shuffle_drop=[]
        sub_f2_shuffle_drop=[]
        sub_f3_shuffle_drop=[]
        sub_f4_shuffle_drop=[]
        edge_index_frames_drop1=dropout_adj(edge_index_frames, p=0.2)[0]
        edge_index_frames_drop2=dropout_adj(edge_index_frames, p=0.2)[0]
        edge_index_frames_drop3=dropout_adj(edge_index_frames, p=0.2)[0]
        edge_index_frames_drop4=dropout_adj(edge_index_frames, p=0.2)[0]
        edge_index_frames_shuffle_drop1=dropout_adj(edge_index_frames, p=0.2)[0]
        edge_index_frames_shuffle_drop2=dropout_adj(edge_index_frames, p=0.2)[0]
        edge_index_frames_shuffle_drop3=dropout_adj(edge_index_frames, p=0.2)[0]
        edge_index_frames_shuffle_drop4=dropout_adj(edge_index_frames, p=0.2)[0]
        sub_f1_gcn=[]
        sub_f2_gcn=[]
        sub_f3_gcn=[]
        sub_f4_gcn=[]
        sub_f1_gcn_shuffle=[]
        sub_f2_gcn_shuffle=[]
        sub_f3_gcn_shuffle=[]
        sub_f4_gcn_shuffle=[]
        cl=torch.stack(cl).permute([1,0,2])
        #cl_shuffle=torch.stack(cl_shuffle).permute([1,0,2])
        gf1=[]
        gf2_relu=[]
        gf2_drop=[]
        gf1_shuffle=[]
        gf2=[]
        gf2_shuffle=[]
        gf3=[]
        h=[]
        contrast_loss_1=0.0
        contrast_loss_2=0.0
        contrast_loss_3=0.0
        contrast_loss_4=0.0
        contrast_loss_5=0.0
        contrast_loss_6=0.0
        contrast_loss_clip=0.0
        for j in range(cl.size(0)):
            #sub_f1_drop.append(drop_feature(sub_f1[j,:,:], 0.2))
            #sub_f2_drop.append(drop_feature(sub_f2[j,:,:], 0.2))
            #sub_f3_drop.append(drop_feature(sub_f3[j,:,:], 0.2))
            sub_f1_shuffle_drop.append(drop_feature(sub_f1[j,:,:], 0.1))
            sub_f2_shuffle_drop.append(drop_feature(sub_f2[j,:,:], 0.1))
            sub_f3_shuffle_drop.append(drop_feature(sub_f3[j,:,:], 0.1))
            sub_f4_shuffle_drop.append(drop_feature(sub_f4[j,:,:], 0.1))
            sub_f1_gcn.append(self.gcn_f1(sub_f1[j], edge_index_frames)) 
            sub_f2_gcn.append(self.gcn_f2(sub_f2[j], edge_index_frames)) 
            sub_f3_gcn.append(self.gcn_f3(sub_f3[j], edge_index_frames)) 
            sub_f4_gcn.append(self.gcn_f4(sub_f4[j], edge_index_frames)) 
            sub_f1_gcn_shuffle.append(self.gcn_f1(sub_f1_shuffle_drop[j], edge_index_frames_shuffle_drop1)) 
            sub_f2_gcn_shuffle.append(self.gcn_f2(sub_f2_shuffle_drop[j], edge_index_frames_shuffle_drop2)) 
            sub_f3_gcn_shuffle.append(self.gcn_f3(sub_f3_shuffle_drop[j], edge_index_frames_shuffle_drop3)) 
            sub_f4_gcn_shuffle.append(self.gcn_f4(sub_f4_shuffle_drop[j], edge_index_frames_shuffle_drop4)) 

            
            gf1.append(self.gcn1(cl[j,:,:], edge_index_clip[j]))
            #gf2.append(self.gcn2(gf1[j], edge_index_clip[j]))  
            gf1_shuffle.append(self.gcn1(drop_feature(cl[j,:,:], 0.1), edge_index_clip_drop[j]))
            #gf2_shuffle.append(self.gcn2(gf1_shuffle[j], edge_index_clip_drop[j]))  
 
            #gf1.append(self.gatn1(cl[j,:,:], edge_index_clip[j]))
            #gf1 = [F.elu(k) for k in gf1]
            #gf1 = [self.dropout(p) for p in gf1]
            #gf2.append(self.gatn2(gf1[j], edge_index_clip[j]))

            #pf = []  # pairwise concat
            #for m in range(self.tuple_len):
            #    for n in range(m+1, self.tuple_len):
            #        pf.append(torch.cat([gf1[j][m], gf1[j][n]], dim=0))
            #pf = [self.fc7(k) for k in pf]
            #pf = [self.relu(p) for p in pf]
            #h = torch.cat(pf, dim=0)
            #h = self.dropout(h)
            #gf2.append(self.fc8(h))#

            fea1,fea2,fea3,fea4=self.fusion(gf1[j][0],gf1[j][1],gf1[j][2],gf1[j][3])
            h.append(torch.cat((fea1,fea2,fea3,fea4), dim=0))
            gf2.append(self.fc7(h[j]))
            gf2_relu.append(self.relu(gf2[j])) 
            gf2_drop.append(self.dropout(gf2_relu[j]))
            gf3.append(self.fc8(gf2_drop[j]))

            #contrast_loss_1 = contrast_loss_1+self.loss(sub_f1_gcn[j], sub_f2_gcn_shuffle[j], batch_size=0)
            #contrast_loss_2 = contrast_loss_2+self.loss(sub_f1_gcn[j], sub_f3_gcn_shuffle[j], batch_size=0)
            #contrast_loss_3 = contrast_loss_3+self.loss(sub_f2_gcn[j], sub_f3_gcn_shuffle[j], batch_size=0)
            contrast_loss_1 = contrast_loss_1+self.loss(sub_f1_gcn[j], sub_f1_gcn_shuffle[j], batch_size=0)
            contrast_loss_2 = contrast_loss_2+self.loss(sub_f2_gcn[j], sub_f2_gcn_shuffle[j], batch_size=0)
            contrast_loss_3 = contrast_loss_3+self.loss(sub_f3_gcn[j], sub_f3_gcn_shuffle[j], batch_size=0)
            contrast_loss_4 = contrast_loss_4+self.loss(sub_f4_gcn[j], sub_f4_gcn_shuffle[j], batch_size=0)
            contrast_loss_clip = contrast_loss_clip+self.loss(gf1[j], gf1_shuffle[j], batch_size=0)
            #gf2.append(self.gcn2(gf1_drop[j], edge_index_clip))
        #return f,gf2
        gf3=torch.stack(gf3)
        #sub_f1_gcn=torch.stack(sub_f1_gcn)
        #sub_f2_gcn=torch.stack(sub_f2_gcn)
        #sub_f3_gcn=torch.stack(sub_f3_gcn)
        #sub_f1_gcn_drop=torch.stack(sub_f1_gcn_drop)
        #sub_f2_gcn_drop=torch.stack(sub_f2_gcn_drop)
        #sub_f3_gcn_drop=torch.stack(sub_f3_gcn_drop)

        return contrast_loss_1, contrast_loss_2, contrast_loss_3, contrast_loss_4, contrast_loss_clip, gf3        

class TCG_FourClip_R3D_R21D(nn.Module):
    """Video clip order prediction with PFE (Pairwire Feature Extraction), the same as OPN."""
    def __init__(self, base_network, feature_size, tuple_len):
        """
        Args:
            feature_size (int): 512
        """
        super(TCG_FourClip_R3D_R21D, self).__init__()

        self.base_network = base_network
        self.feature_size = feature_size
        self.tuple_len = tuple_len
        self.class_num = math.factorial(tuple_len)
        self.gcn_f = GCNConv(self.feature_size, 512)
        self.gcn_f1 = GCNConv(self.feature_size, 512)
        self.gcn_f2 = GCNConv(self.feature_size, 512)
        self.gcn_f3 = GCNConv(self.feature_size, 512)
        self.gcn_f4 = GCNConv(self.feature_size, 512)
        self.gcn_f5 = GCNConv(self.feature_size, 512)
        self.gcn_f6 = GCNConv(self.feature_size, 512)
        self.gcn_f7 = GCNConv(self.feature_size, 512)
        self.gcn_f8 = GCNConv(self.feature_size, 512)
        self.gcn1 = GCNConv(self.feature_size, 512)
        self.gcn1_shuffle = GCNConv(self.feature_size, 512)
        self.gcn2 = GCNConv(512, 512)

        #self.hid = 256
        #self.in_head = 8
        #self.out_head = 1
        #self.gatn1 = GATConv(self.feature_size, 256, heads=self.in_head, dropout=0.6)
        #self.gatn2 = GATConv(self.hid*self.in_head, 256, heads=self.out_head, dropout=0.6)

        #self.fc7 = nn.Linear(512*tuple_len, self.class_num)
        self.fc7 = nn.Linear(self.feature_size*2, 512)
        #self.fc7 = nn.Linear(512*tuple_len, 512)
        #self.fc8 = nn.Linear(512, self.class_num)

        pair_num = int(tuple_len*(tuple_len-1)/2)
        self.fc8 = nn.Linear(512*pair_num, self.class_num)
        #self.fc8 = nn.Linear(512, self.class_num)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = torch.nn.Linear(512, 256)
        self.fc2 = torch.nn.Linear(256, 256)

        self.fusion=SE_Fusion_Four(512,512,512,512,8)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)
        #return z

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / 0.5)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / 0.5)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret
    def repeating(self, x):
        _,c, t, h, w = x.shape
        x_new=[]
        for ind in range(t):
            one_frame = x[:,:,ind,:,:] # c, h, w
            one_frame = torch.unsqueeze(one_frame, 2)# -> c, 1, h, w
            x_new.append(one_frame.repeat(1,1,2,1,1))
        x_repeat=torch.cat([x_new[0],x_new[1],x_new[2],x_new[3]],2)
        return x_repeat

    def repeating2(self, x):
        _,c, t, h, w = x.shape
        x_repeat=torch.cat([x,x],2)
        return x_repeat

    def adjacent_shuffle(self, x):
        # (C X T x H x W)
        tmp = torch.chunk(x, 4, dim=2)
        order = [0,1,2,3]
        ind1 = random.randint(0,3)
        ind2 = (ind1 + random.randint(0,2) + 1) % 4
        order[ind1], order[ind2] = order[ind2], order[ind1]
        x_new = torch.cat((tmp[order[0]], tmp[order[1]], tmp[order[2]], tmp[order[3]]),2)
        return x_new

    def adjacent_shuffle_clip(self, x):
        # (C X T x H x W)
        tmp = torch.chunk(x, 16, dim=2)
        ind = torch.randperm(16)
        x_new=[]
        for i in range(16):
            x_new.append(x[:,:,ind[i],:,:])
        x_new= torch.stack(x_new,2)
        return x_new

    def forward(self, tuple, tuple_orders):

        #edge_index_clip_frames=torch.tensor([[0,0,1,2,1,1,1,0,2,3,2,2,2,2,0,1,3,4,3,3,3,3,1,2,4,5,4,4,4,4,2,3,5,6,5,5,5,5,3,4,6,7,6,6,6,6,4,5,7,8,7,7,7,7,5,6,8,9,8,8,8,8,6,7,9,10,9,9,9,9,7,8,9,10,10,10,10,10,8,9,11,12,11,11,11,11,9,10,12,13,12,12,12,12,10,11,13,14,13,13,13,13,11,12,14,15,14,14,14,14,12,13,15,16,15,15,13,14],\
        #    [1,2,0,0,0,2,3,1,1,1,0,1,3,4,2,2,2,2,1,2,4,5,3,3,3,3,2,3,5,6,4,4,4,4,3,4,6,7,5,5,5,5,4,5,7,8,6,6,6,6,5,6,8,9,7,7,7,7,6,7,9,10,8,8,8,8,7,8,9,10,9,9,9,9,8,9,11,12,10,10,10,10,9,10,12,13,11,11,11,11,10,11,13,14,12,12,12,12,11,12,14,15,13,13,13,13,12,13,15,16,14,14,14,14,13,14,15,15]],dtype=torch.long).cuda()
        #edge_index_clip=torch.tensor([[0,1,1,2],[1,0,2,1]],dtype=torch.long).cuda()
        edge_index_clip=[]
        edge_index_clip_drop=[]
        edge_index_clip_1=torch.tensor([[0,1,1,2,2,3],[1,0,2,1,3,2]],dtype=torch.long).cuda()
        edge_index_clip_2=torch.tensor([[0,1,1,3,2,3],[1,0,3,1,3,2]],dtype=torch.long).cuda()
        edge_index_clip_3=torch.tensor([[0,2,1,2,1,3],[2,0,2,1,3,1]],dtype=torch.long).cuda()
        edge_index_clip_4=torch.tensor([[0,3,1,2,1,3],[3,0,2,1,3,1]],dtype=torch.long).cuda()
        edge_index_clip_5=torch.tensor([[0,2,1,3,2,3],[2,0,3,1,3,2]],dtype=torch.long).cuda()
        edge_index_clip_6=torch.tensor([[0,3,1,2,2,3],[3,0,2,1,3,2]],dtype=torch.long).cuda()

        edge_index_clip_7=torch.tensor([[0,1,0,2,2,3],[1,0,2,0,3,2]],dtype=torch.long).cuda()
        edge_index_clip_8=torch.tensor([[0,1,0,3,2,3],[1,0,3,0,3,2]],dtype=torch.long).cuda()
        edge_index_clip_9=torch.tensor([[0,1,0,2,1,3],[1,0,2,0,3,1]],dtype=torch.long).cuda()
        edge_index_clip_10=torch.tensor([[0,1,0,3,1,2],[1,0,3,0,2,1]],dtype=torch.long).cuda()
        edge_index_clip_11=torch.tensor([[0,2,0,3,1,3],[2,0,3,0,3,1]],dtype=torch.long).cuda()
        edge_index_clip_12=torch.tensor([[0,3,0,3,1,2],[2,0,3,0,2,1]],dtype=torch.long).cuda()

        edge_index_clip_13=torch.tensor([[0,2,0,3,1,2],[2,0,3,0,2,1]],dtype=torch.long).cuda()
        edge_index_clip_14=torch.tensor([[0,2,0,3,1,3],[2,0,3,0,3,1]],dtype=torch.long).cuda()
        edge_index_clip_15=torch.tensor([[0,1,0,3,1,2],[1,0,3,0,2,1]],dtype=torch.long).cuda()
        edge_index_clip_16=torch.tensor([[0,1,0,2,1,3],[1,0,2,0,3,1]],dtype=torch.long).cuda()
        edge_index_clip_17=torch.tensor([[0,1,0,3,2,3],[1,0,3,0,3,2]],dtype=torch.long).cuda()
        edge_index_clip_18=torch.tensor([[0,1,0,2,2,3],[1,0,2,0,3,2]],dtype=torch.long).cuda()

        edge_index_clip_19=torch.tensor([[0,3,1,2,2,3],[3,0,2,1,3,2]],dtype=torch.long).cuda()
        edge_index_clip_20=torch.tensor([[0,2,1,3,2,3],[2,0,3,1,3,2]],dtype=torch.long).cuda()
        edge_index_clip_21=torch.tensor([[0,3,1,2,1,3],[3,0,2,1,3,1]],dtype=torch.long).cuda()
        edge_index_clip_22=torch.tensor([[0,2,1,2,1,3],[2,0,2,1,3,1]],dtype=torch.long).cuda()
        edge_index_clip_23=torch.tensor([[0,1,1,3,2,3],[1,0,3,1,3,2]],dtype=torch.long).cuda()
        edge_index_clip_24=torch.tensor([[0,1,1,2,2,3],[1,0,2,1,3,2]],dtype=torch.long).cuda()

        for i in range(tuple_orders.size(0)):
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([0,1,2,3])):
                edge_index_clip.append(edge_index_clip_1)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_1, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([0,1,3,2])):
                edge_index_clip.append(edge_index_clip_2)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_2, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([0,2,1,3])):
                edge_index_clip.append(edge_index_clip_3)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_3, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([0,2,3,1])):
                edge_index_clip.append(edge_index_clip_4)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_4, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([0,3,1,2])):
                edge_index_clip.append(edge_index_clip_5)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_5, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([0,3,2,1])):
                edge_index_clip.append(edge_index_clip_6)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_6, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([1,0,2,3])):
                edge_index_clip.append(edge_index_clip_7)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_7, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([1,0,3,2])):
                edge_index_clip.append(edge_index_clip_8)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_8, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([1,2,0,3])):
                edge_index_clip.append(edge_index_clip_9)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_9, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([1,2,3,0])):
                edge_index_clip.append(edge_index_clip_10)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_10, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([1,3,0,2])):
                edge_index_clip.append(edge_index_clip_11)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_11, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([1,3,2,0])):
                edge_index_clip.append(edge_index_clip_12)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_12, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([2,0,1,3])):
                edge_index_clip.append(edge_index_clip_13)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_13, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([2,0,3,1])):
                edge_index_clip.append(edge_index_clip_14)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_14, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([2,1,0,3])):
                edge_index_clip.append(edge_index_clip_15)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_15, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([2,1,3,0])):
                edge_index_clip.append(edge_index_clip_16)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_16, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([2,3,0,1])):
                edge_index_clip.append(edge_index_clip_17)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_17, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([2,3,1,0])):
                edge_index_clip.append(edge_index_clip_18)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_18, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([3,0,1,2])):
                edge_index_clip.append(edge_index_clip_19)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_19, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([3,0,2,1])):
                edge_index_clip.append(edge_index_clip_20)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_20, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([3,1,0,2])):
                edge_index_clip.append(edge_index_clip_21)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_21, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([3,1,2,0])):
                edge_index_clip.append(edge_index_clip_22)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_22, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([3,2,0,1])):
                edge_index_clip.append(edge_index_clip_23)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_23, p=0.2)[0])
            if torch.equal(tuple_orders[i].cpu(),torch.tensor([3,2,1,0])):
                edge_index_clip.append(edge_index_clip_24)
                edge_index_clip_drop.append(dropout_adj(edge_index_clip_24, p=0.2)[0])
        #edge_index_clip=torch.tensor(np.kron(np.eye(2), edge_index_clip.numpy())).cuda()
        #edge_index_frames=torch.tensor([[0,0,0,1,2,3,1,1,1,0,2,3,2,2,2,0,1,3,3,3,3,0,1,2],\
        #    [1,2,3,0,0,0,0,2,3,1,1,1,0,1,3,2,2,2,0,1,2,3,3,3]],dtype=torch.long).cuda()
        edge_index_frames=torch.tensor([[0,1,1,2,2,3],\
            [1,0,2,1,3,2]],dtype=torch.long).cuda()
        sub_f = []  # clip sub-frames features
        sub_shuffle_f = []
        cl= []  # clip features
        cl_shuffle= []  # clip features
        for i in range(self.tuple_len):
            cl.append(self.base_network(tuple[:, i, :, :, :, :]))
            #cl_shuffle.append(self.base_network(self.adjacent_shuffle_clip(tuple[:, i, :, :, :, :])))
            for j in range(4):
                sub_f.append(self.base_network(tuple[:, i, :, 4*j:4*(j+1), :, :]))
                #sub_f.append(self.base_network(tuple[:, i, :, 8*j:8*(j+1), :, :]))
                #sub_f.append(self.base_network(self.repeating(self.adjacent_shuffle(tuple[:, i, :, 4*j:4*(j+1), :, :]))))

        sub_f1=sub_f[0:4]
        sub_f2=sub_f[4:8]
        sub_f3=sub_f[8:12]
        sub_f4=sub_f[12:16]
        #sub_shuffle_f1=sub_shuffle_f[0:4]
        #sub_shuffle_f2=sub_shuffle_f[4:8]
        #sub_shuffle_f3=sub_shuffle_f[8:12]
        sub_f1=torch.stack(sub_f1).permute([1,0,2])
        sub_f2=torch.stack(sub_f2).permute([1,0,2])
        sub_f3=torch.stack(sub_f3).permute([1,0,2])
        sub_f4=torch.stack(sub_f4).permute([1,0,2])
        #sub_shuffle_f1=torch.stack(sub_shuffle_f1).permute([1,0,2])
        #sub_shuffle_f2=torch.stack(sub_shuffle_f2).permute([1,0,2])
        #sub_shuffle_f3=torch.stack(sub_shuffle_f3).permute([1,0,2])
        sub_f1_drop=[]
        sub_f2_drop=[]
        sub_f3_drop=[]
        sub_f4_drop=[]
        sub_f1_shuffle_drop=[]
        sub_f2_shuffle_drop=[]
        sub_f3_shuffle_drop=[]
        sub_f4_shuffle_drop=[]
        edge_index_frames_drop1=dropout_adj(edge_index_frames, p=0.2)[0]
        edge_index_frames_drop2=dropout_adj(edge_index_frames, p=0.2)[0]
        edge_index_frames_drop3=dropout_adj(edge_index_frames, p=0.2)[0]
        edge_index_frames_drop4=dropout_adj(edge_index_frames, p=0.2)[0]
        edge_index_frames_shuffle_drop1=dropout_adj(edge_index_frames, p=0.2)[0]
        edge_index_frames_shuffle_drop2=dropout_adj(edge_index_frames, p=0.2)[0]
        edge_index_frames_shuffle_drop3=dropout_adj(edge_index_frames, p=0.2)[0]
        edge_index_frames_shuffle_drop4=dropout_adj(edge_index_frames, p=0.2)[0]
        sub_f1_gcn=[]
        sub_f2_gcn=[]
        sub_f3_gcn=[]
        sub_f4_gcn=[]
        sub_f1_gcn_shuffle=[]
        sub_f2_gcn_shuffle=[]
        sub_f3_gcn_shuffle=[]
        sub_f4_gcn_shuffle=[]
        cl=torch.stack(cl).permute([1,0,2])
        #cl_shuffle=torch.stack(cl_shuffle).permute([1,0,2])
        gf1=[]
        gf2_relu=[]
        gf2_drop=[]
        gf1_shuffle=[]
        gf2=[]
        gf2_shuffle=[]
        gf3=[]
        h=[]
        contrast_loss_1=0.0
        contrast_loss_2=0.0
        contrast_loss_3=0.0
        contrast_loss_4=0.0
        contrast_loss_5=0.0
        contrast_loss_6=0.0
        contrast_loss_clip=0.0
        for j in range(cl.size(0)):
            #sub_f1_drop.append(drop_feature(sub_f1[j,:,:], 0.2))
            #sub_f2_drop.append(drop_feature(sub_f2[j,:,:], 0.2))
            #sub_f3_drop.append(drop_feature(sub_f3[j,:,:], 0.2))
            sub_f1_shuffle_drop.append(drop_feature(sub_f1[j,:,:], 0.1))
            sub_f2_shuffle_drop.append(drop_feature(sub_f2[j,:,:], 0.1))
            sub_f3_shuffle_drop.append(drop_feature(sub_f3[j,:,:], 0.1))
            sub_f4_shuffle_drop.append(drop_feature(sub_f4[j,:,:], 0.1))
            sub_f1_gcn.append(self.gcn_f1(sub_f1[j], edge_index_frames)) 
            sub_f2_gcn.append(self.gcn_f2(sub_f2[j], edge_index_frames)) 
            sub_f3_gcn.append(self.gcn_f3(sub_f3[j], edge_index_frames)) 
            sub_f4_gcn.append(self.gcn_f4(sub_f4[j], edge_index_frames)) 
            sub_f1_gcn_shuffle.append(self.gcn_f1(sub_f1_shuffle_drop[j], edge_index_frames_shuffle_drop1)) 
            sub_f2_gcn_shuffle.append(self.gcn_f2(sub_f2_shuffle_drop[j], edge_index_frames_shuffle_drop2)) 
            sub_f3_gcn_shuffle.append(self.gcn_f3(sub_f3_shuffle_drop[j], edge_index_frames_shuffle_drop3)) 
            sub_f4_gcn_shuffle.append(self.gcn_f4(sub_f4_shuffle_drop[j], edge_index_frames_shuffle_drop4)) 

            
            gf1.append(self.gcn1(cl[j,:,:], edge_index_clip[j]))
            #gf2.append(self.gcn2(gf1[j], edge_index_clip[j]))  
            gf1_shuffle.append(self.gcn1(drop_feature(cl[j,:,:], 0.1), edge_index_clip_drop[j]))
            #gf2_shuffle.append(self.gcn2(gf1_shuffle[j], edge_index_clip_drop[j]))  
 
            #gf1.append(self.gatn1(cl[j,:,:], edge_index_clip[j]))
            #gf1 = [F.elu(k) for k in gf1]
            #gf1 = [self.dropout(p) for p in gf1]
            #gf2.append(self.gatn2(gf1[j], edge_index_clip[j]))

            #pf = []  # pairwise concat
            #for m in range(self.tuple_len):
            #    for n in range(m+1, self.tuple_len):
            #        pf.append(torch.cat([gf1[j][m], gf1[j][n]], dim=0))
            #pf = [self.fc7(k) for k in pf]
            #pf = [self.relu(p) for p in pf]
            #h = torch.cat(pf, dim=0)
            #h = self.dropout(h)
            #gf2.append(self.fc8(h))

            fea1,fea2,fea3,fea4=self.fusion(gf1[j][0],gf1[j][1],gf1[j][2],gf1[j][3])
            pf=[]
            pf.append(torch.cat([fea1, fea2], dim=0))
            pf.append(torch.cat([fea1, fea3], dim=0))
            pf.append(torch.cat([fea1, fea4], dim=0))
            pf.append(torch.cat([fea2, fea3], dim=0))
            pf.append(torch.cat([fea2, fea4], dim=0))
            pf.append(torch.cat([fea3, fea4], dim=0))
            pf = [self.fc7(k) for k in pf]
            pf = [self.relu(p) for p in pf]
            h = torch.cat(pf, dim=0)
            h = self.dropout(h)
            gf2.append(self.fc8(h))

            #h.append(torch.cat((fea1,fea2,fea3,fea4), dim=0))
            #gf2.append(self.fc7(h[j]))
            #gf2_relu.append(self.relu(gf2[j])) 
            #gf2_drop.append(self.dropout(gf2_relu[j]))
            #gf3.append(self.fc8(gf2_drop[j]))

            #contrast_loss_1 = contrast_loss_1+self.loss(sub_f1_gcn[j], sub_f2_gcn_shuffle[j], batch_size=0)
            #contrast_loss_2 = contrast_loss_2+self.loss(sub_f1_gcn[j], sub_f3_gcn_shuffle[j], batch_size=0)
            #contrast_loss_3 = contrast_loss_3+self.loss(sub_f2_gcn[j], sub_f3_gcn_shuffle[j], batch_size=0)
            contrast_loss_1 = contrast_loss_1+self.loss(sub_f1_gcn[j], sub_f1_gcn_shuffle[j], batch_size=0)
            contrast_loss_2 = contrast_loss_2+self.loss(sub_f2_gcn[j], sub_f2_gcn_shuffle[j], batch_size=0)
            contrast_loss_3 = contrast_loss_3+self.loss(sub_f3_gcn[j], sub_f3_gcn_shuffle[j], batch_size=0)
            contrast_loss_4 = contrast_loss_4+self.loss(sub_f4_gcn[j], sub_f4_gcn_shuffle[j], batch_size=0)
            contrast_loss_clip = contrast_loss_clip+self.loss(gf1[j], gf1_shuffle[j], batch_size=0)
            #gf2.append(self.gcn2(gf1_drop[j], edge_index_clip))
        #return f,gf2
        gf2=torch.stack(gf2)
        #sub_f1_gcn=torch.stack(sub_f1_gcn)
        #sub_f2_gcn=torch.stack(sub_f2_gcn)
        #sub_f3_gcn=torch.stack(sub_f3_gcn)
        #sub_f1_gcn_drop=torch.stack(sub_f1_gcn_drop)
        #sub_f2_gcn_drop=torch.stack(sub_f2_gcn_drop)
        #sub_f3_gcn_drop=torch.stack(sub_f3_gcn_drop)

        return contrast_loss_1, contrast_loss_2, contrast_loss_3, contrast_loss_4, contrast_loss_clip, gf2        