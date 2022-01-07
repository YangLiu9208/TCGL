import torch,math 
import torch.nn.functional as F
def get_1d_dct(i, freq, L):
    result = math.cos(math.pi * freq * (i + 0.5) / L) / math.sqrt(L)
    if freq == 0: 
        return result 
    else: 
        return result * math.sqrt(2) 
def get_dct_weights( width, height, channel, fidx_u= [0,0,6,0,0,1,1,4,5,1,3,0,0,0,2,3], fidx_v= [0,1,0,5,2,0,2,0,0,6,0,4,6,3,2,5]):
    # width : width of input 
    # height : height of input 
    # channel : channel of input 
    # fidx_u : horizontal indices of selected fequency 
    # according to the paper, should be [0,0,6,0,0,1,1,4,5,1,3,0,0,0,2,3]
    # fidx_v : vertical indices of selected fequency 
    # according to the paper, should be [0,1,0,5,2,0,2,0,0,6,0,4,6,3,2,5]
    # [0,0],[0,1],[6,0],[0,5],[0,2],[1,0],[1,2],[4,0],
    # [5,0],[1,6],[3,0],[0,4],[0,6],[0,3],[2,2],[3,5],
    scale_ratio = width//7
    fidx_u = [u*scale_ratio for u in fidx_u]
    fidx_v = [v*scale_ratio for v in fidx_v]
    dct_weights = torch.zeros(1, channel, width, height) 
    c_part = channel // len(fidx_u) 
    # split channel for multi-spectal attention 
    for i, (u_x, v_y) in enumerate(zip(fidx_u, fidx_v)): 
        for t_x in range(width): 
            for t_y in range(height): 
                dct_weights[:, i * c_part: (i+1)*c_part, t_x, t_y]\
                =get_1d_dct(t_x, u_x, width) * get_1d_dct(t_y, v_y, height) 
    # Eq. 7 in our paper 
    return dct_weights 
from torch import nn


class FcaLayer(nn.Module):
    def __init__(self,
                 channel,
                 reduction,width,height):
        super(FcaLayer, self).__init__()
        self.width = width
        self.height = height
        self.register_buffer('pre_computed_dct_weights',get_dct_weights(self.width,self.height,channel))
        #self.register_parameter('pre_computed_dct_weights',torch.nn.Parameter(get_dct_weights(width,height,channel)))
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x,(self.height,self.width))
        y = torch.sum(y*self.pre_computed_dct_weights,dim=(2,3))
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
