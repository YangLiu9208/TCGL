import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class xcorr2(nn.Module):

    '''
    Correlates two images. Images must be of the same size.
    '''

    def __init__(self, zero_mean_normalize=True):
        super(xcorr2, self).__init__()
        #self.InstanceNorm = nn.InstanceNorm1d(512, affine=False, track_running_stats=False)
        self.zero_mean_normalize = zero_mean_normalize


    def forward(self, x1, x2):
        """
        Args:
            x1 (torch.Tensor): Batch of Img1 of dimensions [B, C, H, W].
            x2 (torch.Tensor): Batch of Img2 of dimensions [B, C, H, W].
        Returns:
            scores (torch.Tensor): The correlation scores for the pairs. The output shape is [8, 1].
        """

        if x1.shape == x2.shape:
            scores = self.match_corr_same_size(x1, x2)
        else:
            scores = self.match_corr(x1, x2)
            
        return scores


    def match_corr_same_size(self, x1, x2):
        N, D= x2.shape
        x1_mu = x1.mean(dim=0)
        x1_std = x1.std(dim=0)
        x2_mu = x2.mean(dim=0)
        x2_std = x2.std(dim=0)
        if self.zero_mean_normalize:
            x1_norm = (x1-x1_mu.repeat(N,1))/x1_std.repeat(N,1)
            x2_norm = (x2-x2_mu.repeat(N,1))/x2_std.repeat(N,1)
            #x1_norm = F.normalize(x1)
            #x2_norm = F.normalize(x2)
            #x1_norm=self.InstanceNorm(x1)
            #x2_norm=self.InstanceNorm(x2)

        c= torch.matmul(x1_norm.t(), x2_norm)
        c_diff=(c-torch.eye(D).cuda()).pow(2)
        c_diag = torch.diag(c_diff)
        a_diag = torch.diag_embed(c_diag)
        c_diff = c_diff - a_diag
        c_diff=c_diff/N
        c_diff=c_diff+a_diag
        loss=c_diff.mean()
        return loss


    def match_corr(self, embed_ref, embed_srch):
        """ Matches the two embeddings using the correlation layer. As per usual
        it expects input tensors of the form [B, C, H, W].

        Args:
            embed_ref: (torch.Tensor) The embedding of the reference image, or
                the template of reference (the average of many embeddings for
                example).
            embed_srch: (torch.Tensor) The embedding of the search image.

        Returns:
            match_map: (torch.Tensor) The correlation between
        """
        b, c, h, w = embed_srch.shape

        # Here the correlation layer is implemented using a trick with the
        # conv2d function using groups in order to do the correlation with
        # batch dimension. Basically we concatenate each element of the batch
        # in the channel dimension for the search image (making it
        # [1 x (B.C) x H' x W']) and setting the number of groups to the size of
        # the batch. This grouped convolution/correlation is equivalent to a
        # correlation between the two images, though it is not obvious.
        if self.zero_mean_normalize:
            embed_ref = self.InstanceNorm(embed_ref)
            embed_srch = self.InstanceNorm(embed_srch)

        # Has problems with mixed-precision training
        match_map = F.conv2d(embed_srch.view(1, b * c, h, w), embed_ref, groups=b).float() 
        match_map /= (self.img_size*self.img_size*c)

        # Here we reorder the dimensions to get back the batch dimension.
        match_map = match_map.permute(1, 0, 2, 3)
        
        return match_map