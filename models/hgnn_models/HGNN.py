from torch import nn
#from models import HGNN_conv
from models.hgnn_models import HGNN_conv
import torch.nn.functional as F


class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid).half()
        self.hgc2 = HGNN_conv(n_hid, n_class).half()

    def forward(self, x, G):
        #G = construct_G_from_H(G)
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        return x
