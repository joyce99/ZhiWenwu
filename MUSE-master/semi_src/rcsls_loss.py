import torch
from torch import nn
from torch.autograd import Variable
class RCSLS(nn.Module):

    def __init__(self):
        super(RCSLS, self).__init__()
    
    def getknn(self, sc,  k=10):
        sidx = sc.topk(10, 1, True)[1][:, :k] 
        f = (sc[torch.arange(sc.shape[0])[:, None], sidx]).sum()
        return f / k

    def forward(self,  X_trans, Y_tgt, Z_src,Z_tgt,back_emb, knn=10):#Z_src.shape = (1000000,300)  X_src=(1000,300)
        src_emb = X_trans / X_trans.norm(2, 1, keepdim=True).expand_as(X_trans)
        tgt_emb = Y_tgt / Y_tgt.norm(2, 1, keepdim=True).expand_as(Y_tgt)
        Z_src = Z_src / Z_src.norm(2, 1, keepdim=True).expand_as(Z_src)
        Z_tgt = Z_tgt / Z_tgt.norm(2, 1, keepdim=True).expand_as(Z_tgt)
        back_emb = back_emb / back_emb.norm(2, 1, keepdim=True).expand_as(back_emb)
        f = 2* (src_emb * tgt_emb).sum()  # 2 * (X_trans * Y_tgt).sum()
        fk0 = self.getknn(src_emb.mm(Z_tgt.t()), knn)
        fk1 = self.getknn(back_emb.mm(Z_src.t()),  knn)
        f = f - fk0 - fk1
        return (-f / X_trans.shape[0])


