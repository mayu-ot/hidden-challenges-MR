# import chainer
# import chainer.links as L
# import chainer.functions as F
# import chainer.distributions as D
# from chainer import reporter
import torch
from torch import nn
from torch.distributions import MultivariateNormal

class CandidateGenerator(chainer.Chain):
    def __init__(self, query_net, encoder, k=1):
        super(CandidateGenerator, self).__init__()
        
        self.k = k
        
        with self.init_scope():
            self.query_net = query_net
            self.encoder = encoder
            
    def sample(self, x, n):
        h = self.query_net(x)
        d = self.encoder(h)
        y = d.sample(n)
        prob = d.prob(y)
        return y, prob
    
    def __call__(self, x, y):
        h = self.query_net(x)
        d = self.encoder(h)
        reconstr = F.mean(
            d.log_prob(
                F.broadcast_to(y, (self.k,)+y.shape)
            )
        )
        
        loss = -reconstr
        reporter.report({'loss': loss}, self)
        
        return loss
    
class QueryEmbedNet(nn.Module):
    def __init__(self, n_vocab):
        super(QueryEmbedNet, self).__init__()
        
        self.emb = nn.Embedding(n_vocab, embedding_dim=100, padding_idx=0)
        self.lstm = nn.LSTM(100, 100)
        
    def __call__(self, x):
        embeddings = self.emb(x)
        embeddings=embeddings.transpose(1,0)
        _, (h_n, _) = self.lstm(embeddings)
        
        return h_n.sqeeze()
        

class BoundaryDist(nn.Module):
    def __init__(self):
        super(BoundaryDist, self).__init__()
        self.scale_l = nn.Linear(10, 2)
        self.cov_l = nn.Linear(10, 1)
        self.loc_l = nn.Linear(10, 2)
        self.relu = nn.ReLU()
        
    def __call__(self, x):
        scale = self.relu(self.scale_l(x))
        cov = self.cov_l(x)
        
        scale = torch.diag_embed(scale)
        scale[:,1,0] = torch.flatten(cov)
        
        
        loc = self.loc_l(x)
        
        dist = MultivariateNormal(loc, scale_tril=scale)
        return dist