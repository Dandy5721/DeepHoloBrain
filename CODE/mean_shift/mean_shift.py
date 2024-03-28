import torch
from torch import nn
from spdnet.spd import Normalize
import numpy as np

class GBMS_RNN(nn.Module):
    def __init__(self, bandwidth=0.1, normalize=True):
        super(GBMS_RNN, self).__init__()
        self.bandwidth = nn.Parameter(torch.tensor(bandwidth))
        self.normalize = None
        if normalize:
            self.normalize = Normalize()

    def forward(self, X):
        bandwidth = self.bandwidth
        if self.normalize:
            W = torch.exp((X @ X.transpose(-2, -1) - 1) / (bandwidth * bandwidth))
        else:
            pair_dis = torch.cdist(X, X)
            pair_dis_square = pair_dis**2
            W = torch.exp(-0.5 * pair_dis_square / (bandwidth * bandwidth))
        D = W.sum(dim=-1).diag_embed()
        D_inv = D.inverse()
        X = (X.transpose(-2, -1) @ W @ D_inv).transpose(-2, -1)
        if self.normalize:
            X = self.normalize(X)
        output = X
        return output

class SPD_GBMS_RNN(nn.Module):
    def __init__(self, bandwidth=0.5):
        super(SPD_GBMS_RNN, self).__init__()
        self.bandwidth = nn.Parameter(torch.tensor(bandwidth))

    def log(self, X):
        S, U = X.symeig(eigenvectors=True)
        S = S.log().diag_embed()
        return U @ S @ U.transpose(-2, -1)

    def exp(self, X):
        S, U = X.symeig(eigenvectors=True)
        S = S.exp().diag_embed()
        return U @ S @ U.transpose(-2, -1)

    def logm(self, X, Y):
        return self.log(Y) - self.log(X)

    def expm(self, X, Y):
        return self.exp(self.log(X) + Y)

    def forward(self, X):
        bandwidth = self.bandwidth
        try:
            log_X = self.log(X)
        except RuntimeError:
            print(X)
            np.save('error_data.npy', X.detach().numpy())
            exit(-1)
        pair_dis = torch.norm(log_X.unsqueeze(-4) - log_X.unsqueeze(-3) + 1e-7, p='fro', dim=(-2, -1))
        log_Y_X = log_X.unsqueeze(-4) - log_X.unsqueeze(-3)
        pair_dis_square = pair_dis ** 2
        W = torch.exp(-0.5 * pair_dis_square / (bandwidth * bandwidth))
        D = W.sum(dim=-1).diag_embed()
        D_inv = D.inverse()

        M = ((log_Y_X.permute(2, 3, 0, 1) @ W).diagonal(dim1=-2, dim2=-1) @ D_inv).permute(2, 0, 1)
        output = self.expm(X, M)

        # output2 = []
        # for y in X:
        #     sum_weights = 0
        #     ms_vector = torch.zeros_like(y)
        #     for x in X:
        #         dis = torch.norm(self.log(x) - self.log(y))
        #         weight = torch.exp(-0.5 * dis * dis / (bandwidth * bandwidth))
        #         delta = self.log(x) - self.log(y)
        #         ms_vector += weight * delta
        #         sum_weights += weight
        #     ms_vector = ms_vector / sum_weights
        #     output2.append(self.expm(y, ms_vector))
        # output2 = torch.stack(output2)

        return output

def cosine_similarity(input):
    output = input @ input.transpose(-2, -1) * 0.5 + 0.5
    return output


def similarity_loss(input, targets, alpha=0):
    similarity = cosine_similarity(input)
    identity_matrix = targets.unsqueeze(-2) == targets.unsqueeze(-2).transpose(-2, -1)
    loss = (1 - similarity) * identity_matrix + torch.clamp(
        similarity - alpha, min=0
    ) * (~identity_matrix)
    loss = torch.mean(loss)
    return loss

def distance_loss(inputs, targets):
    identity_matrix = targets.unsqueeze(0) == targets.unsqueeze(0).T
    # S, U = inputs.symeig(eigenvectors=True)
    # S = S.log().diag_embed()
    L, U = torch.linalg.eigh(inputs, UPLO='U')
    S = torch.diag_embed(L.log()) 
    log_X = U @ S @ U.transpose(-2, -1)

    pair_dis = torch.norm(log_X.unsqueeze(-4) - log_X.unsqueeze(-3) + 1e-7, p='fro', dim=(-2, -1))
    # print(pair_dis)
    loss = pair_dis * identity_matrix - pair_dis * (~identity_matrix)
    loss = loss.mean()

    return loss