import torch
from torch import nn
from spdnet import StiefelParameter, SPDParameter
import numpy as np
from spdnet.wavalet import create_normalized_matrices
from utils import fc2vector
import time

# def log(X):
#     S, U = X.symeig(eigenvectors=True)
#     S = S.log().diag_embed()
#     return U @ S @ U.transpose(-2, -1)

# def exp(X):
#     S, U = X.symeig(eigenvectors=True)
#     S = S.exp().diag_embed()
#     return U @ S @ U.transpose(-2, -1)
def log(X):
    L, U = torch.linalg.eigh(X, UPLO='U')
    L = torch.diag_embed(L.log())
    return U @ L @ U.transpose(-2, -1)

def exp(X):
    L, U = torch.linalg.eigh(X, UPLO='U')
    L = torch.diag_embed(L.exp())
    return U @ L @ U.transpose(-2, -1)

def sqrtm(X):
    return exp(0.5 * log(X))

def logm(X, Y):
    C = sqrtm(X)
    C_inv = C.inverse()
    return C @ log(C_inv @ Y @ C_inv) @ C

def expm(X, Y):
    C = sqrtm(X)
    C_inv = C.inverse()
    return C @ exp(C_inv @ Y @ C_inv) @ C

def mean(spds, num_iter=20):
    mean = torch.mean(spds, dim=0)
    for iter in range(num_iter):
        c = sqrtm(mean)
        c_inv = c.inverse()
        # tangent_mean = np.asarray(Parallel(n_jobs=-1)(delayed(scipy.linalg.logm)(c_inv @ spd @ c_inv) for spd in spds))
        tangent_mean = log(c_inv @ spds @ c_inv)
        tangent_mean = torch.mean(c @ tangent_mean @ c, dim=0)
        mean = c @ exp(c_inv @ tangent_mean @ c_inv) @ c
        # eps = la.norm(tangent_mean, ord='fro')
        # if eps < 1e-6:
        #     break
    return mean

class SPDTransform(nn.Module):
    def __init__(self, input_size, output_size, in_channels=1):
        super(SPDTransform, self).__init__()

        if in_channels > 1:
            self.weight = StiefelParameter(
                torch.Tensor(in_channels, input_size, output_size), requires_grad=True
            )
        else:
            self.weight = StiefelParameter(
                torch.Tensor(input_size, output_size), requires_grad=True
            )
        nn.init.orthogonal_(self.weight)

    def forward(self, input):
        weight = self.weight
        output = weight.transpose(-2, -1) @ input @ weight
        return output


class SPDTransform_scaled(nn.Module):
    def __init__(self, input_size, output_size, in_channels=1):
        super(SPDTransform_scaled, self).__init__()

        if in_channels > 1:
            self.weight = StiefelParameter(
                torch.Tensor(in_channels, input_size, output_size), requires_grad=True
            )
        else:
            self.weight = StiefelParameter(
                torch.Tensor(input_size, output_size), requires_grad=True
            )
        nn.init.orthogonal_(self.weight)
        
        self.scale = nn.Parameter(torch.randn(in_channels))

        
    def forward(self, input, sc):
        # weight = self.weight
        graph_wavelet = create_normalized_matrices(sc)
        weight = torch.matrix_exp(-self.scale * graph_wavelet)
        output = weight.transpose(-2, -1) @ input @ weight
        return output

# class SPDTangentSpace(nn.Module):
#     def __init__(self):
#         super(SPDTangentSpace, self).__init__()

#     def forward(self, input):
#         s, u = torch.linalg.eigh(input)
#         s = s.log().diag_embed()
#         output = u @ s @ u.transpose(-2, -1)
#         print((torch.flatten(output, 1)).shape)
#         return torch.flatten(output, 1)


class SPDVectorize(nn.Module):

    def __init__(self, vectorize_all=True):
        super(SPDVectorize, self).__init__()
        self.register_buffer('vectorize_all', torch.tensor(vectorize_all))

    def forward(self, input):
        row_idx, col_idx = np.triu_indices(input.shape[-1])
        output = input[..., row_idx, col_idx]

        if self.vectorize_all:
            output = torch.flatten(output, 1)
        return output

class SPDTangentSpace(nn.Module):

    def __init__(self, vectorize=True, vectorize_all=True):
        super(SPDTangentSpace, self).__init__()
        self.vectorize = vectorize
        if vectorize:
            self.vec = SPDVectorize(vectorize_all=vectorize_all)

    def forward(self, input):

        # try:
        s, u = torch.linalg.eigh(input)
            # u, s, v = input.svd()
        # except:
            # print(input)
            # torch.save(input, 'error.pt')

        s = s.log().diag_embed()
        if s.isnan().any():
            print('SPDTangentSpace log negative')
            raise ValueError
        output = u @ s @ u.transpose(-2, -1)

        if self.vectorize:
            output = self.vec(output)

        return output
    
class SPDExpMap(nn.Module):
    def __init__(self):
        super(SPDExpMap, self).__init__()

    def forward(self, input):
        s, u = torch.linalg.eigh(input)
        s = s.exp().diag_embed()
        output = u @ s @ u.transpose(-2, -1)
        return output


class SPDRectified(nn.Module):
    def __init__(self, epsilon=1e-4):
        super(SPDRectified, self).__init__()
        self.register_buffer('epsilon', torch.DoubleTensor([epsilon]))

    def forward(self, input):
        s, u = torch.linalg.eigh(input)
        s = s.clamp(min=self.epsilon[0])
        s = s.diag_embed()
        output = u @ s @ u.transpose(-2, -1)

        return output


class Normalize(nn.Module):
    def __init__(self, p=2, dim=-1):
        super(Normalize, self).__init__()
        self.p = p
        self.dim = dim

    def forward(self, input):
        norm = input.norm(self.p, self.dim, keepdim=True)
        output = input / norm
        return output

class SPDNormalization(nn.Module):
    def __init__(self, input_size):
        super(SPDNormalization, self).__init__()
        self.G = SPDParameter(torch.eye(input_size), requires_grad=True)
        # temp = torch.randn(input_size, input_size)
        # self.G = SPDParameter(
        #     temp @ temp.T + torch.eye(input_size),
        #     requires_grad=True)

    def forward(self, input):
        center = mean(input, num_iter=1)
        center_sqrt = sqrtm(center)
        center_sqrt_inv = center_sqrt.inverse()
        center_norm = center_sqrt_inv @ input @ center_sqrt_inv

        # G_sqrt = sqrtm(self.G)
        # output = G_sqrt @ center_norm @ G_sqrt

        return center_norm

class SCaled_graph(nn.Module):
    def __init__(self):
        super(SCaled_graph, self).__init__()
        # self.dim = dim
        self.scale = nn.Parameter(torch.randn(1))
        # self.pool = nn.AvgPool2d(kernel_size=116, stride=116)
        self.pool = nn.MaxPool2d(kernel_size=116, stride=116)
        # self.vector = fc2vector()
    def forward(self, input):
        # print(input)
        graph_wavelet = create_normalized_matrices(input)
        weight = torch.exp(-self.scale * graph_wavelet)
        # print(f'scale = {self.scale}')
        # print(f'weight_scaled = {weight}')
        weight = graph_wavelet
        # print(f'graph_wavelet = {weight}')
        output = weight.transpose(-2, -1) @ input @ weight
        # batch_size = output.shape[0]
        if len(output.shape) == 2:
            
            output = output.unsqueeze(0)
        elif len(output.shape) == 3:
            
            pass
        else:
            
            raise RuntimeError("Unexpected output shape for pooling")
        # output_reshaped = output.view(batch_size, 1, 116*116, 116*116)
        pooled_output = self.pool(output)
        # print(f'pooled_output = {pooled_output}')
        # vector = fc2vector(pooled_output)
        # pooled_output=pooled_output.squeeze()
        # print(vector.shape)
        return pooled_output, self.scale
    
class SCaled_weighted_graph(nn.Module):
    def __init__(self):
        super(SCaled_weighted_graph, self).__init__()
        # self.dim = dim
        self.scale = nn.Parameter(torch.randn(1))
        self.attention_row = nn.Parameter(torch.randn(1,1,116*116))
        self.attention_column = nn.Parameter(torch.randn(1,116,1))
        self.pool = nn.MaxPool2d(kernel_size=116, stride=116)
        # self.pool= nn.AdaptiveAvgPool2d(output_size=(116, 116))

        # self.pool = nn.AvgPool2d(kernel_size=116, stride=116)

    def forward(self, input,vec):
        # graph_wavelet = create_normalized_matrices(input)
        # weight = torch.exp(-self.scale * graph_wavelet)
        # output = weight.transpose(-2, -1) @ input @ weight
        graph_wavelet = create_normalized_matrices(input)
        # graph_wavelet = graph_wavelet*10
        graph_wavelet  = torch.exp(-self.scale * graph_wavelet)
        # print(f'graph_wavelet = {graph_wavelet}')
        if vec ==0:
        # print(graph_wavelet)
        # transport_matrix = graph_wavelet.transpose(-2, -1)
            attention_rows = torch.nn.functional.softmax(self.attention_row, dim=2).squeeze(0) #nor
          
            weighted_matrix = graph_wavelet * self.attention_row
            # print(f'weighted_matrix = {weighted_matrix}')
            # print(weighted_matrix.shape)
            attention_columns =  torch.nn.functional.softmax(self.attention_column,dim=1).squeeze(0)
            # attention_columns =  torch.nn.functional.softmax(self.attention_column.view(116, -1), dim=0).view(116, 1)
            # print(attention_columns)
            # vector_expanded = attention_columns.unsqueeze(-1).expand(-1, -1, 116).contiguous().view(1, 116, -1)
            # print(vector_expanded)
            # weighted_matrix = weighted_matrix * vector_expanded.view(1, 1, -1)
          
            weighted_matrix = weighted_matrix * self.attention_column
            # print(f'weighted_matrix = {weighted_matrix}')
            # print(weighted_matrix.shape)
            # print(attention_rows)
            # print(f'input= {input}')

        
        #norm version
        if vec == 1:

            # attention_rows = (torch.nn.functional.softmax(self.attention_row, dim=2).view(1,1,-1))*100
            attention_rows = (torch.nn.functional.softmax(self.attention_row, dim=2).view(1,1,-1))

            weighted_matrix = graph_wavelet * attention_rows
            # attention_columns =  (torch.nn.functional.softmax(self.attention_column,dim=1))*100
            attention_columns =  (torch.nn.functional.softmax(self.attention_column,dim=1))

            weighted_matrix = weighted_matrix * attention_columns
            attention_rows = attention_rows.squeeze(0)
            attention_columns = attention_columns.squeeze(0)
        output =  weighted_matrix.transpose(-2, -1) @ input @ weighted_matrix
        # reshaped_rows = attention_rows.view(116, -1)
        # # print(reshaped_rows)
        # indices = torch.argmax(reshaped_rows, dim=0)
        # indices = indices.unsqueeze(1)
        # # print(indices.shape)
        # # print(indices)
        # max_indices = indices + torch.arange(0, attention_rows.size(0), 116).unsqueeze(1)
        # # print(max_indices)
        # selected_columns = weighted_matrix[:, :, max_indices.squeeze()]
        # print(selected_columns.shape)
        # print(f'output = {output}')
        pooled_output = self.pool(output)
        # output = output.squeeze(0)
        # eigenvalues = torch.linalg.eigvals(output)
        # min_real_eigenvalue = torch.min(eigenvalues.real)
        # print("Minimum Real Eigenvalue:", min_real_eigenvalue)

        # result = torch.allclose(output, output.transpose(-2, -1)) and (torch.eig(output)[0][:, 0] > 0).all()
        # print(result)

        # if not result:
        #     epsilon = 1e-4
        #     output = (output + output.transpose(-2, -1)) / 2
        #     output = output + epsilon * torch.eye(output.shape[-1])
        #     output = torch.linalg.cholesky(output)
        #     output = torch.matmul(output, output.transpose(-2, -1))

            # print(output)
        # result = torch.allclose(output, output.transpose(-2, -1)) and (torch.linalg.eigvals(output).real > 0).all()
        # print(result)

        if len(output.shape) == 2:
            
            output = output.unsqueeze(0)
        elif len(output.shape) == 3:
            
            pass
        else:
          
            raise RuntimeError("Unexpected output shape for pooling")
        # output_reshaped = output.view(batch_size, 1, 116*116, 116*116)
        
        # print(f'pooled_output = {pooled_output}')

        # vec = 0
        # print(pooled_output.shape)
        if vec==0:
            start_time = time.time()
            pooled_output = fc2vector(pooled_output)
            end_time = time.time()
            tangentspace_time = end_time - start_time
            print("tangentspace_time: {:.6f} seconds".format(tangentspace_time))
        # print(f'pooled_output = {pooled_output}')
        # vector = fc2vector(pooled_output)
        # print(pooled_output.shape)
        return pooled_output, self.scale, attention_rows, attention_columns