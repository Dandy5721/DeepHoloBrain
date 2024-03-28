import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import torch

def create_normalized_matrices(batch_tensor):
    
    if len(batch_tensor.shape) == 2:
        
        batch_tensor = batch_tensor.unsqueeze(0)

    batch_size, node_num, _ = batch_tensor.shape
    D = batch_tensor.sum(dim=2)

   
    D_inv_sqrt = torch.where(D > 0, 1.0 / torch.sqrt(D), torch.zeros_like(D))
    L = -batch_tensor.clone()
    L += torch.eye(node_num, device=batch_tensor.device).expand_as(L) 
    D_inv_sqrt_L = D_inv_sqrt.unsqueeze(2) * L
    norm_L = D_inv_sqrt_L * D_inv_sqrt.unsqueeze(1)

    
    sym_L = (norm_L + norm_L.transpose(1, 2)) / 2

    
    U, S, _ = torch.linalg.svd(sym_L)
#     print(U.shape)
#     S = torch.flip(S, [0]) #descend to ascend order
    S_clamped = torch.clamp(S, min=0)  
#     print(S_clamped.shape)
    
    big_matrix_batch = torch.zeros(batch_size, 0, node_num * node_num, device=batch_tensor.device, dtype=batch_tensor.dtype)
#     for i in range(node_num):
#         zero_matrix = torch.zeros(batch_size, node_num, node_num, device=batch_tensor.device, dtype=batch_tensor.dtype)
#         zero_matrix[:, i, i] = S_clamped[:, i]
#         matrix = torch.matmul(torch.matmul(U, zero_matrix), U.transpose(-2, -1))
#         big_matrix_batch = torch.cat((big_matrix_batch, matrix.view(batch_size, 1, -1)), dim=1)
    diag = torch.diag_embed(S_clamped)
    diag = diag.unsqueeze(0).repeat(node_num,1,1,1)
    lhs = torch.einsum("bjk,nbkl,bli->nbji", U, diag, U.transpose(1, 2))
#     print(lhs.shape)
    big_matrix_batch=lhs.view(lhs.size(0), lhs.size(1), -1)
#     print(big_matrix_batch.shape)
    big_matrix_batch = big_matrix_batch.permute(1, 0, 2)

#     zero_matrices = torch.zeros(batch_size, node_num, node_num, device=batch_tensor.device, dtype=batch_tensor.dtype)
#     zero_matrices[:, torch.arange(node_num), torch.arange(node_num)] = S_clamped[:, :node_num].unsqueeze(-1).expand(-1, node_num, -1)

#     U_expanded = U.unsqueeze(1).unsqueeze(-1)
#     big_matrix_batch = torch.matmul(torch.matmul(U_expanded, zero_matrices.unsqueeze(-1)), U_expanded.transpose(-2, -1)).squeeze(-1)

#     big_matrix_batch=big_matrix_batch.squeeze()
#     print(big_matrix_batch.shape)
    return big_matrix_batch

