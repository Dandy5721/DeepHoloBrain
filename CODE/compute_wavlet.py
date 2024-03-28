import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time

def create_normalized_matrices(input_tensor):
    # Ensure the input is a PyTorch tensor
    # print(input_tensor.dtype)
    W = torch.tensor(input_tensor)
    node_num = W.size(1)
    D = W.sum(dim=1)

    # First normalization
    ind = D > 0
    L = -W.clone()
    L[ind, :] = L[ind, :] / torch.sqrt(D[ind]).unsqueeze(1)
    L[:, ind] = L[:, ind] / torch.sqrt(D[ind]).unsqueeze(0)
    L[torch.arange(node_num), torch.arange(node_num)] = 1

    # Second normalization
    symL = (L + L.transpose(0, 1)) / 2
    U, S, V = torch.svd(symL)
    S = torch.clamp(S, min=0)  # Ensure no negative eigenvalues
    inds = torch.argsort(S, descending=False)
    U = U[:, inds]
    signs = torch.sign(U[0, :])
    signs[signs == 0] = 1
    U = U * signs.unsqueeze(0)

    big_matrix = torch.zeros(0, node_num * node_num, device=W.device, dtype=W.dtype)

    for i in range(node_num):
        zero_matrix = torch.zeros(node_num, node_num, device=W.device, dtype=W.dtype)
        zero_matrix[i, i] = S[inds[i]]
        matrix = U @ zero_matrix @ U.transpose(0, 1)
        big_matrix = torch.cat((big_matrix, matrix.view(1, -1)), dim=0)

    return big_matrix
# Assuming W is your input matrix
mat_data = scipy.io.loadmat('**')
W = mat_data['SC_avg56']
node_num = W.shape[0]
D = np.sum(W, axis=1)

# The first normalization
ind = D > 0
L = -W
L[ind, :] = L[ind, :] / np.sqrt(D[ind, None])
L[:, ind] = L[:, ind] / np.sqrt(D[None, ind])
np.fill_diagonal(L, 1)
# print(L)
# The second normalization
symL = (L + L.T) / 2
U, S, _ = np.linalg.svd(symL)
# print(S)
E = np.sort(S)
# print(E)
inds = np.argsort(S)
# print(inds)
U = U[:, inds]
signs = np.sign(U[0, :])
signs[signs == 0] = 1
U = U @ np.diag(signs)
# print((np.diag(signs)).shape)
# print(U)
big_matrix = np.zeros((0, node_num * node_num))

for i in range(node_num):
    zero_matrix = np.zeros((node_num, node_num))
    zero_matrix[i, i] = E[i]
    # print(S[i])
    # print(U)
    matrix = U @ zero_matrix @ U.T
    print(matrix)
    print(np.max(matrix))
    print(np.average(matrix))
  
    big_matrix = np.vstack((big_matrix, matrix.flatten()))
    # print((matrix.flatten()).shape) 
    # print(big_matrix.shape)
    plt.imshow(matrix, cmap='PuRd')
    plt.colorbar()
    plt.savefig(f"matrix_{i+1}.png")  # Save each matrix as a PNG file
    time.sleep(10)
print(big_matrix.shape)
# print(big_matrix)    
    # print(zero_matrix)
    # save_path = '/ICML2024/'
    # file_name = f"matrix_{i+1}.mat"
    # data_to_save = {'matrix': big_matrix}
    # scipy.io.savemat(save_path + file_name, data_to_save)
    # time.sleep(1)
    # plt.imshow(matrix, cmap='PuRd')
    # plt.colorbar()
    # plt.savefig(f"matrix_{i+1}.png")  # Save each matrix as a PNG file

    # plt.show()
    # plt.pause(1)  # Uncomment if you want to pause between plot

# test = create_normalized_matrices(W)
# print(test.shape)
# print(test)


