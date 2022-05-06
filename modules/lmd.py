import torch

from torch import nn
import numpy as np


# SEED_RANDOM_DIRS = 2
#
#
# def get_random_dirs(components, dimensions):
#     gen = np.random.RandomState(seed=SEED_RANDOM_DIRS)
#     dirs = gen.normal(size=(components, dimensions))
#     dirs /= np.sqrt(np.sum(dirs ** 2, axis=1, keepdims=True))
#     return dirs.astype(np.float32)


def initial_directions(components, dimensions, device):
    M = torch.randn(dimensions, components).to(device)
    Q, R = adjust_sign(*QR_Decomposition(M))
    return Q


def QR_Decomposition(A):
    n = A.size(0)  # get the shape of A
    m = A.size(1)
    Q = torch.empty((n, n))  # initialize matrix Q
    u = torch.empty((n, n))  # initialize matrix u
    u[:, 0] = A[:, 0]
    Q[:, 0] = u[:, 0] / torch.linalg.norm(u[:, 0])
    for i in range(1, n):
        print(i)
        u[:, i] = A[:, i]
        for j in range(i):
            u[:, i] -= (A[:, i] @ Q[:, j]) * Q[:, j]  # get each u vector
        Q[:, i] = u[:, i] / torch.linalg.norm(u[:, i])  # compute each e vetor
    R = torch.zeros((n, m))
    for i in range(n):
        for j in range(i, m):
            R[i, j] = A[:, j] @ Q[:, i]
    return Q, R


def diag_sign(A):
    "Compute the signs of the diagonal of matrix A"
    D = torch.diag(torch.sign(torch.diag(A)))
    return D


def adjust_sign(Q, R):
    """
    Adjust the signs of the columns in Q and rows in R to
    impose positive diagonal of Q
    """
    D = diag_sign(Q)
    Q[:, :] = Q @ D
    R[:, :] = D @ R
    return Q, R


class LinearMotionDecomposition(nn.Module):
    motion_dictionary = []
    MOTION_DICTIONARY_SIZE = 20
    MOTION_DICTIONARY_DIMENSION = 512

    #self.register_parameter(name='bias', param=torch.nn.Parameter(torch.randn(3)))

    def __init__(self):
        super(LinearMotionDecomposition, self).__init__()
        self.motion_dictionary = torch.nn.Parameter(
            initial_directions(self.MOTION_DICTIONARY_DIMENSION, self.MOTION_DICTIONARY_SIZE, device='cpu'),
            requires_grad=True)

    @staticmethod
    def generate_latent_path(self, magnitudes):
        Z = torch.empty(1, 512)
        M = torch.transpose(self.motion_dictionary)
        for i in range(magnitudes.shape[0]):
            for j in range(M.shape[1]):
                total = 0
                for k in range(magnitudes.shape[1]):
                    total += magnitudes[i, k] * M[k, j]
                Z[i, j] = total
        return Z

    def generate_target_code(self, source_latent_code, magnitudes):
        return source_latent_code + self.generate_latent_path(magnitudes)

    def forward(self, x, magnitudes):
        self.motion_dictionary = adjust_sign(*QR_Decomposition(self.motion_dictionary))
        target_code = self.generate_target_code(x, magnitudes)
        return target_code
