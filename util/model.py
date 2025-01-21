import torch
import torch.nn as nn
import torch.nn.functional as F



class PWE(nn.Module):
    # point-wise embedding: Conv_with_bias (without BN) + LeakyReLU
    def __init__(self, C_in, C_out, neg_slope=None):
        super(PWE, self).__init__()
        self.neg_slope = neg_slope
        self.conv = nn.Conv1d(C_in, C_out, 1, bias=True)
    def forward(self, P_in):
        # P_in: [B, N, C_in]
        # B is the "batch size"
        # N is the "number of points"
        # C_in is the "feature dimension"
        P_in = P_in.permute(0, 2, 1).contiguous() # [B, C_in, N]
        P_out = self.conv(P_in) # [B, C_out, N]
        if self.neg_slope is not None:
            P_out = F.leaky_relu(P_out, self.neg_slope, True) # [B, C_out, N]
        P_out = P_out.permute(0, 2, 1).contiguous() # [B, N, C_out]
        return P_out # [B, N, C_out]


class Cutting(nn.Module):
    def __init__(self):
        super(Cutting, self).__init__()
        hidden_dim = 64
        self.mlp_1 = nn.Sequential(PWE(3, 512, 0.01), PWE(512, 512, 0.01), PWE(512, hidden_dim, None))
        self.mlp_2 = nn.Sequential(PWE(hidden_dim+3, 512, 0.01), PWE(512, 512, 0.01), PWE(512, 3, None))
    def forward(self, X):
        # X: [B, N, 3]
        Xo = self.mlp_2(torch.cat((X, self.mlp_1(X)), dim=-1)) # [B, N, 3]
        Xc = X + Xo
        return Xc


class Unwrapping(nn.Module):
    def __init__(self):
        super(Unwrapping, self).__init__()
        self.cut = Cutting()
        self.mlp = nn.Sequential(PWE(3, 512, 0.01), PWE(512, 512, 0.01), PWE(512, 2, None))
    def forward(self, points_3d):
        # points_3d: [B, N, 3]
        dfm = self.cut(points_3d)
        unwrapped_points_2d = self.mlp(dfm) # [B, N, 2]
        return dfm, unwrapped_points_2d


class GridDeforming(nn.Module):
    def __init__(self):
        super(GridDeforming, self).__init__()
        hidden_dim = 64
        self.mlp_1 = nn.Sequential(PWE(2, 512, 0.01), PWE(512, 512, 0.01), PWE(512, 512, 0.01), PWE(512, hidden_dim, None))
        self.mlp_2 = nn.Sequential(PWE(hidden_dim+2, 512, 0.01), PWE(512, 512, 0.01), PWE(512, 512, 0.01), PWE(512, 2, None))
    def forward(self, points_2d):
        # points_2d: [B, N, 2]
        offsets_2d = self.mlp_2(torch.cat((points_2d, self.mlp_1(points_2d)), dim=-1)) # [B, N, 2]
        deformed_points_2d = points_2d + offsets_2d # [B, N, 2]
        return deformed_points_2d


class Wrapping(nn.Module):
    def __init__(self):
        super(Wrapping, self).__init__()
        hidden_dim = 64
        self.mlp_1 = nn.Sequential(PWE(2, 512, 0.01), PWE(512, 512, 0.01), PWE(512, 512, 0.01), PWE(512, hidden_dim, None))
        self.mlp_2 = nn.Sequential(PWE(hidden_dim+2, 512, 0.01), PWE(512, 512, 0.01), PWE(512, 512, 0.01), PWE(512, 6, None))
    def forward(self, points_2d):
        # points_2d: [B, N, 2]
        wrapped_6d = self.mlp_2(torch.cat((points_2d, self.mlp_1(points_2d)), dim=-1)) # [B, N, 6]
        wrapped_points = wrapped_6d[:, :, 0:3] # [B, N, 3]
        wrapped_normals = wrapped_6d[:, :, 3:6] # [B, N, 3]
        return wrapped_points, wrapped_normals


class FlattenAnythingModel(nn.Module):
    def __init__(self):
        super(FlattenAnythingModel, self).__init__()
        self.unwrapping = Unwrapping()
        self.grid_deforming = GridDeforming()
        self.wrapping = Wrapping()
    def forward(self, G, P):
        # G: [B, M, 2]
        # P: [B, N, 3]
        #### [3D -> 2D -> 3D] cycle mapping
        P_opened, Q = self.unwrapping(P)
        P_cycle, P_cycle_n = self.wrapping(Q)
        #### [2D -> 3D -> 2D] cycle mapping
        Q_hat = self.grid_deforming(G)
        P_hat, P_hat_n = self.wrapping(Q_hat)
        P_hat_opened, Q_hat_cycle = self.unwrapping(P_hat)
        return P_opened, Q, P_cycle, P_cycle_n, Q_hat, P_hat, P_hat_n, P_hat_opened, Q_hat_cycle
