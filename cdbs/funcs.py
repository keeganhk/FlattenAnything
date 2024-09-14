import os, sys
sys.path.append(os.path.abspath('..'))
from cdbs.pkgs import *
from cdbs.basic import *



def bounding_box_normalization(pc):
    # pc: (num_points, num_channels)
    # pc_normalized: (num_points, num_channels)
    num_points, num_channels = pc.shape
    xyz = pc[:, 0:3]
    attr = pc[:, 3:]
    xyz = xyz - (np.min(xyz, axis=0) + np.max(xyz, axis=0))/2
    max_d = np.max(np.sqrt(np.abs(np.sum(xyz**2, axis=1)))) # a scalar
    xyz_normalized = xyz / max_d
    pc_normalized = np.concatenate((xyz_normalized, attr), axis=1)
    return pc_normalized


def farthest_point_sampling(pc, num_sample):
    # pc: (num_points, num_channels)
    # pc_sampled: (num_sample, num_channels)
    num_points, num_channels = pc.shape
    assert num_sample < num_points
    xyz = pc[:, 0:3] # sampling is based on spatial distance
    centroids = np.zeros((num_sample,))
    distance = np.ones((num_points,)) * 1e10
    farthest = np.random.randint(0, num_points)
    for i in range(num_sample):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    pc_sampled = pc[centroids.astype(np.int32)]
    return pc_sampled


def index_points(pc, idx):
    # pc: [B, N, C]
    # 1) idx: [B, S] -> pc_selected: [B, S, C]
    # 2) idx: [B, S, K] -> pc_selected: [B, S, K, C]
    device = pc.device
    B = pc.shape[0]
    view_shape = list(idx.shape) 
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B).to(device).view(view_shape).repeat(repeat_shape)
    pc_selected = pc[batch_indices, idx, :]
    return pc_selected


def get_fps_idx(xyz, num_sample):
    # xyz: torch.Tensor, [batch_size, num_input, 3]
    # fps_idx: [batch_size, num_sample]
    # assert xyz.ndim==3 and xyz.size(2)==3
    batch_size, num_input, device = xyz.size(0), xyz.size(1), xyz.device
    batch_indices = torch.arange(batch_size, dtype=torch.long).to(device)
    fps_idx = torch.zeros(batch_size, num_sample, dtype=torch.long).to(device)
    distance = torch.ones(batch_size, num_input).to(device) * 1e10
    farthest = torch.randint(0, num_input, (batch_size,), dtype=torch.long).to(device)
    for i in range(num_sample):
        fps_idx[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batch_size, 1, -1)
        dist = torch.sum((xyz-centroid)**2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return fps_idx


def get_fps_idx_zero_as_first(xyz, num_sample):
    # xyz: torch.Tensor, [batch_size, num_input, 3]
    # fps_idx: [batch_size, num_sample]
    # assert xyz.ndim==3 and xyz.size(2)==3
    batch_size, num_input, device = xyz.size(0), xyz.size(1), xyz.device
    batch_indices = torch.arange(batch_size, dtype=torch.long).to(device)
    fps_idx = torch.zeros(batch_size, num_sample, dtype=torch.long).to(device)
    distance = torch.ones(batch_size, num_input).to(device) * 1e10
    farthest = torch.zeros(batch_size, dtype=torch.long).to(device) # torch.randint(0, num_input, (batch_size,), dtype=torch.long).to(device)
    for i in range(num_sample):
        fps_idx[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batch_size, 1, -1)
        dist = torch.sum((xyz-centroid)**2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return fps_idx


def get_fps_idx_specified_first(xyz, num_sample, first):
    # xyz: torch.Tensor, [batch_size, num_input, 3]
    # fps_idx: [batch_size, num_sample]
    # first: [batch_size]
    # assert xyz.ndim==3 and xyz.size(2)==3
    batch_size, num_input, device = xyz.size(0), xyz.size(1), xyz.device
    batch_indices = torch.arange(batch_size, dtype=torch.long).to(device)
    fps_idx = torch.zeros(batch_size, num_sample, dtype=torch.long).to(device)
    distance = torch.ones(batch_size, num_input).to(device) * 1e10
    farthest = first.long().to(device) # [batch_size]
    for i in range(num_sample):
        fps_idx[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batch_size, 1, -1)
        dist = torch.sum((xyz-centroid)**2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return fps_idx


def align_number(raw_number, expected_num_digits):
    # align a number string
    string_number = str(raw_number)
    ori_num_digits = len(string_number)
    assert ori_num_digits <= expected_num_digits
    return (expected_num_digits - ori_num_digits) * '0' + string_number


def load_mesh_model_vfn(mesh_load_path):
    mesh_v, mesh_f = pcu.load_mesh_vf(mesh_load_path)
    mesh_vn = rescale_normals(pcu.estimate_mesh_vertex_normals(mesh_v, mesh_f), scale=1.0)
    mesh_v, mesh_vn, mesh_f = mesh_v.astype(np.float32), mesh_vn.astype(np.float32), mesh_f.astype(np.int64)
    return mesh_v, mesh_vn, mesh_f


def save_pc_as_ply(save_path, points, colors=None, normals=None):
    assert save_path[-3:] == 'ply', 'not .ply file'
    if type(points) == torch.Tensor:
        points = np.asarray(points.detach().cpu())
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        if type(colors) == torch.Tensor:
            colors = np.asarray(colors.detach().cpu())
        assert colors.min()>=0 and colors.max()<=1
        pcd.colors = o3d.utility.Vector3dVector(colors) # should be within the range of [0, 1]
    if normals is not None:
        if type(normals) == torch.Tensor:
            normals = np.asarray(normals.detach().cpu())
        pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud(save_path, pcd, write_ascii=True) # should be saved as .ply file


def min_max_normalization(x):
    x_min = x.min()
    x_max = x.max()
    x_mmn = (x - x_min) / (x_max - x_min)
    return x_mmn


def ts2np(x):
    y = np.asarray(x.detach().cpu())
    return y


def build_colormap(num_colors):
    # cmap: (num_colors, 3), within the range of [0, 1]
    queries = np.linspace(0.1, 0.9, num_colors, dtype=np.float32)
    cmap = matplotlib.cm.jet(queries)[:, 0:3] # (num_colors, 3)
    return cmap


def rescale_normals(normals, scale=1.0):
    # normals: (num_pts, 3)
    rescaled_normals = normals / (np.linalg.norm(normals, ord=2, axis=-1, keepdims=True) + 1e-8) * scale
    return rescaled_normals



####################################################################################################################
def build_2d_grids(H, W):
    h_p = np.linspace(-1, +1, H, dtype=np.float32) # np.linspace(-1.0, +1.0, H, dtype=np.float32)
    w_p = np.linspace(-1, +1, W, dtype=np.float32) # np.linspace(-1.0, +1.0, W, dtype=np.float32)
    grid_points = np.array(list(itertools.product(h_p, w_p))).reshape(H, W, 2) # (H, W, 2)
    return grid_points


def uv_bounding_box_normalization(uv_points):
    # uv_points: [B, N, 2]
    centroids = ((uv_points.min(dim=1)[0] + uv_points.max(dim=1)[0]) / 2).unsqueeze(1) # [B, 1, 2]
    uv_points = uv_points - centroids
    max_d = (uv_points**2).sum(dim=-1).sqrt().max(dim=-1)[0].view(-1, 1, 1) # [B, 1, 1]
    uv_points = uv_points / max_d
    return uv_points


def load_texture_map(img_path, map_res, binarize=False):
    num_pix = map_res ** 2
    texture_map = Image.open(img_path)
    texture_map = np.asarray(texture_map.resize((map_res, map_res)))[:, :, 0:3] / 255.0
    texture_uv = build_2d_grids(map_res, map_res).reshape(-1, 2) # (num_pix, 2)
    texture_rgb = texture_map.reshape(-1, 3) # (num_pix, 3)
    if binarize:
        texture_rgb = np.where(texture_rgb < 0.5, 0, 1)
    return texture_uv, texture_rgb
    

def compute_uv_grads(points_3D, points_2D):
    # points_3D: [B, N, 3]
    # points_2D: [B, N, 2]
    assert points_3D.size(1)==points_2D.size(1) and points_3D.size(2)==3 and points_2D.size(2)==2
    B, N, device = points_3D.size(0), points_3D.size(1), points_3D.device
    dx = torch.autograd.grad(points_3D[:, :, 0], points_2D, torch.ones_like(points_3D[:, :, 0]).float().to(device), create_graph=True)[0] # [B, N, 2]
    dy = torch.autograd.grad(points_3D[:, :, 1], points_2D, torch.ones_like(points_3D[:, :, 1]).float().to(device), create_graph=True)[0] # [B, N, 2]
    dz = torch.autograd.grad(points_3D[:, :, 2], points_2D, torch.ones_like(points_3D[:, :, 2]).float().to(device), create_graph=True)[0] # [B, N, 2]
    dxyz = torch.cat((dx.unsqueeze(2), dy.unsqueeze(2), dz.unsqueeze(2)), dim=2) # [B, N, 3, 2]
    grad_u = dxyz[:, :, :, 0] # [B, N, 3]
    grad_v = dxyz[:, :, :, 1] # [B, N, 3]
    return grad_u, grad_v


def compute_differential_properties(points_3D, points_2D):
    # points_3D: [B, N, 3]
    # points_2D: [B, N, 2]
    grad_u, grad_v = compute_uv_grads(points_3D, points_2D) # grad_u & grad_v: [B, N, 3]
    unit_normals = F.normalize(torch.cross(grad_u, grad_v, dim=-1), dim=-1) # [B, N, 3]
    Jf = torch.cat((grad_u.unsqueeze(-1), grad_v.unsqueeze(-1)), dim=-1) # [B, N, 3, 2]
    I = Jf.permute(0, 1, 3, 2).contiguous() @ Jf # [B, N, 2, 2]
    E = I[:, :, 0, 0] # [B, N]
    G = I[:, :, 1, 1] # [B, N]
    FF = I[:, :, 0, 1] # [B, N]
    item_1 = E + G # [B, N]
    item_2 = torch.sqrt(4*(FF**2) + (E-G)**2) # [B, N]
    lambda_1 = 0.5 * (item_1 + item_2) # [B, N]
    lambda_2 = 0.5 * (item_1 - item_2) # [B, N]
    # Note that we always have: "lambda_1 >= lambda_2"
    return unit_normals, lambda_1, lambda_2


def compute_repulsion_loss(points, K, minimum_distance):
    # points: [B, N, C]
    # K: number of neighbors
    # minimum_distance: Euclidean distance threshold
    knn_idx = knn_search(points, points, K+1)[:, :, 1:] # [B, N, K]
    knn_points = index_points(points, knn_idx) # [B, N, K, C]
    knn_distances = (((points.unsqueeze(2) - knn_points)**2).sum(dim=-1) + 1e-8).sqrt() # [B, N, K]
    loss = (F.relu(-knn_distances + minimum_distance)).mean()
    return loss


def compute_normal_cos_sim_loss(x1, x2):
    # x1: [B, N, 3]
    # x2: [B, N, 3]
    loss = (1 - F.cosine_similarity(x1.view(-1, 3), x2.view(-1, 3))).mean()
    return loss


def extract_edge_points(points_3d, points_2d, K, T):
    # points_3d: [B, N, 3]
    # points_2d: [B, N, 2]
    assert points_3d.size(0)==1 and points_2d.size(0)==1
    points_2d = uv_bounding_box_normalization(points_2d)
    knn_idx = knn_search(points_3d, points_3d, K+1)[:, :, 1:] # [B, N, K]
    points_2d_mapped = index_points(points_2d, knn_idx) # [B, N, K, 2]
    dists = ((points_2d.unsqueeze(2) - points_2d_mapped)**2).sum(dim=-1).sqrt() # [B, N, K]
    dists_max = dists.max(dim=-1)[0] # [B, N]
    edge_mask = (dists_max.squeeze(0) > T)
    return edge_mask


