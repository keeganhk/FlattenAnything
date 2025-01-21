import os
import trimesh
import argparse
import pymeshlab
import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm
import point_cloud_utils as pcu
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .custo_knn_cpu import nearest_neighbors as custo_nearest_neighbors



def as_arr(torch_tensor):
    #### torch.Tensor -> numpy.ndarray
    return torch_tensor.detach().cpu().numpy()


def get_normalization_stats(points):
    # points: (N, 3)
    # norm_c: (3,), bounding-box center
    # norm_s: scalar, maximum scale after centralization
    assert points.ndim==2 and points.shape[-1]==3
    norm_c = (np.min(points, axis=0) + np.max(points, axis=0)) / 2.0
    points_moved = points - norm_c
    norm_s = np.linalg.norm(points_moved, ord=2, axis=-1).max()
    return norm_c, norm_s


def normalize_coordinates(points):
    # points: (N, 3)
    assert points.ndim==2 and points.shape[-1]==3
    norm_c, norm_s = get_normalization_stats(points)
    points_normalized = (points - norm_c) / (norm_s + 1e-8)
    return points_normalized


def normalize_normals(normal_vectors):
    # normal_vectors: (N, 3)
    # unit_normal_vectors: (N, 3)
    assert normal_vectors.ndim==2 and normal_vectors.shape[-1]==3
    scales = np.linalg.norm(normal_vectors, ord=2, axis=-1, keepdims=True) # (N, 1)
    unit_normal_vectors = normal_vectors / (scales + 1e-8)
    return unit_normal_vectors


def random_sampling(inputs, num_samples):
    # inputs: (num_points, num_channels)
    num_points, num_channels = inputs.shape
    assert num_samples <= num_points
    sampling_idx = np.random.choice(num_points, num_samples, replace=False) # (num_samples,)
    inputs_sampled = inputs[sampling_idx, :] # (num_samples, num_channels)
    return inputs_sampled


def get_knn_idx_custo(source, query, K):
    #### customized implementation of k-NN on CPU, pre-compiled for Python 3.9
    # source: [B, N, C]
    # query: [B, M, C]
    assert source.size(0) == query.size(0) and source.size(2) == query.size(2)
    source_cpu = source.detach().cpu()
    query_cpu = query.detach().cpu()
    knn_idx = custo_nearest_neighbors.knn_batch(source_cpu, query_cpu, K, omp=True) # (B, M, K), the last dim is sorted from near to far
    return knn_idx


def get_knn_idx_torch(source, query, K):
    # source: [B, N, C]
    # query: [B, M, C]
    assert (source.size(0) == query.size(0)) and (source.size(2) == query.size(2))
    pair_wise_distances = torch.cdist(query, source) # [B, M, N]
    knn_idx = torch.topk(pair_wise_distances, K, dim=-1, largest=False).indices # [B, M, K], the last dim is sorted from near to far
    return knn_idx


def get_fps_idx(inputs, num_samples):
    # inputs: [B, N, C]
    B, N, device = inputs.size(0), inputs.size(1), inputs.device
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    fps_idx = torch.zeros(B, num_samples, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    for si in range(num_samples):
        fps_idx[:, si] = farthest
        ctr = inputs[batch_indices, farthest, :].view(B, 1, -1)
        dist = torch.sum((inputs - ctr)**2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return fps_idx # [B, num_samples]


def index_points(source, idx):
    # source: [B, N, C]
    # 1) idx: [B, S] -> source_fetched: [B, S, C]
    # 2) idx: [B, S, K] -> source_fetched: [B, S, K, C]
    B, device = source.size(0), source.device
    view_shape = list(idx.shape) 
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B).to(device).view(view_shape).repeat(repeat_shape)
    source_fetched = source[batch_indices, idx, :]
    return source_fetched


def clean_mesh(load_mesh_path, save_mesh_path):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(load_mesh_path)
    ms.meshing_remove_null_faces()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_unreferenced_vertices()
    m = ms.current_mesh()
    V = m.vertex_matrix().astype(np.float32)
    F = m.face_matrix().astype(np.int64)
    Vn = m.vertex_normal_matrix().astype(np.float32)
    tm = trimesh.Trimesh(vertices=V, faces=F, vertex_normals=Vn)
    tm.export(save_mesh_path)


def load_mesh_with_normalization(load_mesh_path, normalize_vtx_pos=True, normalize_vtx_nor=True):
    mesh = trimesh.load(load_mesh_path, force="mesh")
    vtx_pos = np.array(mesh.vertices).astype(np.float32) # (num_verts, 3)
    vtx_nor = np.array(mesh.vertex_normals).astype(np.float32) # (num_verts, 3)
    tri_vid = np.array(mesh.faces).astype(np.int32) # (num_faces, 3)
    if normalize_vtx_pos:
        vtx_pos = normalize_coordinates(vtx_pos)
    if normalize_vtx_nor:
        vtx_nor = normalize_normals(vtx_nor)
    return vtx_pos, vtx_nor, tri_vid


def save_ply_point_cloud(save_point_cloud_path, points, colors=None, normals=None):
    #### color values should be within [0.0, 1.0]
    assert save_point_cloud_path.endswith(".ply")
    if isinstance(points, torch.Tensor):
        points = as_arr(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float32))
    if colors is not None:
        assert (colors.min() >= 0.0) and (colors.max() <= 1.0) 
        if isinstance(colors, torch.Tensor):
            colors = as_arr(colors)
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32)) 
    if normals is not None:
        if isinstance(normals, torch.Tensor):
            normals = as_arr(normals)
        pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float32))
    o3d.io.write_point_cloud(save_point_cloud_path, pcd, write_ascii=True)


def sample_points_from_mesh_approx(vertex_positions, face_indices, num_samples_approx, vertex_attributes=None):
    #### Note that the number of points obtained from Poisson disk sampling is not fixed, i.e., "num_samples_approx" != "num_samples"
    # vertex_positions: (num_vertices, 3)
    # face_indices: (num_faces, 3)
    # vertex_attributes: (num_vertices, attr_channels)
    assert vertex_positions.shape[0] ==vertex_attributes.shape[0]
    fi, bc = pcu.sample_mesh_poisson_disk(vertex_positions, face_indices, num_samples=num_samples_approx) 
    poisson_positions = pcu.interpolate_barycentric_coords(face_indices, fi, bc, vertex_positions) # (num_samples, 3)
    if vertex_attributes is None:
        return poisson_positions
    else:
        poisson_attributes = pcu.interpolate_barycentric_coords(face_indices, fi, bc, vertex_attributes) # (num_samples, 3)
        return poisson_positions, poisson_attributes


def build_2d_grids(H, W):
    import itertools
    h_p = np.linspace(-1, +1, H, dtype=np.float32)
    w_p = np.linspace(-1, +1, W, dtype=np.float32)
    grids_h_w_2 = np.array(list(itertools.product(h_p, w_p))).reshape(H, W, 2) # (H, W, 2)
    return grids_h_w_2


def batch_normalize_uv_points(uv_points):
    # uv_points: [B, N, 2]
    batch_min = uv_points.min(dim=1)[0] # [B, 2]
    batch_max = uv_points.max(dim=1)[0] # [B, 2]
    norm_c = ((batch_min + batch_max) / 2.0) # [B, 2]
    uv_points_centralized = uv_points - norm_c.unsqueeze(1) # [B, N, 2]
    norm_c = torch.norm(uv_points_centralized, p=2, dim=-1 ).max(dim=-1)[0] # [B]
    uv_points_normalized = uv_points_centralized / (norm_c.view(-1, 1, 1) + 1e-8) # [B, N, 2]
    return uv_points_normalized


def chamfer_distance(pts_1, pts_2, mode="max"):
    # pts_1: [B, N1, C]
    # pts_2: [B, N2, C]
    assert (pts_1.size(0) == pts_2.size(0)) and (pts_1.size(2) == pts_2.size(2))
    assert mode in ["max", "avg"]
    indices_of_1 = get_knn_idx_custo(pts_1, pts_2, 1)[:, :, 0] # (B, N2)
    indices_of_2 = get_knn_idx_custo(pts_2, pts_1, 1)[:, :, 0] # (B, N1)
    dists_2 = torch.sqrt(((index_points(pts_1, indices_of_1) - pts_2) ** 2).sum(dim=-1)) # [B, N2]
    dists_1 = torch.sqrt(((index_points(pts_2, indices_of_2) - pts_1) ** 2).sum(dim=-1)) # [B, N1]
    dists_2_mean = dists_2.mean(dim=-1, keepdim=True) # [B, 1]
    dists_1_mean = dists_1.mean(dim=-1, keepdim=True) # [B, 1]
    dists_concat = torch.cat((dists_2_mean, dists_1_mean), dim=-1) # [B, 2]
    if mode == "max": #### the "max" mode is highly recommended
        cd = dists_concat.max(dim=-1)[0].mean()
    if mode == "avg":
        cd = dists_concat.mean(dim=-1).mean()
    return cd


def repulsion_loss(points, K, min_euc_dist):
    # points: [B, N, C]
    knn_idx = get_knn_idx_custo(points, points, K+1)[:, :, 1:] # (B, N, K)
    knn_points = index_points(points, knn_idx) # [B, N, K, C]
    knn_distances = (((points.unsqueeze(2) - knn_points)**2).sum(dim=-1) + 1e-8).sqrt() # [B, N, K]
    loss = (F.relu(-knn_distances + min_euc_dist)).mean()
    return loss


def normal_cosine_similarity(normal_vectors_1, normal_vectors_2):
    # normal_vectors_1: [B, N, 3]
    # normal_vectors_2: [B, N, 3]
    loss = (1.0 - F.cosine_similarity(normal_vectors_1.view(-1, 3), normal_vectors_2.view(-1, 3))).mean()
    return loss


def compute_uv_grads(points_3D, points_2D):
    # points_3D: [B, N, 3]
    # points_2D: [B, N, 2]
    assert points_3D.size(0) == points_2D.size(0) and points_3D.size(1) == points_2D.size(1) 
    assert points_3D.size(2) == 3 and points_2D.size(2) == 2
    B, N, device = points_3D.size(0), points_3D.size(1), points_3D.device
    dx = torch.autograd.grad(points_3D[:, :, 0], points_2D, torch.ones_like(points_3D[:, :, 0]).float().to(device), create_graph=True)[0] # [B, N, 2]
    dy = torch.autograd.grad(points_3D[:, :, 1], points_2D, torch.ones_like(points_3D[:, :, 1]).float().to(device), create_graph=True)[0] # [B, N, 2]
    dz = torch.autograd.grad(points_3D[:, :, 2], points_2D, torch.ones_like(points_3D[:, :, 2]).float().to(device), create_graph=True)[0] # [B, N, 2]
    dxyz = torch.cat((dx.unsqueeze(2), dy.unsqueeze(2), dz.unsqueeze(2)), dim=2) # [B, N, 3, 2]
    grad_u = dxyz[:, :, :, 0] # [B, N, 3]
    grad_v = dxyz[:, :, :, 1] # [B, N, 3]
    return grad_u, grad_v


def get_diff_properties(points_3D, points_2D):
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
    return unit_normals, lambda_1, lambda_2 #### lambda_1 >= lambda_2


def fig2img(fig):
    fig.canvas.draw()
    img_arr = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(fig.canvas.get_width_height() + (4,)) # (height, width, 4), the last dim: Alpha & RGB
    img_pil = Image.fromarray(img_arr[:, :, 1:]) # drop the alpha channel
    return img_pil


def visualize_uv_points(uv_points, figure_size, marker_size, marker_color):
    # uv_points: [num_points, 2] or (num_points, 2)
    # marker_color: either be string (e.g., "r", "g", "b") or an array of (num_points, 3) with values within [0.0, 1.0]
    if isinstance(uv_points, torch.Tensor):
        uv_points = as_arr(uv_points) # (num_points, 2)
    u_coords = uv_points[:, 0]
    v_coords = uv_points[:, 1]
    fig, ax = plt.subplots(figsize=(figure_size, figure_size))
    plt.tight_layout()
    plt.axis('off')
    ax.scatter(u_coords, v_coords, s=marker_size, c=marker_color) 
    uv_points_pil = fig2img(fig)
    plt.close()
    return uv_points_pil


def one_row_export_image_list(image_list, title_list, each_figure_size, each_title_size, save_fig_path, dpi=150):
    num_images = len(image_list)
    assert len(title_list) == num_images
    fig, axes = plt.subplots(1, num_images, figsize=(each_figure_size*num_images, each_figure_size))
    plt.tight_layout()
    for ax, image, title in zip(axes, image_list, title_list):
        ax.axis("off")
        ax.imshow(image)
        ax.set_title(title, fontsize=each_title_size)
    plt.savefig(save_fig_path, dpi=dpi)
    plt.close()


def extract_edge_points(points, normalized_uv_points, K, threshold):
    # points: [B=1, N, 3]
    # normalized_uv_points: [B=1, N, 2]
    assert (points.size(0) == 1) and (normalized_uv_points.size(0) == 1)
    knn_idx = get_knn_idx_custo(points, points, K+1)[:, :, 1:] # (B=1, N, K)
    normalized_uv_points_mapped = index_points(normalized_uv_points, knn_idx) # [B=1, N, K, 2]
    dists = ((normalized_uv_points.unsqueeze(2) - normalized_uv_points_mapped) ** 2).sum(dim=-1).sqrt() # [B=1, N, K]
    dists_max = dists.max(dim=-1)[0] # [B=1, N]
    edge_mask = (dists_max.squeeze(0) > threshold) # [N]
    return edge_mask


def apply_checker_map(load_checker_map_path, map_size, normalized_uv_points):
    # normalized_uv_points: (num_uv_points, 2)
    checker_map_pil = Image.open(load_checker_map_path).resize((map_size, map_size))
    checker_map_arr = (np.array(checker_map_pil).astype(np.float32))[:, :, 0:3] / 255.0
    checker_map_arr = np.where(checker_map_arr<0.5, 0.0, 1.0) #### ensure pure black or white
    grids = build_2d_grids(map_size, map_size).reshape(-1, 2)
    nn_idx = get_knn_idx_custo(torch.tensor(grids).unsqueeze(0), torch.tensor(normalized_uv_points).unsqueeze(0), 1)[0, :, 0]
    mapped_colors = ((checker_map_arr.reshape(-1, 3))[nn_idx, :]).astype(np.float32)
    return mapped_colors
