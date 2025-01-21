from util.func import *
from util.workflow import train_fam



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("load_mesh_path", type=str)
    parser.add_argument("export_root", type=str)
    parser.add_argument("N", type=int, help="number of points fed into FAM during each iteration")
    parser.add_argument("num_iter", type=int)
    args = parser.parse_args()
    
    mesh_name = args.load_mesh_path.split("/")[-1].split(".")[0]
    export_folder = os.path.join(args.export_root, mesh_name)
    os.makedirs(export_folder, exist_ok=True)
    
    vtx_pos, vtx_nor, tri_vid = load_mesh_with_normalization(args.load_mesh_path, True, True)
    # vtx_pos: (num_verts, 3)
    # vtx_nor: (num_verts, 3)
    # tri_vid: (num_faces, 3)
    num_verts = vtx_pos.shape[0]
    num_faces = tri_vid.shape[0]
    print(f"{num_verts=}, {num_faces=}")
    
    N_poisson_approx = int(args.N * 2.0)
    poisson_points, poisson_normals = sample_points_from_mesh_approx(vtx_pos, tri_vid, N_poisson_approx, vtx_nor) # (N_poisson, 3), N_poisson > N_poisson_approx
    poisson_normals = normalize_normals(poisson_normals) # After sampling, the normal values may slightly overflow [-1, +1].
    poisson_points = torch.tensor(poisson_points).unsqueeze(0).float().cuda() # [1, N_poisson, 3]
    poisson_normals = torch.tensor(poisson_normals).unsqueeze(0).float().cuda() # [1, N_poisson, 3]
    fps_idx = get_fps_idx(poisson_points, args.N) # [1, N]
    points = as_arr(index_points(poisson_points, fps_idx).squeeze(0)) # (N, 3)
    normals = as_arr(index_points(poisson_normals, fps_idx).squeeze(0)) # (N, 3)
    
    print(f"start training on [{mesh_name}] ...")
    train_fam(points, normals, args.num_iter, export_folder)

if __name__ == '__main__':
    main()
    print("training finished.")
