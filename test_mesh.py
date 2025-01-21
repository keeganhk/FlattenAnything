from util.func import *
from util.workflow import test_fam



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("load_mesh_path", type=str)
    parser.add_argument("load_ckpt_path", type=str)
    parser.add_argument("load_check_map_path", type=str)
    parser.add_argument("export_folder", type=str)
    parser.add_argument("input_format", type=str, choices=["mesh_verts", "sampled_points"])
    parser.add_argument("--N_poisson_approx", type=int, default=100000, help="number of points fed into FAM during each iteration")
    args = parser.parse_args()
    
    suffix = f"tested_on_{args.input_format}"
    save_uv_image_path = os.path.join(args.export_folder, f"UV_{suffix}.png")
    save_edge_points_path = os.path.join(args.export_folder, f"edge_points_{suffix}.ply")
    save_textured_points_path = os.path.join(args.export_folder, f"textured_points_{suffix}.ply")
    
    vtx_pos, vtx_nor, tri_vid = load_mesh_with_normalization(args.load_mesh_path, True, True)
    # vtx_pos: (num_verts, 3)
    # vtx_nor: (num_verts, 3)
    # tri_vid: (num_faces, 3)
    num_verts = vtx_pos.shape[0]
    num_faces = tri_vid.shape[0]
    print(f"{num_verts=}, {num_faces=}")
    
    if args.input_format == "mesh_verts":
        vtx_uv_image_list, vtx_uv, vtx_edge, vtx_checker_colors = test_fam(vtx_pos, vtx_nor, args.load_ckpt_path, args.load_check_map_path, args.export_folder)
        one_row_export_image_list(vtx_uv_image_list, ["Q_hat", "Q_hat_cycle", "Q"], 4.0, 8.0, save_uv_image_path)
        if vtx_edge is not None:
            save_ply_point_cloud(save_edge_points_path, vtx_edge)
        save_ply_point_cloud(save_textured_points_path, vtx_pos, vtx_checker_colors, vtx_nor)
        
    if args.input_format == "sampled_points":
        poisson_points, poisson_normals = sample_points_from_mesh_approx(vtx_pos, tri_vid, args.N_poisson_approx, vtx_nor)
        poisson_normals = normalize_normals(poisson_normals) # After sampling, the normal values may slightly overflow [-1, +1].
        N_poisson = poisson_points.shape[0]
        print(f"actual number of sampled points: {N_poisson}")
        pts_uv_image_list, pts_uv, pts_edge, pts_checker_colors = test_fam(poisson_points, poisson_normals, args.load_ckpt_path, args.load_check_map_path, args.export_folder)
        one_row_export_image_list(pts_uv_image_list, ["Q_hat", "Q_hat_cycle", "Q"], 4.0, 8.0, save_uv_image_path)
        if pts_edge is not None:
            save_ply_point_cloud(save_edge_points_path, pts_edge)
        save_ply_point_cloud(save_textured_points_path, poisson_points, pts_checker_colors, poisson_normals)


if __name__ == '__main__':
    main()
    print("testing finished.")
