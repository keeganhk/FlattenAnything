from .func import *
from .model import FlattenAnythingModel



def train_fam(points, normals, num_iter, export_folder):
    # points: (num_points, 3)
    # normals: (num_points, 3) or None
    # num_iter -> number of training iterations
    num_points = points.shape[0]
    P = torch.tensor(points).unsqueeze(0).float().cuda()
    if normals is not None:
        P_gtn = torch.tensor(normals).unsqueeze(0).float().cuda()
    normals_cc = as_arr(((P_gtn.squeeze(0) + 1.0) / 2.0)) if normals is not None else "b"
    grid_h = int(np.sqrt(num_points))
    grid_w = int(np.sqrt(num_points))
    num_grids = int(grid_h * grid_w)
    G = torch.tensor(build_2d_grids(grid_h, grid_w)).view(num_grids, 2).unsqueeze(0).cuda() 
    net = FlattenAnythingModel().cuda()
    max_lr, min_lr = 1e-3, 1e-5
    optimizer = optim.AdamW(net.parameters(), lr=max_lr, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iter, eta_min=min_lr)
    num_printing = 10
    print_counter = 0
    for iter_index in tqdm(range(1, num_iter+1)):
        net.zero_grad()
        P_opened, Q, P_cycle, P_cycle_n, Q_hat, P_hat, P_hat_n, P_hat_opened, Q_hat_cycle = net(G, P)
        Q_norm = batch_normalize_uv_points(Q)
        Q_hat_norm = batch_normalize_uv_points(Q_hat)
        Q_hat_cycle_norm = batch_normalize_uv_points(Q_hat_cycle)
        
        L_chamf = chamfer_distance(P_hat, P)
        
        repul_threshold = (2.0 / (np.ceil(np.sqrt(num_grids)) - 1.0)) * 0.25
        L_repul_Q = repulsion_loss(Q_norm, 8, repul_threshold)
        L_repul_Q_hat = repulsion_loss(Q_hat_norm, 8, repul_threshold)
        L_repul_Q_hat_cycle = repulsion_loss(Q_hat_cycle_norm, 8, repul_threshold)
        L_repul = L_repul_Q + L_repul_Q_hat + L_repul_Q_hat_cycle
        
        L_cycle_points = F.l1_loss(P, P_cycle) + F.l1_loss(Q_hat, Q_hat_cycle)
        if normals is not None:
            L_cycle_normals = normal_cosine_similarity(P_gtn, P_cycle_n)
            L_cycle = L_cycle_points + L_cycle_normals
        else:
            L_cycle = L_cycle_points
        
        _, e1, e2 = get_diff_properties(P_cycle, Q)
        """
        -- naive conformal constraint: L_disto = (e1 - e2).abs().mean()
        -- the common MIPS constraint: L_disto = (e1 / (e2 + 1e-8) + e2 / (e1 + 1e-8)).mean()
        -- naive isometric constraint: L_disto = (e1 - 1.0).abs().mean() + (e2 - 1.0).abs().mean()
        """
        L_disto = (e1 - e2).abs().mean()
        
        W_chamf = 1.0
        W_repul = 0.01
        W_cycle = 0.01
        W_disto = 0.01
        loss = L_chamf*W_chamf + L_repul*W_repul + L_cycle*W_cycle + L_disto*W_disto
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        if np.mod(iter_index, num_iter//num_printing) == 0:
            print_counter += 1
            print(f"iteration: {iter_index:6d}, L_chamf: {L_chamf.item():.8f}, L_repul: {L_repul.item():.8f}, L_cycle: {L_cycle.item():.8f}, L_disto: {L_disto.item():.8f}")
            Q_hat_norm_pil = visualize_uv_points(Q_hat_norm.squeeze(0), 5.0, 1.0, "r")
            Q_hat_cycle_norm_pil = visualize_uv_points(Q_hat_cycle_norm.squeeze(0), 5.0, 1.0, "g")
            Q_norm_pil = visualize_uv_points(Q_norm.squeeze(0), 5.0, 1.0, normals_cc)
            uv_image_list = [Q_hat_norm_pil, Q_hat_cycle_norm_pil, Q_norm_pil]
            uv_title_list = [f"Q_hat (iter: {iter_index})", f"Q_hat_cycle (iter: {iter_index})", f"Q (iter_index: {iter_index})"]
            one_row_export_image_list(uv_image_list, uv_title_list, 4.0, 8.0, os.path.join(export_folder, f"UV_{(print_counter):03d}.png"))
    torch.save(net.state_dict(), os.path.join(export_folder, "fam.pth"))


def test_fam(points, normals, load_ckpt_path, load_check_map_path, export_folder):
    # points: (num_points, 3)
    # normals: (num_points, 3) or None
    num_points = points.shape[0]
    P = torch.tensor(points).unsqueeze(0).float().cuda()
    if normals is not None:
        P_gtn = torch.tensor(normals).unsqueeze(0).float().cuda()
    normals_cc = as_arr(((P_gtn.squeeze(0) + 1.0) / 2.0)) if normals is not None else "b"
    grid_h = int(np.sqrt(num_points))
    grid_w = int(np.sqrt(num_points))
    num_grids = int(grid_h * grid_w)
    G = torch.tensor(build_2d_grids(grid_h, grid_w)).view(num_grids, 2).unsqueeze(0).cuda() 
    net = FlattenAnythingModel().cuda()
    net.load_state_dict(torch.load(load_ckpt_path))
    with torch.no_grad():
        P_opened, Q, P_cycle, P_cycle_n, Q_hat, P_hat, P_hat_n, P_hat_opened, Q_hat_cycle = net(G, P)
    Q_norm = batch_normalize_uv_points(Q)
    Q_hat_norm = batch_normalize_uv_points(Q_hat)
    Q_hat_cycle_norm = batch_normalize_uv_points(Q_hat_cycle)
    P_cycle_n = F.normalize(P_cycle_n, dim=-1)
    P_hat_n = F.normalize(P_hat_n, dim=-1)
    
    Q_hat_norm_pil = visualize_uv_points(Q_hat_norm.squeeze(0), 5.0, 1.0, "r")
    Q_hat_cycle_norm_pil = visualize_uv_points(Q_hat_cycle_norm.squeeze(0), 5.0, 1.0, "g")
    Q_norm_pil = visualize_uv_points(Q_norm.squeeze(0), 5.0, 1.0, normals_cc)
    uv_image_list = [Q_hat_norm_pil, Q_hat_cycle_norm_pil, Q_norm_pil]
    
    edge_mask = extract_edge_points(P, Q_norm, 3, 0.02) # [num_points]
    if edge_mask.sum().item() == 0:
        print("No Edge Found.")
        P_edge = None
    else:
        P_edge = P[:, edge_mask, :] # [1, num_edge_points, 3]
    
    P_checker_colors = apply_checker_map(load_check_map_path, 512, as_arr(Q_norm.squeeze(0))) # (num_points, 3)
    Q_norm = as_arr(Q_norm.squeeze(0))
    P_edge = as_arr(P_edge.squeeze(0))
    
    return uv_image_list, Q_norm, P_edge, P_checker_colors
