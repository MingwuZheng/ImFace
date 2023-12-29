from scipy import misc
import torch
import os
import trimesh
import math
from utils import geometry
import numpy as np
from skimage.measure import marching_cubes
import traceback
import trimesh
from scipy.spatial import cKDTree as KDTree

def sdf2mesh(sdf_3d, mask, voxel_grid_origin=np.array([-1, -1, -1]), offset=None, scale=None, level=0.0, return_value=False):
    """
    Convert sdf samples to .ply with color-coded template coordinates
    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    numpy_3d_sdf_tensor = np.array(sdf_3d)
    voxel_size = 2.0 / (numpy_3d_sdf_tensor.shape[0] - 1)

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    try:
        verts, faces, normals, values = marching_cubes(numpy_3d_sdf_tensor, level=level, spacing=[voxel_size] * 3, mask=mask)
    except:
        traceback.print_exc()
        pass

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    if return_value:
        return mesh_points, faces, values
    return mesh_points, faces

def extract_mesh(config, model, id_num, exp_num, save_path, device, chunk_size=8192):
    model.eval()
    torch.set_grad_enabled(True)
    vox_resolution = int(config.VOX_RESOLUTION)
    grid, _, mask = geometry.create_grid(vox_resolution, mask_size=1.)

    batch_split = math.ceil((vox_resolution**3) / chunk_size)
    grid_tensor = torch.chunk(torch.tensor(grid).to(torch.float32).to(device), batch_split)

    data = {}
    sdf_batch = []
    data['EXP_INDEX'] = torch.LongTensor([exp_num]).to(device)
    data['ID_INDEX'] = torch.LongTensor([id_num]).to(device)
    for j in range(batch_split):
        data['XYZ'] = grid_tensor[j].unsqueeze(0)
        data['XYZ'].requires_grad = True
        output = model(data)[-1]
        sdf_split = output['residual_sdf']
        sdf_batch.append(sdf_split.flatten().detach().cpu())
    sdf_val = torch.cat(sdf_batch, dim=0).numpy()
    sdf_grid = sdf_val.reshape((vox_resolution, ) * 3).transpose(1, 0, 2)
    v, f = sdf2mesh(sdf_grid, mask)
    trimesh.Trimesh(v, f).export(save_path)

def extract_mesh_and_compute_error(config, model, id_num, exp_num, save_path, device, chunk_size=8192):
    model.eval()
    torch.set_grad_enabled(True)
    vox_resolution = int(config.VOX_RESOLUTION)
    grid, _, mask = geometry.create_grid(vox_resolution, mask_size=1.)

    batch_split = math.ceil((vox_resolution**3) / chunk_size)
    grid_tensor = torch.chunk(torch.tensor(grid).to(torch.float32).to(device), batch_split)

    data = {}
    sdf_batch = []
    data['EXP_INDEX'] = torch.LongTensor([exp_num]).to(device)
    data['ID_INDEX'] = torch.LongTensor([id_num]).to(device)
    for j in range(batch_split):
        data['XYZ'] = grid_tensor[j].unsqueeze(0)
        data['XYZ'].requires_grad = True
        output = model(data)[-1]
        sdf_split = output['residual_sdf']
        sdf_batch.append(sdf_split.flatten().detach().cpu())
    sdf_val = torch.cat(sdf_batch, dim=0).numpy()
    sdf_grid = sdf_val.reshape((vox_resolution, ) * 3).transpose(1, 0, 2)
    v, f = sdf2mesh(sdf_grid, mask)
    trimesh.Trimesh(v, f).export(save_path)
    chamfer, fscore, _, _ = compute_recon_error(save_path, os.path.join(config.GT_PATH, 'demo.obj'))

    return chamfer, fscore

def register_mesh(src_mesh, dst_mesh, num_pts=150000, max_iterations=20, tolerance=0.001):
    A = trimesh.sample.sample_surface(src_mesh, num_pts)[0]
    B = trimesh.sample.sample_surface(dst_mesh, num_pts)[0]
    src = np.ones((4, A.shape[0]))
    dst = np.ones((4, B.shape[0]))
    src[:3, :] = np.copy(A.T)
    dst[:3, :] = np.copy(B.T)
    prev_error = 0
    for _ in range(max_iterations):
        distances, indices = geometry.nearest_neighbor(src[:3, :].T, dst[:3, :].T)
        T, _, _ = geometry.transform_fit(src[:3, :].T, dst[:3, indices].T)
        src = np.dot(T, src)
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    T, _, _ = geometry.transform_fit(A, src[:3, :].T)
    src_mesh.vertices = geometry.transformation_4d(src_mesh.vertices, T)
    return src_mesh

def compute_recon_error(recon_path, gt_path, num_pts=150000, facetor_to_mm=100):
    recon_mesh = trimesh.load(recon_path)
    if isinstance(recon_mesh, trimesh.Scene):
        recon_mesh = recon_mesh.dump().sum()
    gt_mesh = trimesh.load(gt_path)
    if isinstance(recon_mesh, trimesh.Scene):
        gt_mesh = gt_mesh.dump().sum()

    recon_mesh = register_mesh(recon_mesh, gt_mesh)
    recon_pts = trimesh.sample.sample_surface(recon_mesh, num_pts)[0]
    gt_pts = trimesh.sample.sample_surface(gt_mesh, num_pts)[0]
    return compute_chamfer(recon_pts, gt_pts, facetor_to_mm=facetor_to_mm)

def compute_chamfer(recon_pts, gt_pts, f_score_threshold=0.01, facetor_to_mm=100):
    # one direction
    gen_points_kd_tree = KDTree(recon_pts)
    completeness, _ = gen_points_kd_tree.query(gt_pts)

    # other direction
    gt_points_kd_tree = KDTree(gt_pts)
    accuracy, _ = gt_points_kd_tree.query(recon_pts)

    # L2 chamfer
    l2_chamfer = ((completeness).mean() + (accuracy).mean()) / 2
    # F-score
    f_completeness = np.mean(completeness <= f_score_threshold)
    f_accuracy = np.mean(accuracy <= f_score_threshold)
    f_score = facetor_to_mm * 2 * f_completeness * f_accuracy / (f_completeness + f_accuracy)  # harmonic mean
    return l2_chamfer * facetor_to_mm, f_score, (completeness).mean() * facetor_to_mm, (accuracy).mean() * facetor_to_mm

