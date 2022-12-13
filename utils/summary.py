import skimage
import torch, math, os, trimesh, plyfile, time

import utils.common
from utils import geometry, visualization
import numpy as np
from tqdm import tqdm
import matplotlib as mpl
from skimage.measure import marching_cubes
import traceback
import trimesh
from scipy.spatial import cKDTree as KDTree
from glob import glob
# mpl.use('Agg')
import matplotlib.pyplot as plt
from utils import fileio, visualization, common
from sketches.evaluate import icp, nearest_neighbor


def compute_chamfer_i3dmm(recon_pts, gt_pts, f_score_threshold=0.01, facetor_to_mm=100):
    # one direction
    gen_points_kd_tree = KDTree(recon_pts)
    one_distances, _ = gen_points_kd_tree.query(gt_pts)

    # other direction
    gt_points_kd_tree = KDTree(gt_pts)
    two_distances, _ = gt_points_kd_tree.query(recon_pts)

    completeness = one_distances
    accuracy = two_distances
    # max_side_length = np.max(bb_max - bb_min)
    # f_score_threshold = 0.01  # deep structured implicit functions sets tau = 0.01
    # L2 chamfer
    l2_chamfer = ((completeness).mean() + (accuracy).mean()) / 2
    # F-score
    f_completeness = np.mean(completeness <= f_score_threshold)
    f_accuracy = np.mean(accuracy <= f_score_threshold)
    f_score = facetor_to_mm * 2 * f_completeness * f_accuracy / (f_completeness + f_accuracy)  # harmonic mean
    return l2_chamfer * facetor_to_mm, f_score


def compute_recon_error(recon_path, gt_path, num_pts=150000, facetor_to_mm=100):
    recon_mesh = trimesh.load(recon_path)
    if isinstance(recon_mesh, trimesh.Scene):
        recon_mesh = recon_mesh.dump().sum()

    recon_pts = trimesh.sample.sample_surface(recon_mesh, num_pts)[0]

    gt_mesh = trimesh.load(gt_path)
    if isinstance(recon_mesh, trimesh.Scene):
        gt_mesh = gt_mesh.dump().sum()
    gt_pts = trimesh.sample.sample_surface(gt_mesh, num_pts)[0]

    distances, indices = nearest_neighbor(recon_pts, gt_pts)
    T, _, _ = icp(recon_pts, gt_pts[indices])
    recon_pts = common.transformation_4d(recon_pts, T)

    # cd, f_score, cmplt, acc = compute_chamfer_i3dmm(recon_pts, gt_pts)

    return compute_chamfer_i3dmm(recon_pts, gt_pts)


def make_contour_plot(array_2d, mode='log'):
    fig, ax = plt.subplots(figsize=(2.75, 2.75), dpi=300)

    if mode == 'log':
        num_levels = 6
        levels_pos = np.logspace(-2, 0, num=num_levels)  # logspace
        levels_neg = -1. * levels_pos[::-1]
        levels = np.concatenate((levels_neg, np.zeros((0)), levels_pos), axis=0)
        colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=num_levels * 2 + 1))
    else:  # mode == 'lin':
        num_levels = 10
        levels = np.linspace(-.5, .5, num=num_levels)
        colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=num_levels))

    sample = np.flipud(array_2d)
    CS = ax.contourf(sample, levels=levels, colors=colors)
    cbar = fig.colorbar(CS)

    ax.contour(sample, levels=levels, colors='k', linewidths=0.1)
    ax.contour(sample, levels=[0], colors='k', linewidths=0.3)
    ax.axis('off')
    return fig


def get_mesh_color(mesh_points, exp_embedding, id_embedding, model, config, int_idx):
    model.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    # mesh_points.requires_grad = False
    mesh_colors = np.zeros_like(mesh_points)
    num_samples = mesh_points.shape[0]
    max_batch = config.points_per_inference

    head = 0
    while head < num_samples:
        sample_subset = torch.from_numpy(mesh_points[head: min(head + max_batch, num_samples), 0:3]).float().cuda()[
            None, ...]

        mesh_colors[head: min(head + max_batch, num_samples), 0:3] = (
            model.get_template_coords(sample_subset, exp_embedding, id_embedding, int_idx)
                .squeeze()  # .squeeze(1)
                .detach()
                .cpu()
        )
        head += max_batch

    mesh_colors = np.clip(mesh_colors / 2 + 0.5, 0, 1)  # normalize color to 0-1

    return mesh_colors


def convert_sdf_samples_to_mesh(sdf_3d, mask, voxel_grid_origin=np.array([-1, -1, -1]), offset=None, scale=None,
                                level=0.0, return_value=False):
    """
    Convert sdf samples to .ply with color-coded template coordinates
    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    numpy_3d_sdf_tensor = np.array(sdf_3d)
    voxel_size = 2.0 / (numpy_3d_sdf_tensor.shape[0] - 1)

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    try:
        verts, faces, normals, values = marching_cubes(numpy_3d_sdf_tensor, level=level, spacing=[voxel_size] * 3,
                                                       mask=mask)
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


def test_decoder_on_trainset_2d(config, test_decoder, train_data, test_obj_list, device, writer,
                                batch_cnt):
    with torch.no_grad():
        test_decoder.eval()
        vox_resolution = int(config.voxel_resolution)
        slice_coords_2d = geometry.get_mgrid(vox_resolution)
        slice_coords_2d = torch.flip(slice_coords_2d, dims=[0])
        batch_split = math.ceil((vox_resolution ** 2) / config.points_per_inference)

        yz_slice_coords = torch.cat((torch.zeros_like(slice_coords_2d[:, :1]), slice_coords_2d), dim=-1)
        xz_slice_coords = torch.cat((slice_coords_2d[:, :1], torch.zeros_like(slice_coords_2d[:, :1]),
                                     slice_coords_2d[:, -1:]), dim=-1)
        xy_slice_coords = torch.cat((slice_coords_2d[:, :2], -0.75 * torch.ones_like(slice_coords_2d[:, :1])), dim=-1)
        slice_str = ['yz_slice', 'xz_slice', 'xy_slice']

        for idx in test_obj_list:
            data, sdf_file = train_data[idx]
            exp_idx = torch.tensor(data['exp'])
            id_idx = torch.tensor(data['id'])

            for k, slice in enumerate([yz_slice_coords, xz_slice_coords, xy_slice_coords]):
                grid_tensor = torch.chunk(slice.to(torch.float32), batch_split)
                sdf_batch = []
                for i in range(batch_split):
                    sdf_split, = test_decoder.inference(
                        grid_tensor[i].to(device),
                        exp_idx.to(device),
                        id_idx.to(device),
                        int_idx=True
                    )
                    sdf_batch.append(sdf_split.flatten().detach().cpu())
                sdf_val = torch.cat(sdf_batch, dim=0).numpy()
                sdf_grid = sdf_val.reshape((vox_resolution, vox_resolution))

                id_number = sdf_file.split('/')[-2]
                fig = make_contour_plot(sdf_grid)

                writer.add_figure('{}_{}'.format(id_number, slice_str[k]), fig, global_step=batch_cnt)

    del grid_tensor, exp_idx, id_idx, yz_slice_coords, xz_slice_coords, xy_slice_coords, sdf_split, sdf_batch, \
        sdf_val, sdf_grid, slice, slice_coords_2d


def convert_sdf_with_correspondence_color_to_ply(model, exp_embedding, id_embedding, int_idx, sdf_3d, ply_filename_out,
                                                 config, mask,
                                                 voxel_grid_origin=np.array([-1, -1, -1]), offset=None, scale=None,
                                                 level=0.0):
    mesh_points, faces = convert_sdf_samples_to_mesh(sdf_3d, mask, voxel_grid_origin, offset, scale, level)
    mesh_colors = get_mesh_color(mesh_points, exp_embedding, id_embedding, model, config, int_idx)
    mesh_colors = np.clip(mesh_colors * 255, 0, 255).astype(np.uint8)
    fileio.write_plyfile(mesh_points, faces, mesh_colors, ply_filename_out)


def convert_sdf_samples_with_color_to_ply(sdf_3d, ply_filename_out, mask, voxel_grid_origin=np.array([-1, -1, -1]),
                                          offset=None, scale=None, level=0.0):
    mesh_points, faces = convert_sdf_samples_to_mesh(sdf_3d, mask, voxel_grid_origin, offset, scale, level)
    mesh_colors = np.clip(mesh_points / 2 + 0.5, 0, 1)
    mesh_colors = np.clip(mesh_colors * 255, 0, 255).astype(np.uint8)
    fileio.write_plyfile(mesh_points, faces, mesh_colors, ply_filename_out)


def test_decoder_correspondence(config, test_decoder, test_data, sdf_vis_path, device):
    test_decoder.eval()
    vox_resolution = int(config.voxel_resolution)
    grid, vox_idx, mask = geometry.create_grid(vox_resolution, mask_size=1.)
    grid_tensor = torch.tensor(grid)
    for i in tqdm(range(len(test_data))):
        data_dict, sdf_file = test_data[i]
        exp_idx = torch.tensor(data_dict['exp'])
        id_idx = torch.tensor(data_dict['id'])
        exp_name = os.path.basename(sdf_file).split('.')[0].split('_')[0]
        save_path = os.path.join(sdf_vis_path, sdf_file.split('/')[-2])
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        sdf_val, = test_decoder.inference_by_batch(grid_tensor, exp_idx.to(device), id_idx.to(device), device,
                                                   points_per_inference=config.points_per_inference, int_idx=True)
        sdf_grid = sdf_val.reshape((vox_resolution, vox_resolution, vox_resolution)).transpose(1, 0, 2)
        plyfile_name = os.path.join(save_path, '{}_{}.ply'.format(exp_name, vox_resolution))
        mesh_points, faces = convert_sdf_samples_to_mesh(sdf_grid, mask)
        trimesh.Trimesh(mesh_points, faces).export(plyfile_name)
        # convert_sdf_with_correspondence_color_to_ply(test_decoder, exp_idx.to(device), None, True, sdf_grid,
        #                                              plyfile_name, config, mask)


def test_decoder_sparse_correspondence(config, test_decoder, test_data, sdf_vis_path, device):
    test_decoder.eval()
    vox_resolution = int(config.voxel_resolution)
    grid, vox_idx, mask = geometry.create_grid(vox_resolution, mask_size=1.)
    grid_tensor = torch.tensor(grid)
    ids_seen = np.zeros(1000)

    sdf_val, = test_decoder.inference_by_batch(grid_tensor, None, None, device=device,
                                               points_per_inference=config.points_per_inference, int_idx=True)
    sdf_grid = sdf_val.reshape((vox_resolution, vox_resolution, vox_resolution)).transpose(1, 0, 2)
    plyfile_name = os.path.join(sdf_vis_path, 'template_all.ply')
    convert_sdf_samples_with_color_to_ply(sdf_grid, plyfile_name, mask)

    for i in tqdm(range(len(test_data))):
        data_dict, sdf_file = test_data[i]
        exp_idx = torch.tensor(data_dict['exp'])
        id_idx = torch.tensor(data_dict['id'])
        exp_name = os.path.basename(sdf_file).split('.')[0].split('_')[0]
        save_path = os.path.join(sdf_vis_path, sdf_file.split('/')[-2])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        landmarks_ori, landmarks_deformed, landmarks_refered = test_decoder.get_landmarks(exp_idx.to(device),
                                                                                          id_idx.to(device))

        # Output template
        if not ids_seen[int(id_idx.item())]:
            obj_name = os.path.join(save_path, 'ldmk_refered.obj')
            trimesh.Trimesh(landmarks_refered.squeeze().detach().cpu().numpy()).export(obj_name)
            obj_name = os.path.join(save_path, 'ldmk_deformed.obj')
            trimesh.Trimesh(landmarks_deformed.squeeze().detach().cpu().numpy()).export(obj_name)
            ids_seen[int(id_idx.item())] = 1
            sdf_val, = test_decoder.inference_by_batch(grid_tensor, None, id_idx.to(device), device=device,
                                                       points_per_inference=config.points_per_inference, int_idx=True)
            sdf_grid = sdf_val.reshape((vox_resolution, vox_resolution, vox_resolution)).transpose(1, 0, 2)
            plyfile_name = os.path.join(save_path, 'template_{}.ply'.format(vox_resolution))
            convert_sdf_with_correspondence_color_to_ply(test_decoder, None, id_idx.to(device), True, sdf_grid,
                                                         plyfile_name, config, mask)
        sdf_val, = test_decoder.inference_by_batch(grid_tensor, exp_idx.to(device), id_idx.to(device), device,
                                                   points_per_inference=config.points_per_inference, int_idx=True)

        sdf_grid = sdf_val.reshape((vox_resolution, vox_resolution, vox_resolution)).transpose(1, 0, 2)
        plyfile_name = os.path.join(save_path, '{}.ply'.format(exp_name))
        # mesh_points, faces, _ = convert_sdf_samples_to_mesh(sdf_grid, mask)
        # trimesh.Trimesh(mesh_points, faces).export(plyfile_name)
        convert_sdf_with_correspondence_color_to_ply(test_decoder, exp_idx.to(device), id_idx.to(device), True,
                                                     sdf_grid, plyfile_name, config, mask)

        obj_name = os.path.join(save_path, '{}_ldmk_gt.obj'.format(exp_name))
        trimesh.Trimesh(data_dict['key_pts'].numpy()).export(obj_name)
        obj_name = os.path.join(save_path, '{}_ldmk_ori.obj'.format(exp_name))
        trimesh.Trimesh(landmarks_ori.squeeze().detach().cpu().numpy()).export(obj_name)

        # convert_sdf_with_correspondence_color_to_ply(test_decoder, exp_idx.to(device), None, True, sdf_grid,
        #                                              plyfile_name, config, mask)


def fit_decoder(config, test_decoder, test_data, exp_embeddings, id_embeddings, path, device, fit_id_num):
    test_decoder.eval()
    vox_resolution = int(config.voxel_resolution)
    grid, vox_idx, mask = geometry.create_grid(vox_resolution, mask_size=1.)
    grid_tensor = torch.tensor(grid)
    errors = {}
    mean_chamfer = 0.0
    mean_fscore = 0.0
    for i in tqdm(range(len(test_data))):
        data_dict, sdf_file = test_data[i]
        exp_idx = torch.tensor(data_dict['exp'])
        id_idx = torch.tensor(data_dict['id'])
        exp_name = os.path.basename(sdf_file).split('.')[0].split('_')[0]
        save_path = os.path.join(path, sdf_file.split('/')[-2])
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        idx = exp_idx.long() * fit_id_num + id_idx.long()
        exp_embedding = exp_embeddings(idx.to(device))
        id_embedding = id_embeddings(idx.to(device))

        sdf_val, = test_decoder.inference_by_batch(grid_tensor, exp_embedding, id_embedding,
                                                   config.points_per_inference, int_idx=False)
        sdf_grid = sdf_val.reshape((vox_resolution, vox_resolution, vox_resolution)).transpose(1, 0, 2)
        plyfile_name = os.path.join(save_path, '{}.ply'.format(exp_name))
        convert_sdf_with_correspondence_color_to_ply(test_decoder, exp_embedding, id_embedding, False,
                                                     sdf_grid, plyfile_name, config, mask)

        chamfer, fscore = compute_recon_error(plyfile_name, os.path.join(config.gt_path, sdf_file.split('/')[-2],
                                                                         os.path.basename(sdf_file).split('_')[
                                                                             0] + '.obj'))
        mean_chamfer += chamfer
        mean_fscore += fscore
        errors[sdf_file] = (chamfer, fscore)
    return mean_chamfer / len(test_data), mean_fscore / len(test_data), errors

def fit_one_decoder(config, test_decoder, test_data, exp_embeddings, id_embeddings, path, device, fit_id_num):
    test_decoder.eval()
    vox_resolution = int(config.voxel_resolution)
    grid, vox_idx, mask = geometry.create_grid(vox_resolution, mask_size=1.)
    grid_tensor = torch.tensor(grid)
    errors = {}
    mean_chamfer = 0.0
    mean_fscore = 0.0
    for i in tqdm(range(len(test_data))):
        data_dict, sdf_file = test_data[i]
        save_path = path
        idx = torch.tensor(0)
        exp_embedding = exp_embeddings(idx.to(device))
        id_embedding = id_embeddings(idx.to(device))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        sdf_val, = test_decoder.inference_by_batch(grid_tensor, exp_embedding, id_embedding,
                                                   config.points_per_inference, int_idx=False)
        sdf_grid = sdf_val.reshape((vox_resolution, vox_resolution, vox_resolution)).transpose(1, 0, 2)
        plyfile_name = os.path.join(save_path, 'demo_fit.ply')
        convert_sdf_with_correspondence_color_to_ply(test_decoder, exp_embedding, id_embedding, False,
                                                     sdf_grid, plyfile_name, config, mask)

        chamfer, fscore = compute_recon_error(plyfile_name, os.path.join(config.gt_path, 'demo.obj'))
        mean_chamfer += chamfer
        mean_fscore += fscore
        errors[sdf_file] = (chamfer, fscore)
        print(i)
    return mean_chamfer / len(test_data), mean_fscore / len(test_data), errors

def fit_decoder_dirichlet(config, test_decoder, test_data, exp_embeddings, id_embeddings, path, device, fit_id_num):
    from model import diff_opts
    test_decoder.eval()
    vox_resolution = int(config.voxel_resolution)
    grid, vox_idx, mask = geometry.create_grid(vox_resolution, mask_size=1.)
    grid_tensor = torch.tensor(grid)

    for i in tqdm(range(len(test_data))):
        data_dict, sdf_file = test_data[i]
        exp_idx = torch.tensor(data_dict['exp'])
        id_idx = torch.tensor(data_dict['id'])
        save_path = os.path.join(path, sdf_file.split('/')[-2])
        exp_name = os.path.basename(sdf_file).split('.')[0].split('_')[0]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        idx = exp_idx.long() * fit_id_num + id_idx.long()
        exp_embedding = exp_embeddings(idx.to(device).long())
        id_embedding = id_embeddings(idx.to(device).long())

        sdf_val, = test_decoder.inference_by_batch(grid_tensor, exp_embedding, id_embedding,
                                                   config.points_per_inference, int_idx=False)
        sdf_grid = sdf_val.reshape((vox_resolution, vox_resolution, vox_resolution)).transpose(1, 0, 2)
        plyfile_name = os.path.join(save_path, '{}_energy.ply'.format(exp_name))

        points, faces = convert_sdf_samples_to_mesh(sdf_grid, mask)

        # inference(self, coords_ori, exp_embedding, id_embedding, int_idx=True, return_fix=False, return_coords=False)
        points_input = torch.FloatTensor(points).to(device)
        points_input.requires_grad_(True)
        _, coords_dict = test_decoder.inference(points_input, exp_embedding, id_embedding, False, False, True)
        # deformed_coord = coords_dict['deformed_coords']
        refered_coords = coords_dict['refered_coords']
        energies = diff_opts.symmetric_dirichlet_energy(refered_coords, points_input)
        energies = energies.squeeze(0).detach().cpu().numpy()

        energies_vis = np.clip(energies, 4, 1e3)
        energies_vis = np.log10(energies_vis)  # 0.6~3
        energies_vis = (energies_vis - np.log10(3)) / (3 - np.log10(3))
        cmap = plt.get_cmap('PuRd', 256)

        mesh_colors = cmap(energies_vis / 2.)[..., :3] * 255

        fileio.write_plyfile(points, faces, mesh_colors.astype(np.uint8), plyfile_name)


def map2color(color_img, pcl):
    points = pcl[:, :2]  # (-1, 1)
    color_h, color_w, _ = color_img.shape
    points[:, 0] = (color_w - 1) * (points[:, 0] + 1.) / 2.
    points[:, 1] = (color_h - 1) * (points[:, 1] + 1.) / 2.
    points = np.floor(points).astype(int)
    return color_img[points[:, 1], points[:, 0]]


def get_mesh(test_decoder, exp_embedding, id_embedding, voxel_resolution=256, points_per_inference=163840):
    vox_resolution = int(voxel_resolution)
    grid, vox_idx, mask = geometry.create_grid(vox_resolution, mask_size=1.)
    grid_tensor = torch.tensor(grid)
    sdf_val, = test_decoder.inference_by_batch(grid_tensor, exp_embedding, id_embedding,
                                               points_per_inference, int_idx=False)
    sdf_grid = sdf_val.reshape((vox_resolution, vox_resolution, vox_resolution)).transpose(1, 0, 2)
    points_src, faces_src = convert_sdf_samples_to_mesh(sdf_grid, mask)
    return points_src, faces_src


def correspond(test_decoder, exp_emb_src, id_emb_src, exp_emb_dst, id_emb_dst, src_vertices=None):
    if src_vertices is None:
        src_vertices, _ = get_mesh(test_decoder, exp_emb_src, id_emb_src)
    points_input = torch.FloatTensor(src_vertices).to(exp_emb_src.device)
    points_input.requires_grad_(True)
    _, coords_dict = test_decoder.inference(points_input, exp_emb_src, id_emb_src, False, False, True)
    refered_coords_src = coords_dict['refered_coords'].squeeze(0).detach().cpu().numpy()
    src_tree = KDTree(refered_coords_src)

    dst_vertices, dst_faces = get_mesh(test_decoder, exp_emb_dst, id_emb_dst)
    points_dst = torch.FloatTensor(dst_vertices).to(exp_emb_dst.device)
    points_dst.requires_grad_(True)
    _, coords_dict = test_decoder.inference(points_dst, exp_emb_dst, id_emb_dst, False, False, True)
    refered_coords_dst = coords_dict['refered_coords'].squeeze(0).detach().cpu().numpy()
    _, idxs = src_tree.query(refered_coords_dst)
    return idxs, src_vertices, dst_vertices, dst_faces


def texture_transfer(config, test_decoder, dst_datas, src_data, exp_embeddings, id_embeddings, path, device,
                     fit_id_num):
    from utils import common
    common.cond_mkdir(path)
    test_decoder.eval()
    data_dict_src, sdf_file_src = src_data
    exp_name = os.path.basename(sdf_file_src).split('.')[0].split('_')[0]
    id_name = sdf_file_src.split('/')[-2]

    src_file = fileio.OBJ(os.path.join(config.texture_path, id_name, exp_name, exp_name + '.obj'))
    src_jpg = glob(os.path.join(config.texture_path, id_name, exp_name, '*.jpg'))[0]
    src_texture_coords = src_file.texcoords
    points_src = src_file.vertices

    exp_idx = torch.tensor(data_dict_src['exp']).long()
    id_idx = torch.tensor(data_dict_src['id']).long()
    idx = exp_idx * fit_id_num + id_idx
    exp_embedding_src = exp_embeddings(idx.to(device))
    id_embedding_src = id_embeddings(idx.to(device))

    for i in tqdm(range(len(dst_datas))):
        data_dict_dst, sdf_file_dst = dst_datas[i]
        exp_idx = torch.tensor(data_dict_dst['exp']).long()
        id_idx = torch.tensor(data_dict_dst['id']).long()
        idx = exp_idx * fit_id_num + id_idx
        save_path = os.path.join(path, sdf_file_dst.split('/')[-2])
        exp_name_dst = os.path.basename(sdf_file_dst).split('.')[0].split('_')[0]
        id_name_dst = sdf_file_src.split('/')[-2]
        common.cond_mkdir(save_path)
        exp_embedding = exp_embeddings(idx.to(device))
        id_embedding = id_embeddings(idx.to(device))

        idxs, _, dst_vertices, dst_faces = correspond(test_decoder, exp_embedding_src, id_embedding_src, exp_embedding,
                                                      id_embedding, points_src)

        fileio.write_mtl(dst_vertices, dst_faces, src_texture_coords[idxs], exp_name_dst + '.obj',
                         os.path.join(path, id_name_dst, exp_name_dst), src_jpg)


def do_correspondent(config, test_decoder, dst_datas, src_data, exp_embeddings, id_embeddings, path, device,
                     fit_id_num, is_fit=True, deform_type='refered_coords'):
    from scipy.spatial import cKDTree as KDTree
    import cv2
    from utils import common
    from copy import deepcopy
    common.cond_mkdir(path)

    test_decoder.eval()

    data_dict_src, sdf_file_src = src_data
    exp_idx = torch.tensor(data_dict_src['exp']).long()
    id_idx = torch.tensor(data_dict_src['id']).long()
    exp_name = os.path.basename(sdf_file_src).split('.')[0].split('_')[0]
    idx = exp_idx * fit_id_num + id_idx
    exp_embedding = exp_embeddings(idx.to(device))
    if is_fit:
        id_embedding = id_embeddings(idx.to(device))
    else:
        id_embedding = id_embeddings(id_idx.to(device))
    points_src, faces_src = get_mesh(test_decoder, exp_embedding, id_embedding)
    points_input = torch.FloatTensor(points_src).to(device)
    points_input.requires_grad_(True)
    _, coords_dict = test_decoder.inference(points_input, exp_embedding, id_embedding, False, False, True)
    refered_coords_src = coords_dict[deform_type].squeeze(0).detach().cpu().numpy()
    src_tree = KDTree(refered_coords_src)

    src_name = '{}_{}_corr_src'.format(sdf_file_src.split('/')[-2], exp_name)
    fileio.write_mtl(points_src, faces_src, (points_src[:, :2] + 1.) / 2., src_name + '.obj',
                     os.path.join(path, src_name))

    for i in tqdm(range(len(dst_datas))):
        data_dict_dst, sdf_file_dst = dst_datas[i]
        exp_idx = torch.tensor(data_dict_dst['exp']).long()
        id_idx = torch.tensor(data_dict_dst['id']).long()
        idx = exp_idx * fit_id_num + id_idx
        save_path = os.path.join(path, sdf_file_dst.split('/')[-2])
        exp_name = os.path.basename(sdf_file_dst).split('.')[0].split('_')[0]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        exp_embedding = exp_embeddings(idx.to(device))
        if is_fit:
            id_embedding = id_embeddings(idx.to(device))
        else:
            id_embedding = id_embeddings(id_idx.to(device))
        points_dst, faces_dst = get_mesh(test_decoder, exp_embedding, id_embedding)
        points_input = torch.FloatTensor(points_dst).to(device)
        points_input.requires_grad_(True)
        _, coords_dict = test_decoder.inference(points_input, exp_embedding, id_embedding, False, False, True)
        refered_coords_dst = coords_dict[deform_type].squeeze(0).detach().cpu().numpy()
        distances, idxs = src_tree.query(refered_coords_dst)

        dst_name = '{}_corr'.format(exp_name)
        fileio.write_mtl(points_dst, faces_dst, (points_src[idxs][:, :2] + 1.) / 2., dst_name + '.obj',
                         os.path.join(save_path, dst_name))


def get_coordinates(config, test_decoder, data, exp_embeddings, id_embeddings, device, fit_id_num):
    data_dict_src, sdf_file_src = data
    exp_idx = torch.tensor(data_dict_src['exp']).long()
    id_idx = torch.tensor(data_dict_src['id']).long()
    exp_name = os.path.basename(sdf_file_src).split('.')[0].split('_')[0]
    idx = exp_idx * fit_id_num + id_idx
    exp_embedding = exp_embeddings(idx.to(device))
    id_embedding = id_embeddings(idx.to(device))

    points_src, faces_src = get_mesh(test_decoder, exp_embedding, id_embedding)
    points_input = torch.FloatTensor(points_src).to(device)
    points_input.requires_grad_(True)
    _, coords_dict = test_decoder.inference(points_input, exp_embedding, id_embedding, False, False, True)


def mixing(test_decoder, id2idx, exp2idx, exp_embeddings, id_embeddings, path, device, fit_id_num):
    utils.common.cond_mkdir(path)
    id_list = [122, 212, 340, 393, 395, 421, 610, 344, 527, 594]
    exp_list = [3, 5, 16, 20, 12, 17, 13, 15, 18, 14]

    for i in tqdm(range(len(id_list))):
        for j in range(len(exp_list)):
            if i == j:
                continue
            exp_idx = torch.tensor(exp2idx[exp_list[j]]).long()
            id_idx = torch.tensor(id2idx[id_list[i]]).long()
            idx = exp_idx * fit_id_num + id_idx
            exp_embedding = exp_embeddings(idx.to(device))
            id_embedding = id_embeddings(idx.to(device))
            points_src, faces_src = get_mesh(test_decoder, exp_embedding, id_embedding)
            trimesh.Trimesh(points_src, faces_src).export(
                os.path.join(path, '{}id_mix_{}exp.obj'.format(id_list[i], exp_list[j])))


def manipulate(test_decoder, id2idx, exp2idx, exp_embeddings, id_embeddings, path, device, fit_id_num):
    utils.common.cond_mkdir(path)
    # id_list = [122, 212, 340, 393, 395, 421, 610, 344, 527, 594]
    id_list = [122, 212, 340, 395, 344, 527, 594]
    # id_list = [594]
    exp_list = [3, 5, 16, 20, 12, 17, 13, 15, 18, 14]

    exp_random = np.load(
        '/home/zhengmingwu_2020/ImFace-LIF/result/2021-10-19_18-59/PCA/embeddings/random/exp_random_3_embedding.npy')
    for i in tqdm(range(len(id_list))):
        exp = torch.FloatTensor(exp_random).to(device)
        exp_idx = torch.tensor(exp2idx[1]).long()
        id_idx = torch.tensor(id2idx[id_list[i]]).long()
        idx = exp_idx * fit_id_num + id_idx
        id_embedding = id_embeddings(idx.to(device))
        points_src, faces_src = get_mesh(test_decoder, exp, id_embedding)
        trimesh.Trimesh(points_src, faces_src).export(
            os.path.join(path, '{}id_r3_exp.obj'.format(id_list[i])))
        # trimesh.Trimesh(points_src, faces_src).export(
        #     os.path.join(path, '{}id_none_exp.obj'.format(id_list[i])))
    return
    exp_mean = np.load('/home/zhengmingwu_2020/ImFace-LIF/result/2021-10-19_18-59/PCA/exp_mean_embedding.npy')
    exp_mean = torch.FloatTensor(exp_mean).to(device)
    PCA_path = '/home/zhengmingwu_2020/ImFace-LIF/result/2021-10-19_18-59/PCA/embeddings'
    exp_embeddings_files = glob(os.path.join(PCA_path, 'exp_*_*_*.npy'))
    # print(len(exp2idx))
    for i in tqdm(range(len(id_list))):
        for exp_file in exp_embeddings_files:
            exp = torch.FloatTensor(np.load(exp_file)).to(device)
            exp_idx = torch.tensor(exp2idx[1]).long()
            id_idx = torch.tensor(id2idx[id_list[i]]).long()
            idx = exp_idx * fit_id_num + id_idx
            id_embedding = id_embeddings(idx.to(device))
            points_src, faces_src = get_mesh(test_decoder, exp, id_embedding)
            exp_file = os.path.basename(exp_file)
            trimesh.Trimesh(points_src, faces_src).export(
                os.path.join(path, '{}id_{}_exp.obj'.format(id_list[i], exp_file[4:9])))
            # trimesh.Trimesh(points_src, faces_src).export(
            #     os.path.join(path, '{}id_none_exp.obj'.format(id_list[i])))


def principal_analyze(config, test_decoder, train_datas, path, device, n_components=4, shift_times=10):
    from sklearn.decomposition import PCA
    save_path = os.path.join(path, 'PCA')
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'embeddings'), exist_ok=True)
    test_decoder.eval()
    vox_resolution = int(config.voxel_resolution)
    grid, vox_idx, mask = geometry.create_grid(vox_resolution, mask_size=1.)
    grid_tensor = torch.tensor(grid)

    # >>>>>>>>>> Calculate PCA Components >>>>>>>>>>
    id_embeddings = test_decoder.id_embedding.weight.data.detach().cpu().numpy()
    id_mean = np.mean(id_embeddings, axis=0)
    id_embeddings -= id_mean
    exp_embeddings = []

    for i in tqdm(range(len(train_datas))):
        data_dict, sdf_file = train_datas[i]
        exp_idx = torch.tensor(data_dict['exp'])
        id_idx = torch.tensor(data_dict['id'])
        exp_idx = exp_idx * test_decoder.id_embedding.num_embeddings + id_idx
        exp_embedding_batch = test_decoder.exp_embedding(exp_idx.to(device).long())
        exp_embeddings.append(exp_embedding_batch[None, ...])
    exp_embeddings = torch.cat(exp_embeddings, dim=0).detach().cpu().numpy()
    exp_mean = np.mean(exp_embeddings, axis=0)

    np.save(os.path.join(save_path, 'embeddings', 'exp_mean_embedding.npy'), exp_mean)
    np.save(os.path.join(save_path, 'embeddings', 'id_mean_embedding.npy'), id_mean)
    print('Mean embedding saved at \'{}\'.'.format(save_path))

    exp_embeddings -= exp_mean
    pca = PCA(n_components=n_components)
    pca.fit(exp_embeddings.T)
    reduced_exp = pca.transform(exp_embeddings.T).T
    reduced_exp_v = pca.explained_variance_
    pca = PCA(n_components=n_components)
    pca.fit(id_embeddings.T)
    reduced_id = pca.transform(id_embeddings.T).T
    reduced_id_v = pca.explained_variance_

    # >>>>>>>>>> Export Mean Face >>>>>>>>>>
    sdf_val, = test_decoder.inference_by_batch(grid_tensor, torch.FloatTensor(exp_mean).to(device),
                                               torch.FloatTensor(id_mean).to(device),
                                               config.points_per_inference, int_idx=False)
    sdf_grid = sdf_val.reshape((vox_resolution, vox_resolution, vox_resolution)).transpose(1, 0, 2)
    points, faces = convert_sdf_samples_to_mesh(sdf_grid, mask)
    trimesh.Trimesh(points, faces).export(os.path.join(save_path, 'mean.obj'))

    # >>>>>>>>>> Sampling Expression Components >>>>>>>>>>
    numbers = 5
    os.makedirs(os.path.join(save_path, 'embeddings', 'random'), exist_ok=True)
    for i in tqdm(range(numbers)):
        exp_shift = exp_mean
        for direction in tqdm(range(n_components)):
            exp_shift = exp_shift + np.random.normal(0, 1) * reduced_exp[direction] * (reduced_exp_v[direction] ** 0.5)
        np.save(os.path.join(save_path, 'embeddings', 'random', 'exp_random_{}_embedding.npy'.format(i)), exp_shift)
        sdf_val, = test_decoder.inference_by_batch(grid_tensor, torch.FloatTensor(exp_shift).to(device),
                                                   torch.FloatTensor(id_mean).to(device),
                                                   config.points_per_inference, int_idx=False)
        sdf_grid = sdf_val.reshape((vox_resolution, vox_resolution, vox_resolution)).transpose(1, 0, 2)
        points, faces = convert_sdf_samples_to_mesh(sdf_grid, mask)
        trimesh.Trimesh(points, faces).export(
            os.path.join(save_path, 'embeddings', 'random', 'exp_random_{}.obj'.format(i)))


    # >>>>>>>>>> Export Expression Components >>>>>>>>>>
    for direction in tqdm(range(n_components)):
        exp_shift = exp_mean + 3 * (reduced_exp_v[direction] ** 0.5) * reduced_exp[direction]

        np.save(os.path.join(save_path, 'embeddings', 'exp_pos_{}_embedding.npy'.format(direction)), exp_shift)

        sdf_val, = test_decoder.inference_by_batch(grid_tensor, torch.FloatTensor(exp_shift).to(device),
                                                   torch.FloatTensor(id_mean).to(device),
                                                   config.points_per_inference, int_idx=False)
        sdf_grid = sdf_val.reshape((vox_resolution, vox_resolution, vox_resolution)).transpose(1, 0, 2)
        points, faces = convert_sdf_samples_to_mesh(sdf_grid, mask)
        trimesh.Trimesh(points, faces).export(os.path.join(save_path, 'exp_pos_shift_{}.obj'.format(direction)))

        exp_shift = exp_mean - 3 * (reduced_exp_v[direction] ** 0.5) * reduced_exp[direction]

        np.save(os.path.join(save_path, 'embeddings', 'exp_neg_{}_embedding.npy'.format(direction)), exp_shift)

        sdf_val, = test_decoder.inference_by_batch(grid_tensor, torch.FloatTensor(exp_shift).to(device),
                                                   torch.FloatTensor(id_mean).to(device),
                                                   config.points_per_inference, int_idx=False)
        sdf_grid = sdf_val.reshape((vox_resolution, vox_resolution, vox_resolution)).transpose(1, 0, 2)
        points, faces = convert_sdf_samples_to_mesh(sdf_grid, mask)
        trimesh.Trimesh(points, faces).export(os.path.join(save_path, 'exp_neg_shift_{}.obj'.format(direction)))

    # >>>>>>>>>> Export Identity Components >>>>>>>>>>
    for direction in tqdm(range(n_components)):
        id_shift = id_mean + 3 * shift_times * (reduced_id_v[direction] ** 0.5) * reduced_id[direction]
        sdf_val, = test_decoder.inference_by_batch(grid_tensor, torch.FloatTensor(exp_mean).to(device),
                                                   torch.FloatTensor(id_shift).to(device),
                                                   config.points_per_inference, int_idx=False)
        sdf_grid = sdf_val.reshape((vox_resolution, vox_resolution, vox_resolution)).transpose(1, 0, 2)
        points, faces = convert_sdf_samples_to_mesh(sdf_grid, mask)
        trimesh.Trimesh(points, faces).export(os.path.join(save_path, 'id_pos_shift_{}.obj'.format(direction)))

        id_shift = id_mean - 3 * shift_times * (reduced_id_v[direction] ** 0.5) * reduced_id[direction]
        sdf_val, = test_decoder.inference_by_batch(grid_tensor, torch.FloatTensor(exp_mean).to(device),
                                                   torch.FloatTensor(id_shift).to(device),
                                                   config.points_per_inference, int_idx=False)
        sdf_grid = sdf_val.reshape((vox_resolution, vox_resolution, vox_resolution)).transpose(1, 0, 2)
        points, faces = convert_sdf_samples_to_mesh(sdf_grid, mask)
        trimesh.Trimesh(points, faces).export(os.path.join(save_path, 'id_neg_shift_{}.obj'.format(direction)))


def energy_field(config, test_decoder, test_data, exp_embeddings, id_embeddings, path, device, fit_id_num,
                 is_fit=False):
    from model import diff_opts
    test_decoder.eval()
    vox_resolution = 64
    grid, vox_idx, mask = geometry.create_grid(vox_resolution, mask_size=1.)
    grid_tensor = torch.FloatTensor(grid).to(device)
    grid_tensor.requires_grad_(True)
    for i in tqdm(range(len(test_data))):
        data_dict, sdf_file = test_data[i]
        exp_idx = torch.tensor(data_dict['exp']).long()
        id_idx = torch.tensor(data_dict['id']).long()
        save_path = os.path.join(path, sdf_file.split('/')[-2])
        exp_name = os.path.basename(sdf_file).split('.')[0].split('_')[0]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        idx = exp_idx * fit_id_num + id_idx
        exp_embedding = exp_embeddings(idx.to(device))
        if is_fit:
            id_embedding = id_embeddings(idx.to(device))
        else:
            id_embedding = id_embeddings(id_idx.to(device))

        sdf_val, coords_dict = test_decoder.inference(grid_tensor, exp_embedding, id_embedding, int_idx=False,
                                                      return_coords=True)
        deformed_coords = coords_dict['deformed_coords']
        refered_coords = coords_dict['refered_coords']
        energies_deform = diff_opts.symmetric_dirichlet_energy(deformed_coords, grid_tensor)
        energies_deform = energies_deform.squeeze(0).detach().cpu().numpy()
        energies_deform = energies_deform.reshape((vox_resolution, vox_resolution, vox_resolution)).transpose(1, 0, 2)

        energies_ref = diff_opts.symmetric_dirichlet_energy(refered_coords, deformed_coords)
        energies_ref = energies_ref.squeeze(0).detach().cpu().numpy()
        energies_ref = energies_ref.reshape((vox_resolution, vox_resolution, vox_resolution)).transpose(1, 0, 2)

        np.save(os.path.join(save_path, '{}_energy_ref.npy'.format(exp_name)), energies_ref)
        np.save(os.path.join(save_path, '{}_energy_def.npy'.format(exp_name)), energies_deform)
        np.save(os.path.join(save_path, '{}_coord_ref.npy'.format(exp_name)), refered_coords.detach().cpu().numpy())
        np.save(os.path.join(save_path, '{}_coord_def.npy'.format(exp_name)), deformed_coords.detach().cpu().numpy())


def weight_field(test_decoder, path, device):
    if not os.path.exists(path):
        os.makedirs(path)
    test_decoder.eval()
    vox_resolution = 64
    grid, vox_idx, mask = geometry.create_grid(vox_resolution, mask_size=1.)
    grid_tensor = torch.FloatTensor(grid).to(device)
    grid_tensor.requires_grad_(True)
    for i, name in enumerate(tqdm(['deform', 'reference', 'template'])):
        weights = test_decoder.hook_weight(grid_tensor[None, ...], i)  # N^3x5
        weights = weights.squeeze(0).detach().cpu().numpy()
        weights = weights.reshape((vox_resolution, vox_resolution, vox_resolution, weights.shape[-1])).transpose(1, 0,
                                                                                                                 2, 3)
        np.save(os.path.join(path, '{}_weights.npy'.format(name)), weights)
