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

