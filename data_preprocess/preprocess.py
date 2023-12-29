import json
import numpy as np
import os, sys
root_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(root_dir)

os.environ["PYOPENGL_PLATFORM"] = "egl"
import trimesh
from data_preprocess.cut_mesh import cut
from tqdm import tqdm
import pymeshlab
from numba import jit
from scipy.spatial import Delaunay
import data_preprocess.surface_point_cloud as surface_point_cloud
import traceback
import shutil
from multiprocessing import Pool

def crop(v, f):
     # read landmark file
    land_dir = os.path.join('data_preprocess/landmark_indices.npz')
    # the nose tip index
    landmark_index = np.load(land_dir)['v10']
    nose_index = landmark_index[30]

    mview_mesh = trimesh.Trimesh(
            vertices=v, faces=f, process=False, maintain_order=True)
    # nose tip's position
    point = mview_mesh.vertices[nose_index]

    mview_mesh.vertices -= point
    mview_mesh.vertices[:, 2] += 40
    # landmark
    landmark = mview_mesh.vertices[landmark_index]

    v, f = cut(mview_mesh.vertices, mview_mesh.faces, [0, 0, 0], 100)

    # normalization
    return v / 100, f, landmark / 100

def hidden_surface_remove(vertices, triangle_idx, direction=None, epsilon=1e-6):
    if direction is None:
        direction = [0., 0., -1.]
    direction = -np.array(direction)
    vertices, triangle_idx = np.array(vertices, dtype=float), np.array(triangle_idx, dtype=int)
    triangles = vertices[triangle_idx]
    plane_normals, triangle_ok = trimesh.triangles.normals(triangles)
    condition = triangle_ok == True
    triangle_idx = triangle_idx[condition]
    vn, fn = vertices.shape[0], triangle_idx.shape[0]
    triangles = vertices[triangle_idx]
    ray_directions = np.repeat(direction[np.newaxis, :], vn, axis=0)
    index_triangle, index_ray, locations = trimesh.ray.ray_triangle.ray_triangle_id(triangles, vertices, ray_directions)
    hidden_idx = np.where(np.linalg.norm(vertices[index_ray] - locations, axis=1) > epsilon)[0]
    hidden_idx = np.unique(index_ray[hidden_idx])

    mask = np.ones(vn)
    mask[hidden_idx] = 0
    return mask

@jit(nopython=True)
def remove_face_by_point_idx(face_array_matrix, indicator):
    face_array_matrix_crop = []
    lenth = face_array_matrix.shape[0]
    for i in range(lenth):
        if indicator[face_array_matrix[i][0]] and \
                indicator[face_array_matrix[i][1]] and \
                indicator[face_array_matrix[i][2]]:
            face_array_matrix_crop.append(face_array_matrix[i])
    return face_array_matrix_crop

def remove_vertices_from_mesh(vertices, triangle_idx, mask, color):
    """
    :param vertices: (n, 3) float
    :param triangle_idx: (m, 3) int
    :param mask: (n,) where 1 indicates preserve
    :return: vertices, triangle_idx after remove
    """
    indicator = mask
    ms = pymeshlab.MeshSet()
    face_array_matrix_crop = remove_face_by_point_idx(triangle_idx, indicator)
    face_array_matrix_crop = np.array(face_array_matrix_crop)

    tri_crop = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=face_array_matrix_crop)
    ms.add_mesh(tri_crop)
    ms.remove_unreferenced_vertices()
    result_mesh = ms.current_mesh()

    return result_mesh.vertex_matrix(), result_mesh.face_matrix(), result_mesh.vertex_color_matrix()

def delaunay_mesh_in_2d(vertices):
    vertices2d = vertices[:, :2]
    tri = Delaunay(vertices2d)
    return vertices, tri.simplices

def remove_and_watertight(v, f):
    cropped_face_v = v
    cropped_face_f = f

    mask = hidden_surface_remove(cropped_face_v, cropped_face_f)
    color = np.zeros((cropped_face_v.shape[0], 4))
    vs_new, fs_new, _ = remove_vertices_from_mesh(cropped_face_v, cropped_face_f, mask, color)
    vs_remesh, fs_remesh = delaunay_mesh_in_2d(vs_new)

    return vs_remesh, fs_remesh

def process_normal(v, f, des, landmark):
    try:
        mesh = trimesh.Trimesh(
            vertices=v, faces=f)
        surface_points, freespace_points = surface_point_cloud.sample_on_surface(
            mesh, number_of_points=250000)
        surf_pcl, surf_nor = surface_points
        free_pcl, free_sdf, free_grd = freespace_points

        udf = np.where(free_sdf < 0, -free_sdf, free_sdf)
        free_grd[free_sdf < 0] *= -1
        g = np.dot(free_grd, np.array([0, 0, 1]))
        udf[g < 0] *= -1
        free_sdf = udf
        free_grd[g < 0] *= -1
        g = np.dot(surf_nor, np.array([0, 0, 1]))
        surf_nor[g < 0] *= -1

        np.save(des.replace('.npy', '_surf_pcl.npy'), surf_pcl)
        np.save(des.replace('.npy', '_surf_nor.npy'), surf_nor)
        np.save(des.replace('.npy', '_free_pcl.npy'), free_pcl)
        np.save(des.replace('.npy', '_free_sdf.npy'), free_sdf)
        np.save(des.replace('.npy', '_free_grd.npy'), free_grd)
        with open(des.replace('.npy', '.bnd'), 'w') as f:
            for line in landmark:
                f.write('{} {} {}\n'.format(line[0], line[1], line[2]))
    except:
        traceback.print_exc()

"""
input:
    ply_name:   data path
    save_path:   save path (ends with '.npy')
"""
def data_preprocess(ply_path, save_path):
    mesh = trimesh.load(ply_path, process=False, maintain_order=True)
    v_init, f_init = mesh.vertices, mesh.faces
    
    # crop and normalize
    v, f, landmark = crop(v_init, f_init)
    # remove hidden face and watertight
    v, f = remove_and_watertight(v, f)
    mesh = trimesh.Trimesh(vertices=v, faces=f)
    mesh.export(save_path.replace('.npy', '.obj'))
    # get dataset
    process_normal(v, f, save_path, landmark)

if __name__ == "__main__":
    ids = []
    with open('config/list/facescape_all.txt') as f:
        for line in f:
            ids.append(line.strip())
    des_path = 'dataset/Facescape'
    data_path = 'dataset/FacescapeOriginData'
    if not os.path.exists(des_path):
        os.makedirs(des_path)
    for id in ids:
        des_file = os.path.join(des_path, id)
        if not os.path.exists(des_file):
            os.makedirs(des_file)
    obj_files = []
    des_files = []
    expression = ['neutral', 'smile', 'mouth_stretch', 'anger', 'jaw_left', 'jaw_right', 'jaw_forward', 'mouth_left', 'mouth_right', 'dimpler', 'chin_raiser',
              'lip_puckerer', 'lip_funneler', 'sadness', 'lip_roll', 'grin', 'cheek_blowing', 'eye_closed', 'brow_raiser', 'brow_lower']
    for id in ids:
        for obj_id in range(1, 21):
            obj_path = os.path.join(data_path, id, '{0}_{1}.obj'.format(obj_id, expression[obj_id - 1]))
            obj_files.append(obj_path)
            des_files.append(os.path.join(
                des_path, id, '{}.npy'.format(obj_id)))

    worker_count = 52
    print("Using {:d} processes.".format(worker_count))
    pool = Pool(worker_count)
    progress = tqdm(total=len(obj_files))

    def on_complete(*_):
        progress.update()

    for i in range(len(obj_files)):
        pool.apply_async(data_preprocess, args=(
            obj_files[i], des_files[i]), callback=on_complete)
    pool.close()
    pool.join()
