import torch, cv2
import trimesh, trimesh.ray
import numpy as np
from utils import fileio
from numba import jit
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors


def transformation_4d(pcl, matrix_4x4):
    return matrix_4x4.dot(np.vstack((pcl.T, np.ones_like(pcl.T)[0, :][None, ...]))).T[:, :3]

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


def crop_spherically(pcl, center, r):
    """
    :param pcl: n*3
    :param center: [x,y,z]
    :param r: range
    :return:
    """
    n = pcl.shape[0]
    center = np.repeat(center[np.newaxis, :], n, axis=0).reshape((n, 3))
    dis = pcl - center
    dis = (dis * dis).sum(axis=1)
    return pcl[np.where(dis < r * r)], np.where(dis < r * r)


def sample_uniform_points_in_sphere(amount, radius=1.):
    sphere_points = np.random.uniform(-radius, radius, size=(amount * 2 + 20, 3))
    sphere_points = sphere_points[np.linalg.norm(sphere_points, axis=1) < radius]

    points_available = sphere_points.shape[0]
    if points_available < amount:
        # This is a fallback for the rare case that too few points are inside the unit sphere
        result = np.zeros((amount, 3))
        result[:points_available, :] = sphere_points
        result[points_available:, :] = sample_uniform_points_in_sphere(amount - points_available, radius=radius)
        return result
    else:
        return sphere_points[:amount, :]


def check_ray_triangle_intersection(ray_origins, ray_direction, triangle, epsilon=1e-6):
    """
    Optimized to work for:
        >1 ray_origins
        1 ray_direction multiplied to match the dimension of ray_origins
        1 triangle
    Based on: Answer by BrunoLevy at
    https://stackoverflow.com/questions/42740765/intersection-between-line-and-triangle-in-3d
    Thank you!
    Parameters
    ----------
    ray_origin : torch.Tensor, (n_rays, n_dimensions), (x, 3)
    ray_directions : torch.Tensor, (n_rays, n_dimensions), (1, x)
    triangle : torch.Tensor, (n_points, n_dimensions), (3, 3)
    Return
    ------
    intersection : boolean (n_rays,)
    Test
    ----
    triangle = torch.Tensor([[0., 0., 0.],
                             [1., 0., 0.],
                             [0., 1., 0.],
                            ]).to(device)
    ray_origins = torch.Tensor([[0.5, 0.25, 0.25],
                                [5.0, 0.25, 0.25],
                               ]).to(device)
    ray_origins = torch.rand((10000, 3)).to(device)
    ray_direction = torch.Tensor([[0., 0., -10.],]).to(device)
    #ray_direction = torch.Tensor([[0., 0., 10.],]).to(device)
    ray_direction = ray_directions.repeat(ray_origins.shape[0], 1)
    check_ray_triangle_intersection(ray_origins, ray_direction, triangle)
    """

    E1 = triangle[1] - triangle[0]  # vector of edge 1 on triangle
    E2 = triangle[2] - triangle[0]  # vector of edge 2 on triangle
    N = torch.cross(E1, E2)  # normal to E1 and E2

    invdet = 1. / -torch.einsum('ji, i -> j', ray_direction, N)  # inverse determinant

    A0 = ray_origins - triangle[0]
    # print('A0.shape: ', A0.shape)
    # print('ray_direction.shape: ', ray_direction.shape)
    DA0 = torch.cross(A0, ray_direction.repeat(A0.size(0), 1), dim=1)

    u = torch.einsum('ji, i -> j', DA0, E2) * invdet
    v = -torch.einsum('ji, i -> j', DA0, E1) * invdet
    t = torch.einsum('ji, i -> j', A0, N) * invdet

    intersection = (t >= 0.0) * (u >= 0.0) * (v >= 0.0) * ((u + v) <= 1.0)

    return intersection


def hidden_surface_remove(vertices, triangle_idx, direction=None, epsilon=1e-6):
    if direction is None:
        direction = [0., 0., -1.]
    direction = -np.array(direction)
    vertices, triangle_idx = np.array(vertices, dtype=float), np.array(triangle_idx, dtype=int)
    vn, fn = vertices.shape[0], triangle_idx.shape[0]
    triangles = vertices[triangle_idx]
    ray_directions = np.repeat(direction[np.newaxis, :], vn, axis=0)
    index_triangle, index_ray, locations = trimesh.ray.ray_triangle.ray_triangle_id(triangles, vertices, ray_directions)
    hidden_idx = np.where(np.linalg.norm(vertices[index_ray] - locations, axis=1) > epsilon)[0]
    hidden_idx = np.unique(index_ray[hidden_idx])

    mask = np.ones(vn)
    mask[hidden_idx] = 0
    return mask


def remove_vertices_from_mesh(vertices, triangle_idx, mask):
    """
    :param vertices: (n, 3) float
    :param triangle_idx: (m, 3) int
    :param mask: (n,) where 1 indicates preserve
    :return: vertices, triangle_idx after remove
    """
    import pymeshlab
    indicator = mask
    ms = pymeshlab.MeshSet()
    face_array_matrix_crop = remove_face_by_point_idx(triangle_idx, indicator)
    face_array_matrix_crop = np.array(face_array_matrix_crop)

    tri_crop = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=face_array_matrix_crop)
    ms.add_mesh(tri_crop)
    ms.remove_unreferenced_vertices()
    result_mesh = ms.current_mesh()

    return result_mesh.vertex_matrix(), result_mesh.face_matrix()


def rays_triangles_intersection(ray_origins, ray_direction, triangle, epsilon=1e-6):
    """

    :param ray_origins: torch.Tensor, (n_rays, n_dimensions), (n, 3)
    :param ray_direction:  torch.Tensor, (n_rays, n_dimensions), (n, 3)
    :param triangle:  torch.Tensor, (n_triangles, n_points, n_dimensions), (m, 3, 3)
    :param epsilon: error tolerant
    :return: boolean (h_rays,)
    """

    E1 = triangle[:, 1] - triangle[:, 0]  # vector of edge 1 on triangle
    E2 = triangle[:, 2] - triangle[:, 0]  # vector of edge 2 on triangle
    N = torch.cross(E1, E2, dim=1)  # normal to E1 and E2

    invdet = 1. / -torch.einsum('ji, i -> j', ray_direction, N)  # inverse determinant

    A0 = ray_origins - triangle[0]
    # print('A0.shape: ', A0.shape)
    # print('ray_direction.shape: ', ray_direction.shape)
    DA0 = torch.cross(A0, ray_direction.repeat(A0.size(0), 1), dim=1)

    u = torch.einsum('ji, i -> j', DA0, E2) * invdet
    v = -torch.einsum('ji, i -> j', DA0, E1) * invdet
    t = torch.einsum('ji, i -> j', A0, N) * invdet

    intersection = (t >= 0.0) * (u >= 0.0) * (v >= 0.0) * ((u + v) <= 1.0)

    return intersection


def delaunay_mesh_in_2d(vertices):
    vertices2d = vertices[:, :2]
    tri = Delaunay(vertices2d)
    return vertices, tri.simplices


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def create_grid(N=256, mask_size=1., bbox_size=2.):
    N = int(N)
    grid_length = bbox_size / N
    s = np.arange(N)
    x, y, z = np.meshgrid(s, s, s)
    x, y, z = x.flatten(), y.flatten(), z.flatten()
    grid_points_int = np.vstack((x, y, z)).T
    grid_points = grid_points_int.astype(float)
    grid_points -= np.ones(3) * N / 2.
    grid_points *= grid_length

    vector_len = np.linalg.norm(grid_points, axis=1)
    mask = np.where(vector_len <= mask_size, 1, 0).astype(np.int32)
    # mask[grid_points[:, 0] > 0] = 0
    return grid_points, grid_points_int, mask.reshape((N, N, N)).astype(bool)


def create_grid_2d(N=256):
    N = int(N)
    grid_length = 1. / N
    s = np.arange(N)
    x, y = np.meshgrid(s, s)
    x, y = x.flatten(), y.flatten()
    grid_points_int = np.vstack((x, y)).T
    grid_points = grid_points_int.astype(float)
    grid_points -= np.ones(3) * N / 2.
    grid_points *= grid_length

    return grid_points, grid_points_int


def get_mgrid(side_len, dim=2):
    """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1."""
    if isinstance(side_len, int):
        side_len = dim * (side_len,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:side_len[0], :side_len[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (side_len[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (side_len[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:side_len[0], :side_len[1], :side_len[2]], axis=-1)[None, ...].astype(
            np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(side_len[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (side_len[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (side_len[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.tensor(pixel_coords).view(-1, dim)
    return pixel_coords


def texture_transfer(mtl_name, obj_vertices,
                     ldmk_path='/home/zhengmingwu_2020/ImFace-LIF/dataset/landmark_indices.npz'):
    '''
        用于将有贴图的文件贴图于没有贴图的文件
        mtl_name：有贴图文件名
        obj_vertices：需要进行贴图的点云
    '''
    from utils.fileio import OBJ
    def get_vertical2d(triangle_id, vertices3d, mat2d, mat3d, face2d, face3d):
        tmat = mat3d[triangle_id]
        uv = vertices3d - face3d[triangle_id, 0, :]
        uv = uv[:, 0:2]
        uv = uv.reshape((uv.shape[0], 1, 2))
        ab = np.matmul(uv, tmat)
        xy = np.matmul(ab, mat2d[triangle_id])
        xy = xy.reshape((xy.shape[0], 2))
        return xy + face2d[triangle_id, 0, :]

    lm_list_v10 = np.load(ldmk_path)['v10']
    model = trimesh.load_mesh(mtl_name, process=False, maintain_order=True)
    vertices, faces = model.vertices, model.faces
    landmark = vertices[lm_list_v10]
    nose_tip = landmark[30]
    vertices -= nose_tip
    vertices /= 100
    vertices[:, 2] += 0.4

    model.vertices = vertices

    (closest_points,
     distances,
     triangle_id) = model.nearest.on_surface(obj_vertices)

    x = OBJ(mtl_name)
    face2d = np.array(x.face2d)
    face3d = model.triangles

    # 求解逆矩阵等，获取二维点做准备
    f0 = face2d[:, 0]
    f1 = face2d[:, 1]
    f2 = face2d[:, 2]
    f12 = np.concatenate([f1, f2], axis=1)
    f12.resize((f12.shape[0], 2, 2))
    f00 = np.concatenate([f0, f0], axis=1)
    f00.resize((f00.shape[0], 2, 2))
    mat2d = f12 - f00

    f0 = face3d[:, 0, 0:2]
    f1 = face3d[:, 1, 0:2]
    f2 = face3d[:, 2, 0:2]
    f12 = np.concatenate([f1, f2], axis=1)
    f12.resize((f12.shape[0], 2, 2))
    f00 = np.concatenate([f0, f0], axis=1)
    f00.resize((f00.shape[0], 2, 2))
    mat3d = f12 - f00
    mat3d = np.linalg.inv(mat3d)
    # 三维点对应的二维点
    vertices2d = get_vertical2d(triangle_id, closest_points, mat2d, mat3d, face2d, face3d)
    return vertices2d


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    # assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()

def transform_fit(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t

if __name__ == '__main__':
    from time import time
    from tqdm import tqdm
    import datetime

    # ------------ Test Hidden Surface --------------#
    vs, fs = fileio.load_obj_meshlab(r'E:\ProcessedData\PixelFaceCutting\9mm\00194\1.obj')
    mask = hidden_surface_remove(vs, fs)
    vs_new, fs_new = remove_vertices_from_mesh(vs, fs, mask)
    # meshio.write_off(r'sketches\00194_1_hidden.off', vs_new, fs_new)
    # quit()

    # ------------ Test Delaunay Remesh -------------#
    vs_remesh, fs_remesh = delaunay_mesh_in_2d(vs_new)
    fileio.write_off(r'sketches\00194_1_remesh.off', vs_remesh, fs_remesh)
    quit()

    # ------------ Test Inside Judgement --------------#
    vs, fs = fileio.load_obj_meshlab(r'E:\ProcessedData\BU3D_Aligned\F0029\F0029_AN02WH_F3D.obj')

    points = sample_uniform_points_in_sphere(250000, radius=1)

    points = torch.tensor(points).float().cuda()
    num_sec = torch.zeros(points.shape[0])
    ray_direction = torch.tensor([[0., 0., -1.]]).float().cuda()
    st = time()

    for i in tqdm(range(fs.shape[0])):
        triangle = torch.tensor([
            vs[fs[i, 0]],
            vs[fs[i, 1]],
            vs[fs[i, 2]]
        ]).float().cuda()
        # print(triangle)
        # print(points.size(), ray_direction.size(), triangle.size())
        secs = check_ray_triangle_intersection(points, ray_direction, triangle)
        num_sec += secs.cpu()
    # num_sec %= 2

    c = np.array((num_sec / 5) * 255).astype(np.uint8)
    c = cv2.applyColorMap(c, cv2.COLORMAP_JET)
    # print(c.shape) c.squeeze()
    color = np.array((num_sec % 2) * 255)  # outside points are white
    fileio.write_color_obj('sketches/test_ray_inside.txt', points.cpu(), np.array([color, color, color]).T)
    p_addon = np.vstack((vs, np.array(points.cpu())))
    print(fs.shape)
    fileio.write_off('sketches/test_ray_inside.off', p_addon, fs)

    print(str(datetime.timedelta(seconds=time() - st)))
