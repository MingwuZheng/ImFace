import os, h5py, csv, cv2, plyfile
import numpy as np
from utils import geometry
import json


class OBJ:
    def __init__(self, filename, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.texcoords = []
        # 二维与三维三角面的对应关系，相当于一个三角面与另外一个三角面的对应
        self.face3d = []
        self.face2d = []
        # faces中保存二维到三维的对应关系
        self.faces = {}

        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(tuple(map(float, values[1:3])))
            elif values[0] in ('usemtl', 'usemat'):
                continue
            elif values[0] == 'mtllib':
                continue
            elif values[0] == 'f':
                faces = []
                texds = []
                for v in values[1:]:
                    w = v.split('/')
                    face = self.vertices[int(w[0]) - 1]
                    faces.append(face)
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords = self.texcoords[int(w[1]) - 1]
                        texds.append(list(texcoords))
                    else:
                        # 不存在这种情况，如果存在说明数据存在缺失
                        texcoords = 0
                    self.faces[texcoords] = face
                self.face3d.append(faces)
                self.face2d.append(texds)
        self.vertices = np.array(self.vertices)
        self.texcoords = np.array(self.texcoords)

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def write_dict(path, dict):
    assert os.path.exists(os.path.dirname(path)), '{} not exist!'.format(path)
    with open(path, 'w') as f:
        f.write(json.dumps(dict))


def read_dict(path, key2int=True):
    assert os.path.exists(path), '{} not exist!'.format(path)
    with open(path, 'r') as f:
        res = json.load(f)
    if key2int:
        res_int = {}
        for k in res.keys():
            res_int[int(k)] = res[k]
        return res_int
    return res


def read_array(path):
    res = []
    with open(path, 'r') as f:
        for line in f:
            row = []
            for data in line.split():
                row.append(float(data))
            res.append(row)
    return np.array(res)


def read_pcl_txt(path):
    '''
    :param path: N rows, 3 columns, float txt
    :return: Nx3 pcl
    '''
    data = np.array([])
    first = True
    for line in open(path, 'r'):
        line.rstrip('\n')
        line = line.split(' ')
        point = np.array([[float(line[0]), float(line[1]), float(line[2])]])
        if first:
            data = point
            first = False
        else:
            data = np.append(data, point, axis=0)
    return data


def write_xyz(pcl, path):  # pcl: n*3
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    objf = open(path, 'w')
    for i in range(len(pcl)):
        objf.writelines(str(pcl[i][0]) + ',' + str(pcl[i][1]) + ',' + str(pcl[i][2]) + '\n')


def write_off(file, vertices, faces):
    """
    Writes the given vertices and faces to OFF.

    :param vertices: vertices as tuples of (x, y, z) coordinates
    :type vertices: [(float)]
    :param faces: faces as tuples of (num_vertices, vertex_id_1, vertex_id_2, ...)
    :type faces: [(int)]
    """

    num_vertices = len(vertices)
    num_faces = len(faces)

    assert num_vertices > 0
    assert num_faces > 0

    with open(file, 'w') as fp:
        fp.write('OFF\n')
        fp.write(str(num_vertices) + ' ' + str(num_faces) + ' 0\n')

        for vertex in vertices:
            assert len(vertex) == 3, 'invalid vertex with %d dimensions found (%s)' % (len(vertex), file)
            fp.write(str(vertex[0]) + ' ' + str(vertex[1]) + ' ' + str(vertex[2]) + '\n')

        for face in faces:
            assert len(face) == 3, 'only triangular faces supported (%s)' % file
            fp.write('3 ')
            for i in range(len(face)):
                assert face[i] >= 0 and face[i] < num_vertices, 'invalid vertex index %d (of %d vertices) (%s)' % (
                    face[i], num_vertices, file)

                fp.write(str(face[i]))
                if i < len(face) - 1:
                    fp.write(' ')

            fp.write('\n')

        # add empty line to be sure
        fp.write('\n')


def load_off(file):
    """
    Reads vertices and faces from an off file.
    :param file: path to file to read
    :type file: str
    :return: vertices and faces as lists of tuples
    :rtype: [(float)], [(int)]
    """

    assert os.path.exists(file), 'file %s not found' % file

    with open(file, 'r') as fp:
        lines = fp.readlines()
        lines = [line.strip() for line in lines]

        # Fix for ModelNet bug were 'OFF' and the number of vertices and faces are
        # all in the first line.
        if len(lines[0]) > 3:
            assert lines[0][:3] == 'OFF' or lines[0][:3] == 'off', 'invalid OFF file %s' % file

            parts = lines[0][3:].split(' ')
            assert len(parts) == 3

            num_vertices = int(parts[0])
            assert num_vertices > 0

            num_faces = int(parts[1])
            assert num_faces > 0

            start_index = 1
        # This is the regular case!
        else:
            assert lines[0] == 'OFF' or lines[0] == 'off', 'invalid OFF file %s' % file

            parts = lines[1].split(' ')
            assert len(parts) == 3

            num_vertices = int(parts[0])
            assert num_vertices > 0

            num_faces = int(parts[1])
            assert num_faces > 0

            start_index = 2

        vertices = []
        for i in range(num_vertices):
            vertex = lines[start_index + i].split(' ')
            vertex = [float(point.strip()) for point in vertex if point != '']
            assert len(vertex) == 3

            vertices.append(vertex)

        faces = []
        for i in range(num_faces):
            face = lines[start_index + num_vertices + i].split(' ')
            face = [index.strip() for index in face if index != '']

            # check to be sure
            for index in face:
                assert index != '', 'found empty vertex index: %s (%s)' % (lines[start_index + num_vertices + i], file)

            face = [int(index) for index in face]

            assert face[0] == len(face) - 1, 'face should have %d vertices but as %d (%s)' % (
                face[0], len(face) - 1, file)
            assert face[0] == 3, 'only triangular meshes supported (%s)' % file
            for index in face:
                assert index >= 0 and index < num_vertices, 'vertex %d (of %d vertices) does not exist (%s)' % (
                    index, num_vertices, file)

            assert len(face) > 1

            faces.append(face)

        return vertices, faces

    assert False, 'could not open %s' % file


def write_color_obj(path, xyz, color):
    xyz = np.array(xyz).astype(float)
    color = color.astype(np.uint8)
    with open(path, 'w') as f:
        for i in range(xyz.shape[0]):
            # s = 'v '
            s = ''
            s += str(xyz[i][0]) + ' ' + str(xyz[i][1]) + ' ' + str(xyz[i][2]) + ' '
            s += str(color[i][0]) + ' ' + str(color[i][1]) + ' ' + str(color[i][2]) + '\n'
            f.write(s)


def write_color_obj_by_sdf_grid(path, sdfs, resolution, clamp=0.003):
    sdfs = np.array(sdfs).flatten()
    sdf_8bit = ((np.clip(sdfs, -clamp, clamp) / clamp + 1) * 255 / 2).astype(np.uint8)
    sdf_c = cv2.applyColorMap(sdf_8bit, cv2.COLORMAP_JET)
    grid_points, _, _ = geometry.create_grid(resolution)
    write_color_obj(path, grid_points, sdf_c.squeeze())


def write_color_obj_by_sdf_point(path, sdfs, points, clamp=0.003):
    sdfs = np.array(sdfs).flatten()
    sdf_8bit = ((np.clip(sdfs, -clamp, clamp) / clamp + 1) * 255 / 2).astype(np.uint8)
    sdf_c = cv2.applyColorMap(sdf_8bit, cv2.COLORMAP_JET)
    write_color_obj(path, points, sdf_c.squeeze())


def read_hdf5(file):
    """
    Read a tensor, i.e. numpy array, from HDF5.
    :param file: path to file to read
    :type file: str
    :param key: key to read
    :type key: str
    :return: tensor
    :rtype: numpy.ndarray
    """

    assert os.path.exists(file), 'file %s not found' % file
    result = {}
    h5f = h5py.File(file, 'r')
    for key in h5f.keys():
        result[key] = h5f[key][()]
    h5f.close()

    return result


def read_csv(file_location):
    with open(file_location, 'r+', newline='') as csv_file:
        reader = csv.reader(csv_file)
        return [row for row in reader]


def write_csv(file_location, rows):
    with open(file_location, 'w+', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for row in rows:
            writer.writerow(row)


def write_plyfile(mesh_points, faces, mesh_colors, ply_filename_out):
    # try writing to the ply file

    num_verts = mesh_points.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    colors_tuple = np.zeros((num_verts,), dtype=[("red", "u1"), ("green", "u1"), ("blue", "u1")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    for i in range(0, num_verts):
        colors_tuple[i] = tuple(mesh_colors[i, :])

    verts_all = np.empty(num_verts, verts_tuple.dtype.descr + colors_tuple.dtype.descr)

    for prop in verts_tuple.dtype.names:
        verts_all[prop] = verts_tuple[prop]

    for prop in colors_tuple.dtype.names:
        verts_all[prop] = colors_tuple[prop]

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_all, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces], text=True)
    # print("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)


def write_texture_obj(vertices, faces, path):
    with open(path, "w") as f:
        f.write("mtllib ./2_neutral.obj.mtl\n")

        for i in range(len(vertices)):
            f.write("v {0} {1} {2}\n".format(vertices[i][0], vertices[i][1], vertices[i][2]))

        for i in range(len(vertices)):
            f.write("vt {0} {1}\n".format(vertices[i][0], vertices[i][1]))

        f.write("usemtl material_0\n")

        for i in range(len(faces)):
            f.write("f {0}/{1} {2}/{3} {4}/{5}\n".format(faces[i][0], faces[i][0], faces[i][1], faces[i][1],
                                                         faces[i][2], faces[i][2]))


def write_mtl(vertices,
              faces,
              texture_vertices,
              to_name,
              to_path,
              from_png='/home/zhengmingwu_2020/ImFace-LIF/corr_color.png',
              png_name=None):
    from shutil import copyfile
    if png_name is None:
        png_name = os.path.basename(from_png)
    if not os.path.exists(to_path):
        os.makedirs(to_path)
    f = open(to_path + "/" + to_name, "w")

    f.write("mtllib ./" + to_name + ".mtl\n")

    for i in range(len(vertices)):
        f.write("v {0} {1} {2}\n".format(vertices[i][0], vertices[i][1], vertices[i][2]))

    for i in range(len(texture_vertices)):
        f.write("vt {0} {1}\n".format(texture_vertices[i][0], texture_vertices[i][1]))

    f.write("usemtl material_0\n")
    faces += 1
    for i in range(len(faces)):
        a1 = faces[i][0]
        a2 = faces[i][1]
        a3 = faces[i][2]
        if a1 == a2 or a1 == a3 or a2 == a3:
            continue
        f.write("f {0}/{1} {2}/{3} {4}/{5}\n".format(faces[i][0], faces[i][0], faces[i][1], faces[i][1], faces[i][2],
                                                     faces[i][2]))
    f.close()

    f = open(to_path + "/" + to_name + ".mtl", "w")
    f.write("newmtl material_0\n")
    f.write("Ka 0.200000 0.200000 0.200000\n")
    f.write("Kd 0.000000 0.000000 0.000000\n")
    f.write("Ks 1.000000 1.000000 1.000000\n")
    f.write("Tr 0.000000\n")
    f.write("illum 2\n")
    f.write("Ns 0.000000\n")
    f.write("Ns 0.000000\n")
    f.write("map_Kd " + png_name)

    copyfile(from_png, to_path + "/" + png_name)


if __name__ == '__main__':
    pass
