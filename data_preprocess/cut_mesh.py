import trimesh
import numpy as np
import os


class AddTriangle:
    def __init__(self, file_name, to_name):
        self.file_name = file_name
        self.to_name = to_name

    def find_dot2(self, face, vertice, final_face):
        f = vertice[face]
        (a, b, c) = f.shape
        for i in range(a):
            ff = f[i]
            ff = np.multiply(ff, ff).sum(axis=1)
            # find points in the circle
            pos = np.argmin(ff, axis=0)
            x1, y1, z1 = f[i][pos]
            p1 = face[i][pos]
            x2, y2, z2 = f[i][(pos + 1) % 3]
            p2 = face[i][(pos + 1) % 3]
            x3, y3, z3 = f[i][(pos + 2) % 3]
            p3 = face[i][(pos + 2) % 3]

            lx = x1
            wx = x2
            while (1):
                tx = (lx + wx) / 2
                ty = (tx - x1) / (x2 - x1) * (y2 - y1) + y1
                tz = (tx - x1) / (x2 - x1) * (z2 - z1) + z1
                dis = tx ** 2 + ty ** 2 + tz ** 2
                if (abs(dis - 1) <= 1e-6):
                    break
                if dis > 1:
                    wx = tx
                else:
                    lx = tx
            vertice = np.concatenate([vertice, [[tx, ty, tz]]], axis=0)
            tpos1 = vertice.shape[0] - 1

            lx = x1
            wx = x3
            while (1):
                tx2 = (lx + wx) / 2
                ty2 = (tx2 - x1) / (x3 - x1) * (y3 - y1) + y1
                tz2 = (tx2 - x1) / (x3 - x1) * (z3 - z1) + z1
                dis = tx2 ** 2 + ty2 ** 2 + tz2 ** 2
                if (abs(dis - 1) <= 1e-6):
                    break
                if dis > 1:
                    wx = tx2
                else:
                    lx = tx2
            vertice = np.concatenate([vertice, [[tx2, ty2, tz2]]], axis=0)
            tpos2 = vertice.shape[0] - 1
            final_face = np.concatenate([final_face, [[p1, tpos2, tpos1]]], axis=0)
        return vertice, final_face

    def find_dot(self, face, vertice, final_face):
        f = vertice[face]
        (a, b, c) = f.shape
        for i in range(a):
            ff = f[i]
            ff = np.multiply(ff, ff).sum(axis=1)

            pos = np.argmax(ff, axis=0)
            x1, y1, z1 = f[i][pos]
            p1 = face[i][pos]
            x2, y2, z2 = f[i][(pos + 1) % 3]
            p2 = face[i][(pos + 1) % 3]
            x3, y3, z3 = f[i][(pos + 2) % 3]
            p3 = face[i][(pos + 2) % 3]

            wx = x1
            lx = x2
            while (1):
                tx = (lx + wx) / 2
                ty = (tx - x1) / (x2 - x1) * (y2 - y1) + y1
                tz = (tx - x1) / (x2 - x1) * (z2 - z1) + z1
                dis = tx ** 2 + ty ** 2 + tz ** 2
                if (abs(dis - 1) <= 1e-6):
                    break
                if dis > 1:
                    wx = tx
                else:
                    lx = tx
            vertice = np.concatenate([vertice, [[tx, ty, tz]]], axis=0)
            tpos1 = vertice.shape[0] - 1

            wx = x1
            lx = x3
            while (1):
                tx2 = (lx + wx) / 2
                ty2 = (tx2 - x1) / (x3 - x1) * (y3 - y1) + y1
                tz2 = (tx2 - x1) / (x3 - x1) * (z3 - z1) + z1
                dis = tx2 ** 2 + ty2 ** 2 + tz2 ** 2
                if (abs(dis - 1) <= 1e-6):
                    break
                if dis > 1:
                    wx = tx2
                else:
                    lx = tx2

            vertice = np.concatenate([vertice, [[tx2, ty2, tz2]]], axis=0)
            tpos2 = vertice.shape[0] - 1
            if x2 > x3:
                if tx > tx2:
                    final_face = np.concatenate([final_face, [[p2, p3, tpos1], [p3, tpos1, tpos2]]], axis=0)
                else:
                    final_face = np.concatenate([final_face, [[p2, p3, tpos2], [p3, tpos1, tpos2]]], axis=0)
            else:
                if tx > tx2:
                    final_face = np.concatenate([final_face, [[p2, p3, tpos1], [p2, tpos1, tpos2]]], axis=0)
                else:
                    final_face = np.concatenate([final_face, [[p2, p3, tpos2], [p2, tpos1, tpos2]]], axis=0)
        return vertice, final_face

    def getFile(self):
        model = trimesh.load_mesh(self.file_name, process=False, maintain_order=True)

        faces = model.faces
        vertice = model.vertices

        # triangles in the circle
        v = np.multiply(vertice, vertice).sum(axis=1)
        v = v <= 1.0

        face_condition = v[faces]
        face_condition = face_condition.sum(axis=1)
        face3 = faces[face_condition >= 3]

        # triangles with two vertices in the circle
        face2 = faces[face_condition == 2]
        vertice, myfaces = self.find_dot(face2, vertice, face3)

        # triangles with one vertices in the circle
        face1 = faces[face_condition == 1]
        vertice, myfaces = self.find_dot2(face1, vertice, myfaces)

        mesh = trimesh.Trimesh(faces=myfaces, vertices=vertice)
        mesh.export(file_obj=self.to_name, file_type="obj")

### todo: optimize
def find_dot2(face, vertice, final_face, r, circle):
    f = vertice[face]
    (a, b, c) = f.shape
    for i in range(a):
        ff = f[i] - circle
        ff = np.multiply(ff, ff).sum(axis=1)
        pos = np.argmin(ff, axis=0)
        x1, y1, z1 = f[i][pos]
        p1 = face[i][pos]
        x2, y2, z2 = f[i][(pos + 1) % 3]
        p2 = face[i][(pos + 1) % 3]
        x3, y3, z3 = f[i][(pos + 2) % 3]
        p3 = face[i][(pos + 2) % 3]

        lx = x1
        wx = x2
        while (1):
            tx = (lx + wx) / 2
            ty = (tx - x1) / (x2 - x1) * (y2 - y1) + y1
            tz = (tx - x1) / (x2 - x1) * (z2 - z1) + z1
            dis = (tx - circle[0]) ** 2 + (ty - circle[1]) ** 2 + (tz - circle[2]) ** 2
            if (abs(dis - r) <= 1e-6):
                break
            if dis > r:
                wx = tx
            else:
                lx = tx
        vertice = np.concatenate([vertice, [[tx, ty, tz]]], axis=0)
        tpos1 = vertice.shape[0] - 1

        lx = x1
        wx = x3
        while (1):
            tx2 = (lx + wx) / 2
            ty2 = (tx2 - x1) / (x3 - x1) * (y3 - y1) + y1
            tz2 = (tx2 - x1) / (x3 - x1) * (z3 - z1) + z1
            dis = (tx2 - circle[0]) ** 2 + (ty2 - circle[1]) ** 2 + (tz2 - circle[2]) ** 2
            if (abs(dis - r) <= 1e-6):
                break
            if dis > r:
                wx = tx2
            else:
                lx = tx2
        vertice = np.concatenate([vertice, [[tx2, ty2, tz2]]], axis=0)
        tpos2 = vertice.shape[0] - 1
        final_face = np.concatenate([final_face, [[p1, tpos2, tpos1]]], axis=0)
    return vertice, final_face


def find_dot(face, vertice, final_face, r, circle):
    f = vertice[face]
    (a, b, c) = f.shape
    for i in range(a):
        ff = f[i] - circle
        ff = np.multiply(ff, ff).sum(axis=1)

        pos = np.argmax(ff, axis=0)
        x1, y1, z1 = f[i][pos]
        p1 = face[i][pos]
        x2, y2, z2 = f[i][(pos + 1) % 3]
        p2 = face[i][(pos + 1) % 3]
        x3, y3, z3 = f[i][(pos + 2) % 3]
        p3 = face[i][(pos + 2) % 3]

        wx = x1
        lx = x2
        while (1):
            tx = (lx + wx) / 2
            ty = (tx - x1) / (x2 - x1) * (y2 - y1) + y1
            tz = (tx - x1) / (x2 - x1) * (z2 - z1) + z1
            dis = (tx - circle[0]) ** 2 + (ty - circle[1]) ** 2 + (tz - circle[2]) ** 2
            if (abs(dis - r) <= 1e-6):
                break
            if dis > r:
                wx = tx
            else:
                lx = tx
        vertice = np.concatenate([vertice, [[tx, ty, tz]]], axis=0)
        tpos1 = vertice.shape[0] - 1

        wx = x1
        lx = x3
        while (1):
            tx2 = (lx + wx) / 2
            ty2 = (tx2 - x1) / (x3 - x1) * (y3 - y1) + y1
            tz2 = (tx2 - x1) / (x3 - x1) * (z3 - z1) + z1
            dis = (tx2 - circle[0]) ** 2 + (ty2 - circle[1]) ** 2 + (tz2 - circle[2]) ** 2
            if (abs(dis - r) <= 1e-6):
                break
            if dis > r:
                wx = tx2
            else:
                lx = tx2

        vertice = np.concatenate([vertice, [[tx2, ty2, tz2]]], axis=0)
        tpos2 = vertice.shape[0] - 1

        final_face = np.concatenate([final_face, [[p2, tpos2, tpos1], [p2, p3, tpos2]]], axis=0)
    return vertice, final_face

"""
input：
    vertices： nx3 points cloud
    mesh：     mx3 faces
    circle：   center of the circle
    r：        radius
output：
    new_vertice： new points after cutting
    new_mesh：    new faces after cutting
"""
def cut(vertices, mesh, circle, r):
    r = r ** 2
    v = vertices - circle
    v = np.multiply(v, v).sum(axis=1)
    v = v <= r

    face_condition = v[mesh]
    face_condition = face_condition.sum(axis=1)
    face3 = mesh[face_condition >= 3]

    face2 = mesh[face_condition == 2]
    new_vertice, myfaces = find_dot(face2, vertices, face3, r, circle)

    face1 = mesh[face_condition == 1]
    new_vertice, new_mesh = find_dot2(face1, new_vertice, myfaces, r, circle)

    return new_vertice, new_mesh


