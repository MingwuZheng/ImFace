from glob import glob
import os
import trimesh
import numpy as np
from sklearn.neighbors import NearestNeighbors
from utils import common
from scipy.spatial import cKDTree as KDTree


def best_fit_transform(A, B):
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


def compute_chamfer_i3dmm(recon_pts, gt_pts, f_score_threshold=0.01, facetor_to_mm=100):
    # one direction
    gen_points_kd_tree = KDTree(recon_pts)
    completeness, _ = gen_points_kd_tree.query(gt_pts)

    # other direction
    gt_points_kd_tree = KDTree(gt_pts)
    accuracy, _ = gt_points_kd_tree.query(recon_pts)

    # max_side_length = np.max(bb_max - bb_min)
    # f_score_threshold = 0.01  # deep structured implicit functions sets tau = 0.01
    # L2 chamfer
    l2_chamfer = ((completeness).mean() + (accuracy).mean()) / 2
    # F-score
    f_completeness = np.mean(completeness <= f_score_threshold)
    f_accuracy = np.mean(accuracy <= f_score_threshold)
    f_score = facetor_to_mm * 2 * f_completeness * f_accuracy / (f_completeness + f_accuracy)  # harmonic mean
    return l2_chamfer * facetor_to_mm, f_score, (completeness).mean() * facetor_to_mm, (accuracy).mean() * facetor_to_mm


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)

        # compute the transformation between the current source and nearest destination points
        T, _, _ = best_fit_transform(src[:m, :].T, dst[:m, indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T, _, _ = best_fit_transform(A, src[:m, :].T)

    return T, distances, i


def evaluate_imface(path='/home/zhengmingwu_2020/ImFace/dataset/FacescapeAlign'):
    num_pts = 250000
    # target_path = '/home/zhengmingwu_2020/ImFace-LIF/result/2022-03-04_13-30/fit/2022-03-06_08-51/fit_results'
    target_path = '/home/zhengmingwu_2020/ImFace-LIF/result/2022-03-04_13-30/fit/2022-03-07_09-43/fit_results'
    # target_path = '/home/zhengmingwu/FitEvaluate/result/facescape_pub_flame_crop'
    # target_path = '/home/zhengmingwu/FitEvaluate/result/facescape_pub_flame_hr'
    objs = glob(os.path.join(target_path, '*', '?.ply'))
    objs.extend(glob(os.path.join(target_path, '*', '??.ply')))
    chamfer_dists = []
    f_scores = []
    cmplts = []
    accs = []
    for obj in objs:
        id_name = obj.split('/')[-2]
        exp_idx = os.path.basename(obj).split('.')[0]
        gt = os.path.join(path, id_name, exp_idx + '.obj')

        recon_mesh = trimesh.load(obj)
        if isinstance(recon_mesh, trimesh.Scene):
            recon_mesh = recon_mesh.dump().sum()

        recon_pts = trimesh.sample.sample_surface(recon_mesh, num_pts)[0]

        gt_mesh = trimesh.load(gt)
        if isinstance(gt_mesh, trimesh.Scene):
            gt_mesh = gt_mesh.dump().sum()
        gt_pts = trimesh.sample.sample_surface(gt_mesh, num_pts)[0]

        distances, indices = nearest_neighbor(recon_pts, gt_pts)

        T, _, _ = icp(recon_pts, gt_pts[indices])

        recon_pts = common.transformation_4d(recon_pts, T)

        cd, f_score, cmplt, acc = compute_chamfer_i3dmm(recon_pts, gt_pts)

        chamfer_dists.append(cd)
        f_scores.append(f_score)
        cmplts.append(cmplt)
        accs.append(acc)
        out_str = 'CamfDist: {:.4f}mm F-Score:{:.2f} Completeness: {:.2f} Accuracy: {:.2f}'.format(cd, f_score, cmplt,
                                                                                                   acc)
        print(obj, out_str)

    # final_out = 'Average Chamfer Distance: '
    final_out = 'CamfDist:{:.6f}mm F-Score:{:.2f}'.format(np.mean(chamfer_dists), np.mean(f_scores))

    np.save(os.path.join(target_path, 'errors.npy'), {
        'cf': np.array(chamfer_dists),
        'cmplt': np.array(cmplts),
        'acc': np.array(accs),
        'f_score': np.array(f_scores)
    })
    print(final_out)


# evaluate_imface()
