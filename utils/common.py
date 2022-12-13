import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import csv
import torch
from numba import jit
import trimesh
from scipy.spatial import cKDTree as KDTree


def transformation_4d(pcl, matrix_4x4):
    return matrix_4x4.dot(np.vstack((pcl.T, np.ones_like(pcl.T)[0, :][None, ...]))).T[:, :3]


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def dict_to(dict, device):
    for key in dict.keys():
        dict[key] = dict[key].to(device)
    return dict


def bytes2human(n):
    # http://code.activestate.com/recipes/578019
    # >>> bytes2human(10000)
    # '9.8K'
    # >>> bytes2human(100001221)
    # '95.4M'
    symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = {}
    for i, s in enumerate(symbols):
        prefix[s] = 1 << (i + 1) * 10
    for s in reversed(symbols):
        if n >= prefix[s]:
            value = float(n) / prefix[s]
            return '%.2f%s' % (value, s)
    return "%sB" % n


def compute_chamfer(recon_pts, gt_pts, f_score_threshold=0.01, factor_to_mm=100):
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
    l2_chamfer = (completeness.mean() + accuracy.mean()) / 2
    # F-score
    f_completeness = np.mean(completeness <= f_score_threshold)
    f_accuracy = np.mean(accuracy <= f_score_threshold)
    f_score = factor_to_mm * 2 * f_completeness * f_accuracy / (f_completeness + f_accuracy)  # harmonic mean
    return l2_chamfer * factor_to_mm, f_score


def compute_chamfer_mesh(recon_v, recon_f, gt_v, gt_f, num_pts=150000, factor_to_mm=100):
    recon_mesh = trimesh.Trimesh(recon_v, recon_f)
    recon_pts = trimesh.sample.sample_surface(recon_mesh, num_pts)[0]
    gt_mesh = trimesh.Trimesh(gt_v, gt_f)
    gt_pts = trimesh.sample.sample_surface(gt_mesh, num_pts)[0]
    return compute_chamfer(recon_pts, gt_pts, factor_to_mm)


def filename_to_exp_type(filename):
    return int(os.path.basename(filename).split('.')[0].split('_')[0])
