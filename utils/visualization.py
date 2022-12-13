import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.metrics import roc_curve, auc


def show_roc(score, label, color='r', lw=2):
    plt.figure()
    plt.figure(figsize=(5, 5))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    fpr, tpr, thresholds = roc_curve(label, score)
    roc_auc = auc(fpr, tpr)
    plt.semilogx(fpr, tpr, color=color,
                 lw=lw, label='AUC = {:.2%}'.format(roc_auc))
    plt.legend(loc="lower right")
    plt.show()


def show_contour_plot(array_2d, mode='log'):
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
    plt.show()


def lin2img(tensor, image_resolution=None):
    batch_size, num_samples, channels = tensor.shape
    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]

    return tensor.permute(0, 2, 1).view(batch_size, channels, height, width)


def show_sdf_2d(sdfs, w, h):
    canvas = np.zeros((h, w))
    idx = np.array(sdfs[:, :2], dtype=np.int32)
    canvas[idx[:, 1], idx[:, 0]] = sdfs[:, 2]
    plt.imshow(canvas)
    plt.show()


def show_image(img):
    plt.imshow(img)
    plt.show()


def show_hist(data):
    data = np.array(data)
    valid_data = data[np.nonzero(data)]
    plt.hist(valid_data)
    plt.show()


def show_scatter(pts):
    pts = np.array(pts)
    x, y = pts[:, 0], pts[:, 1]
    plt.figure(figsize=(10, 10))
    plt.scatter(x, y, s=18.)
    for i in range(pts.shape[0]):
        plt.annotate(i, xy=(x[i], y[i]), fontsize=12)  # 这里xy是需要标记的坐标，xytext是对应的标签坐标
    plt.show()


def get_jet_color(array, clip_min=None, clip_max=None):
    array = np.array(array).flatten()
    if (clip_min is None) and (clip_max is None):
        array_8bit = array
    else:
        array_8bit = np.clip(array, clip_min, clip_max)
    array_8bit = (array_8bit - np.min(array_8bit)) / (np.max(array_8bit) - np.min(array_8bit))
    array_8bit *= 255
    array_8bit = array_8bit.astype(np.uint8)
    colors = cv2.applyColorMap(array_8bit, cv2.COLORMAP_JET)
    colors = cv2.cvtColor(colors, cv2.COLOR_BGR2RGB)
    return colors.reshape(-1, 3)


# def get_jet_color_corr(array):
#     array = np.array(array).flatten()
#     array = np.clip(array, -1, 1)
#     max_val = np.max(np.abs(array))
#     array = ((array / max_val) + 1) / 2.
#     array *= 255
#     array_8bit = array.astype(np.uint8)
#     colors = cv2.applyColorMap(array_8bit, cv2.COLORMAP_JET)
#     colors = cv2.cvtColor(colors, cv2.COLOR_BGR2RGB)
#     return colors.reshape(-1, 3)

def get_jet_color_corr(array, p=0.5):
    array = np.array(array).flatten()
    array = np.clip(array, -1, 1)

    array_pos = np.where(array > 0, array, 0)
    array_neg = np.where(array < 0, array, 0)
    array_neg_ = -np.power(-array_neg, p)
    array_pos_ = np.power(array_pos, p)
    array = array_neg_ + array_pos_

    max_val = np.max(np.abs(array))
    array = ((array / max_val) + 1) / 2.
    array *= 255
    array_8bit = array.astype(np.uint8)
    colors = cv2.applyColorMap(array_8bit, cv2.COLORMAP_JET)
    colors = cv2.cvtColor(colors, cv2.COLOR_BGR2RGB)
    return colors.reshape(-1, 3)


def get_point_color(mesh_points, clip_min=None, clip_max=None):
    if (clip_min is None) and (clip_max is None):
        mesh_colors = mesh_points
    else:
        mesh_colors = np.clip(mesh_points, clip_min, clip_max)
    mesh_colors = (mesh_colors - np.min(mesh_colors)) / (np.max(mesh_colors) - np.min(mesh_colors))
    mesh_colors *= 255
    return mesh_colors.astype(np.uint8)


if __name__ == "__main__":
    pass
