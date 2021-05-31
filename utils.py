import numpy as np


def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = np.pad(img, ((np.abs(np.minimum(0, y1)), np.maximum(y2 - img.shape[0], 0)),
                       (np.abs(np.minimum(0, x1)), np.maximum(x2 - img.shape[1], 0)), (0, 0)), mode="constant")
    y1 += np.abs(np.minimum(0, y1))
    y2 += np.abs(np.minimum(0, y1))
    x1 += np.abs(np.minimum(0, x1))
    x2 += np.abs(np.minimum(0, x1))
    return img, x1, x2, y1, y2


def img_crop(img, xmin, ymin, xmax, ymax):
    if xmin < 0 or ymin < 0 or xmax > img.shape[1] or ymax > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, xmin, xmax, ymin, ymax)
    else:
        x1 = xmin
        x2 = xmax
        y1 = ymin
        y2 = ymax
    return img[y1:y2, x1:x2, :]
