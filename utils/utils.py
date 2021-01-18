import requests
import os
import json
import h5py
import pdb
import numpy as np
from PIL import Image
import hdfdict
from tqdm import tqdm


def download_from_url(url, path):
    """Download file, with logic (from tensor2tensor) for Google Drive"""
    if 'drive.google.com' not in url:
        print('Downloading %s; may take a few minutes' % url)
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        with open(path, "wb") as file:
            file.write(r.content)
        return
    print('Downloading from Google Drive; may take a few minutes')
    confirm_token = None
    session = requests.Session()
    response = session.get(url, stream=True)
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            confirm_token = v

    if confirm_token:
        url = url + "&confirm=" + confirm_token
        response = session.get(url, stream=True)

    chunk_size = 16 * 1024
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)


def scale(bb, scale_x: float, scale_y: float):
    """
    Scale the box with horizontal and vertical scaling factors
    """
    bb = np.array(bb)
    bb[:, 0::2] *= scale_x
    bb[:, 1::2] *= scale_y
    return bb


def get_size(image_size):
    min_size = 600
    max_size = 1000
    w, h = image_size
    size = min_size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))
    if (w <= h and w == size) or (h <= w and h == size):
        return (w, h)
    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)
    return (ow, oh)


def merge_updown_sg(updown_folder, sg_folder):
    # sg paths
    info_train_path = os.path.join(sg_folder, 'custom_data_info_train.json')
    info_val_path = os.path.join(sg_folder, 'custom_data_info_val.json')
    pred_train_path = os.path.join(sg_folder, 'custom_prediction_train.json')
    pred_val_path = os.path.join(sg_folder, 'custom_prediction_val.json')

    # updown paths
    updown_path = os.path.join(updown_folder, 'coco_detections.hdf5')

    # sg
    with open(info_train_path, 'r') as f:
        info_train = json.load(f)
    with open(info_val_path, 'r') as f:
        info_val = json.load(f)

    with open(pred_train_path, 'r') as f:
        pred_train = json.load(f)
    with open(pred_val_path, 'r') as f:
        pred_val = json.load(f)

    # updown
    updown = h5py.File(updown_path, 'r')
    updown_copy = {}
    for image_idx in tqdm(range(len(info_train['idx_to_files']))):
        image_path = info_train['idx_to_files'][image_idx]
        boxes1 = pred_train[str(image_idx)]['bbox']
        # get bbox scores
        bbox_scores = pred_train[str(image_idx)]['bbox_labels']
        # get rel pairs
        rel_pairs = pred_train[str(image_idx)]['rel_pairs']
        # get rel labels
        rel_labels = pred_train[str(image_idx)]['rel_labels']
        # get rel scores
        rel_scores = pred_train[str(image_idx)]['rel_scores']

        index = int(image_path.split('_')[-1].split('.')[0])
        box_index = str(index) + '_boxes'
        updown_copy[box_index] = updown[box_index]
        boxes2 = updown[box_index][:]
        # senity check
        assert len(boxes1) == len(boxes2)
        cls_prob_index = str(index) + '_cls_prob'
        updown_copy[cls_prob_index] = updown[cls_prob_index]
        features_index = str(index) + '_features'
        updown_copy[features_index] = updown[features_index]
        bbox_scores_index = str(index) + '_boxes_vg_scores'
        updown_copy[bbox_scores_index] = bbox_scores
        rel_pairs_index = str(index) + '_rel_pairs'
        updown_copy[rel_pairs_index] = rel_pairs
        rel_labels_index = str(index) + '_rel_labels'
        updown_copy[rel_labels_index] = rel_labels
        rel_scores_index = str(index) + '_rel_scores'
        updown_copy[rel_scores_index] = rel_scores

    for image_idx in tqdm(range(len(info_val['idx_to_files']))):
        image_path = info_val['idx_to_files'][image_idx]
        boxes1 = pred_val[str(image_idx)]['bbox']
        # get bbox scores
        bbox_scores = pred_val[str(image_idx)]['bbox_labels']
        # get rel pairs
        rel_pairs = pred_val[str(image_idx)]['rel_pairs']
        # get rel labels
        rel_labels = pred_val[str(image_idx)]['rel_labels']
        # get rel scores
        rel_scores = pred_val[str(image_idx)]['rel_scores']

        index = int(image_path.split('_')[-1].split('.')[0])
        box_index = str(index) + '_boxes'
        updown_copy[box_index] = updown[box_index]
        boxes2 = updown[box_index][:]
        # senity check
        assert len(boxes1) == len(boxes2)
        cls_prob_index = str(index) + '_cls_prob'
        updown_copy[cls_prob_index] = updown[cls_prob_index]
        features_index = str(index) + '_features'
        updown_copy[features_index] = updown[features_index]
        bbox_scores_index = str(index) + '_boxes_vg_scores'
        updown_copy[bbox_scores_index] = bbox_scores
        rel_pairs_index = str(index) + '_rel_pairs'
        updown_copy[rel_pairs_index] = rel_pairs
        rel_labels_index = str(index) + '_rel_labels'
        updown_copy[rel_labels_index] = rel_labels
        rel_scores_index = str(index) + '_rel_scores'
        updown_copy[rel_scores_index] = rel_scores

    fname = os.path.join(updown_folder, 'detection_features_relation.hdf5')
    hdfdict.dump(updown_copy, fname)


updown_folder = '/home/xuewyang/Xuewen/Research/data/COCO'
sg_folder = '/home/xuewyang/Xuewen/Research/model/captioning/causal_motifs_sgdet'
# merge_updown_sg(updown_folder, sg_folder)

# boxes1 = pred_val[str(image_idx)]['bbox']
# image_path = info_val['idx_to_files'][image_idx]
# size = get_size(Image.open(image_path).size)
# pic = Image.open(image_path)
# # pic = Image.open(img_path).resize(size)
# # print(pic.size)
# scale_x = pic.size[0] / size[0]
# scale_y = pic.size[1] / size[1]
# num_obj = len(boxes1)
# boxes1 = scale(boxes1, scale_x, scale_y)
# # get bbox scores
# bbox_scores = pred_val[str(image_idx)]['bbox_labels']
# # get rel pairs
# rel_pairs = pred_val[str(image_idx)]['rel_pairs']
# # get rel labels
# rel_labels = pred_val[str(image_idx)]['rel_labels']
# # get rel scores
# rel_scores = pred_val[str(image_idx)]['rel_scores']
# # print(boxes1)
#
# index = int(image_path.split('_')[-1].split('.')[0])
# box_index = str(index) + '_boxes'
# bbox_scores_index = str(index) + '_boxes_vg_scores'
# updown[bbox_scores_index] = bbox_scores
# rel_pairs_index = str(index) + '_rel_pairs'
# updown[rel_pairs_index] = rel_pairs
# rel_labels_index = str(index) + '_rel_labels'
# updown[rel_labels_index] = rel_labels
# rel_scores_index = str(index) + '_rel_scores'
# updown[rel_scores_index] = rel_scores
# pdb.set_trace()













