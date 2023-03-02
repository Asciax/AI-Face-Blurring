# Inspired from https://github.com/Engineering-Course/CIHP_PGN/blob/master/evaluation/test_inst_part_ap.py

# MIT License

# Copyright (c) 2016 Vladimir Nekrasov

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# MIT License

# Copyright (c) 2021 Angelique Loesch

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# @copyright CEA-LIST/DIASI/SIALV/LVA (2021)
# @author CEA-LIST/DIASI/SIALV/LVA <ccihp@cea.fr>

from __future__ import division
import os
from PIL import Image
import numpy as np
import cv2
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('Set directories of inputs and some variables', add_help=False)

    parser.add_argument('--INST_PART_GT_DIR_txt', default='',
                        help='directory where to find characteristic, attribute, and human lookup tables. ex: /path/where/to/find/Instance_part_val++')
    parser.add_argument('--INST_PART_GT_DIR_png', default='',
                        help='directory where to find characteristic instance part ground truthes. ex: /path/where/to/find/Charac_instance_ids')
    parser.add_argument('--PREDICT_DIR_png', default='',
                        help='directory where to find characteristic (size, pattern or color) instance maps from inference and generate_characteristic_instance_part_ccihp.py. ex: /path/where/to/find/pred_cihp_size_inst_part_maps')
    parser.add_argument('--characteristic_type', default='size',
                        help='it can be size, pattern or color')
    return parser


# compute mask overlap
def compute_mask_iou(mask_gt, masks_pre, mask_gt_area, masks_pre_area):
    """Calculates IoU of the given box with the array of the given boxes.
    mask_gt: [H,W] # a mask of gt
    masks_pre: [num_instances, height, width] predict Instance masks
    mask_gt_area: the gt_mask_area , int
    masks_pre_area: array of length masks_count. including all predicted mask, sum

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    intersection = np.logical_and(mask_gt, masks_pre)
    intersection = np.where(intersection == True, 1, 0).astype(np.uint8)
    intersection = NonZero(intersection)

    mask_gt_areas = np.full(len(masks_pre_area), mask_gt_area)

    union = mask_gt_areas + masks_pre_area[:] - intersection[:]

    iou = intersection / union

    return iou


# compute the number of nonzero in mask
def NonZero(masks):
    """
    :param masks: [N,h,w] a three-dimension array, includes N two-dimension mask arrays
    :return: (N) return a tuple with length N. N is the number of non-zero elements in the two-dimension mask
    """
    area = []
    for i in masks:
        _, a = np.nonzero(i)
        area.append(a.shape[0])
    area = tuple(area)
    return area


def compute_mask_overlaps(masks_pre, masks_gt):
    """Computes IoU overlaps between two sets of boxes.
    masks_pre, masks_gt:
    masks_pre  [num_instances,height, width] Instance masks
    masks_gt ground truth
    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of masks_pre and masks_gt , get the nubmer of the non-zero element
    area1 = NonZero(masks_pre)
    area2 = NonZero(masks_gt)

    # Compute overlaps to generate matrix [masks count, masks_gt count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((masks_pre.shape[0], masks_gt.shape[0]))
    for i in range(overlaps.shape[1]):
        mask_gt = masks_gt[i]
        overlaps[:, i] = compute_mask_iou(mask_gt, masks_pre, area2[i], area1)

    return overlaps


def voc_ap(rec, prec, use_07_metric=False):
    """
    Compute VOC AP given precision and recall. If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    Args:
        rec: recall
        prec: precision
        use_07_metric:
    Returns:
        ap: average precision
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def convert2evalformat(inst_id_map, id_to_convert=None):
    """
    param:
        inst_id_map:[h, w]
        id_to_convert: a set
    return:
        masks:[instances,h, w]
    """
    masks = []

    inst_ids = np.unique(inst_id_map)
    background_ind = np.where(inst_ids == 0)[0]
    inst_ids = np.delete(inst_ids, background_ind)

    if id_to_convert == None:
        for i in inst_ids:
            im_mask = (inst_id_map == i).astype(np.uint8)
            masks.append(im_mask)
    else:
        for i in inst_ids:
            if i not in id_to_convert:
                continue
            im_mask = (inst_id_map == i).astype(np.uint8)
            masks.append(im_mask)

    return masks, len(masks)


def compute_class_ap(image_id_list, class_id, iou_threshold, CLASSES, charac_id, PREDICT_DIR_png, INST_PART_GT_DIR_png, INST_PART_GT_DIR_txt):
    """Compute Average Precision at a set IoU threshold (default 0.5).
    Input:
    image_id_list : all pictures id list
    gt_masks:all mask  [N_pictures,num_inst,H,W]
    pre_masks:all predict masks [N_pictures,num_inst,H,W]
    pred_scores:scores for every predicted mask [N_pre_mask]
    pred_class_ids: the indices of all predicted masks

    Returns:
    AP: Average Precision of specific class
    """
    PREDICT_DIR_txt = PREDICT_DIR_png
    iou_thre_num = len(iou_threshold)
    ap = np.zeros((iou_thre_num,))

    gt_mask_num = 0
    pre_mask_num = 0
    tp = []
    fp = []
    scores = []
    for i in range(iou_thre_num):
        tp.append([])
        fp.append([])

    print("process class", CLASSES[class_id], class_id)

    for num_img, image_id in enumerate(image_id_list):
        if (num_img % 100) == 0:
            print(num_img)
        # if image_id == '0035915':
        #     print('pause')

        pre_img = Image.open(os.path.join(PREDICT_DIR_png, '%s.png' % image_id))
        pre_img = np.array(pre_img)

        rfp = open(os.path.join(PREDICT_DIR_txt, '%s.txt' % image_id), 'r')
        items = [x.strip().split(' ') for x in rfp.readlines()]
        items = [[np.float32(x[0]), np.float32(x[1])] for x in items]
        items = [[np.uint(x[0]), np.float32(x[1])] for x in items]
        rfp.close()


        inst_part_gt = Image.open(os.path.join(INST_PART_GT_DIR_png, '%s.png' % image_id))
        inst_part_gt = np.array(inst_part_gt)

        inst_part_gt = cv2.resize(inst_part_gt, (pre_img.shape[1], pre_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        rfp = open(os.path.join(INST_PART_GT_DIR_txt, '%s.txt' % image_id), 'r')
        gt_part_id = []
        for line in rfp.readlines():
            line = line.strip().split('\t')
            if int(line[charac_id]) >= len(CLASSES):
                continue
            else:
                gt_part_id.append([int(line[0]), int(line[charac_id])])
        rfp.close()

        pre_id = []
        pre_scores = []
        for i in range(len(items)):
            if int(items[i][0]) == class_id:
                pre_id.append(i + 1)
                pre_scores.append(float(items[i][1]))

        gt_id = []
        for i in range(len(gt_part_id)):
            if gt_part_id[i][1] == class_id:
                gt_id.append(gt_part_id[i][0])

        gt_mask, n_gt_inst = convert2evalformat(inst_part_gt, set(gt_id))
        pre_mask, n_pre_inst = convert2evalformat(pre_img, set(pre_id))

        gt_mask_num += n_gt_inst
        pre_mask_num += n_pre_inst

        if n_pre_inst == 0:
            continue

        scores += pre_scores
        #############
        if n_pre_inst != len(pre_scores):
            print("************************* 1 ",image_id)
        #############


        if n_gt_inst == 0:
            for i in range(n_pre_inst):
                for k in range(iou_thre_num):
                    fp[k].append(1)
                    tp[k].append(0)
            continue

        gt_mask = np.stack(gt_mask)
        pre_mask = np.stack(pre_mask)
        # Compute IoU overlaps [pred_masks, gt_makss]
        overlaps = compute_mask_overlaps(pre_mask, gt_mask)


        max_overlap_ind = np.argmax(overlaps, axis=1)

        #############
        if len(max_overlap_ind) != len(pre_scores):
            print("************************* 2 ", image_id)
        #############

        for i in np.arange(len(max_overlap_ind)):
            max_iou = overlaps[i][max_overlap_ind[i]]
            for k in range(iou_thre_num):
                if max_iou > iou_threshold[k]:
                    tp[k].append(1)
                    fp[k].append(0)
                else:
                    tp[k].append(0)
                    fp[k].append(1)

    ind = np.argsort(scores)[::-1]

    for k in range(iou_thre_num):
        m_tp = tp[k]
        m_fp = fp[k]
        m_tp = np.array(m_tp)
        m_fp = np.array(m_fp)

        m_tp = m_tp[ind]
        m_fp = m_fp[ind]

        m_tp = np.cumsum(m_tp)
        m_fp = np.cumsum(m_fp)
        recall = m_tp / float(gt_mask_num)
        precition = m_tp / np.maximum(m_fp + m_tp, np.finfo(np.float64).eps)

        # Compute mean AP over recall range
        ap[k] = voc_ap(recall, precition, False)

    return ap

def main(args):
    characteristic_type = args.characteristic_type
    INST_PART_GT_DIR_png = args.INST_PART_GT_DIR_png
    INST_PART_GT_DIR_txt = args.INST_PART_GT_DIR_txt
    PREDICT_DIR_png = args.PREDICT_DIR_png

    if characteristic_type == "size":
        charac_id = 3
        CLASSES = ['background', 'small', 'big', 'undetermined', 'sparse']
    elif characteristic_type == "pattern":
        charac_id = 4
        CLASSES = ['background', 'solid', 'geometrical', 'fancy', 'letters']
    elif characteristic_type == "color":
        charac_id = 5
        CLASSES = ['background', 'dark', 'medium', 'light', 'brown', 'red','pink', 'yellow', 'orange', 'green', 'blue', 'purple', 'multicolor']


    IOU_THRE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print("result of", PREDICT_DIR_png)

    image_id_list = [x[:-4] for x in os.listdir(PREDICT_DIR_png) if x[-3:] == 'png']
    # image_id_list = image_id_list[:500]

    AP = np.zeros((len(CLASSES) - 1, len(IOU_THRE)))
    for ind in range(1, len(CLASSES)):
        AP[ind - 1, :] = compute_class_ap(image_id_list, ind, IOU_THRE, CLASSES, charac_id, PREDICT_DIR_png, INST_PART_GT_DIR_png, INST_PART_GT_DIR_txt)

    print(PREDICT_DIR_png)
    print("-----------------AP-----------------")
    print(AP)
    print("-------------------------------------")
    mAP = np.mean(AP, axis=0)
    print("-----------------mAP-----------------")
    print(mAP)
    print(np.mean(mAP))
    print("-------------------------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('evaluate characteristic instance part (Average precision) script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
