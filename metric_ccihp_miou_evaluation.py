# MIT License

# Copyright (c) 2021 Angelique Loesch

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and  to permit persons to whom the Software is
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


import json
import numpy as np
from PIL import Image
from sklearn.metrics import jaccard_score, accuracy_score
import argparse
from pathlib import Path
import glob


def get_args_parser():
    parser = argparse.ArgumentParser('ccihp metric eval', add_help=False)

    parser.add_argument('--input_dir', default='',
                        help='path where to find inference attribute/characteristics masks (segmentation masks in PNG format)')
    parser.add_argument('--GT_dir', default='',
                        help='path where to find GT attribute/characteristics masks (segmentation masks in PNG format)')
    parser.add_argument('--num_classes', default=0, type=int,
                        help='num classes semantic to evaluate (ex: ccihp attribute labels == number of classes + 1')

    return parser

def main(args):
"""
This script compute the mIoU metric for semantic segmentation (for attributes or characteristics).
The inputs correspond to the predicted attribute (or size, pattern or color) semantic masks, as well as their ground truth and the number of classes to evaluate.

"""

    mask_list = glob.glob(args.input_dir + '/attrib*')
    mask_list.sort()
    gt_list = glob.glob(args.GT_dir + '/*.png')
    gt_list.sort()

    iou_dict = {}
    ioufinal_dict = {}
    for num_i,im_path in enumerate(mask_list):
        im = Image.open(im_path)

        name_png = im_path.split('/')[-1]
        print(im_path)
        name_png = name_png.split('_')[-1]

        width, height = im.size

        im = np.array(im)
        im = im.flatten()
        
        gt = Image.open(args.GT_dir + '/' + name_png)
        gt = np.array(gt)
        gt = gt.flatten()
        gt[gt >= args.num_classes] = 0
        im[im >= args.num_classes] = 0

        IoU = jaccard_score(gt, im, average=None)
        
        mean_IoU = np.mean(IoU)#[1:])
        
        acc = accuracy_score(gt, im, normalize=True)
        

        if 'mean_iou' in iou_dict.keys():
            iou_dict['mean_iou'].append(mean_IoU)
        else:
            iou_dict['mean_iou'] = [mean_IoU]
        if 'acc' in iou_dict.keys():
            iou_dict['acc'].append(acc)
        else:
            iou_dict['acc'] = [acc]

        num_gt = np.unique(gt)
        num_lab = np.unique(im)

        for id, n in enumerate(num_lab):
            if n in iou_dict.keys():
                iou_dict[n].append(IoU[id])
            else:
                iou_dict[n] = [IoU[id]]
        for id, n in enumerate(num_gt):
            if n not in num_lab:
                if n in iou_dict.keys():
                    iou_dict[n].append(0.)
                else:
                    iou_dict[n] = [0.]

    for k in iou_dict.keys():
        ioufinal_dict['mean_' + str(k)] = [np.mean(iou_dict[k])]
    
    print(ioufinal_dict)






if __name__ == '__main__':
    parser = argparse.ArgumentParser('ccihp metric eval script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

