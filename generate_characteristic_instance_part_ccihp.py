# Inspired from https://github.com/Engineering-Course/CIHP_PGN/blob/master/instance_tool/generate_instance_part.m

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

import numpy as np
#import scipy.io
import os
import glob
#import pickle
from PIL import Image
import cv2
import json
#from matplotlib.pyplot import imsave
#from matplotlib import colors
import copy
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Set directories of inputs and outputs and some variables', add_help=False)

    parser.add_argument('--instance_folder', default='',
                        help='output path where to save, empty for no saving. ex: /path/where/to/find/pred_cihp_size_inst_part_maps')
    parser.add_argument('--human_folder', default='',
                        help='directory where to find human instance predictions (segmentation masks in PNG format) from inference. ex: /path/where/to/find/pred_inst/')
    parser.add_argument('--parsing_folder', default='',
                        help='directory where to find attribute predictions (segmentation masks in PNG format) from inference. ex: /path/where/to/find/pred_segm/')
    parser.add_argument('--charac_folder', default='',
                        help='directory where to find characteristic (size, pattern or color) predictions (segmentation masks in PNG format) from inference. ex: /path/where/to/find/pred_size/')
    parser.add_argument('--score_folder', default='',
                        help='directory where to find score predictions from inference in json format. ex: /path/where/to/find/pred_json/')
    parser.add_argument('--filelist', default='',
                        help='file with the image names. ex: /path/where/to/find/validation/dataset/val_id.txt')
    parser.add_argument('--characteristic_type', default='size',
                        help='it can be size, pattern or color')
    parser.add_argument('--class_num', default=21, type=int, help='number of attribute classes +1')
    parser.add_argument('--class_charac', default=5, type=int, help='number of characteristic (size, pattern or color) classes +1')
    return parser


#mat = scipy.io.loadmat('pascal_seg_colormap.mat')
#colormap = mat['colormap']
#cmap = colors.ListedColormap(colormap)


def switch_labels(image, switch_dict):
    switch_image = copy.deepcopy(image)
    for x in range(len(switch_image)):
        switch_image[x] = [switch_dict[y] for y in switch_image[x]]
    return switch_image


def main(args):
    filelist = args.filelist
    instance_folder = args.instance_folder
    score_folder = args.score_folder
    human_folder = args.human_folder
    parsing_folder = args.parsing_folder
    charac_folder = args.charac_folder

    class_num = args.class_num
    class_charac = args.class_charac

    if args.characteristic_type == "size":
        json_id = -3
        json_key = "semantic_size_score"
    elif args.characteristic_type == "pattern":
        json_id = -2
        json_key = "semantic_pattern_score"
    elif args.characteristic_type == "color":
        json_id = -1
        json_key = "semantic_color_score"

    parsing_images_list = glob.glob(parsing_folder + '*.png')
    parsing_images_list.sort()
    charac_images_list = glob.glob(charac_folder + '*.png')
    charac_images_list.sort()
    instance_images_list = glob.glob(human_folder + '*.png')
    instance_images_list.sort()
    json_files = glob.glob(score_folder + '*.json')
    json_files.sort()


    print("start generation")
    for charac_image, parsing_image, instance_image, json_map, fname in zip(charac_images_list, parsing_images_list, instance_images_list, json_files,
                                                              filelist):

        parsing_image_pil = Image.open(parsing_image)
        parsing_map = np.array(parsing_image_pil)
        charac_image_pil = Image.open(charac_image)
        charac_map = np.array(charac_image_pil)
        instance_image_pil = Image.open(instance_image)
        human_map = np.array(instance_image_pil)

        with open(json_map, 'rb') as jf:
            data = json.load(jf)
            charac_score_map = np.asarray(data[json_id][json_key][0])


        instance_map = np.zeros(human_map.shape)

        print(fname)

        with open(instance_folder + '/' + fname + '.txt', 'w') as f_out:
            counter = 0
            scores = []
            for k in range(1, class_num):
                indices = (parsing_map == k)
                indices = np.uint(indices)
                human_map = np.float32(human_map)
                part_map = np.uint8(np.multiply(indices, human_map))

                for char in range(1, class_charac):
                    cur_counter = counter
                    indices_char = (charac_map == char)
                    indices_char = np.uint(indices_char)
                    part_map = np.float32(part_map)
                    charac_part_map = np.uint8(np.multiply(indices_char, part_map))

                    label_id, count_pix = np.unique(charac_part_map, return_counts=True)

                    for l in range(len(label_id)):
                        if count_pix[l] <= 100:
                            continue
                        if label_id[l] != 0:
                            counter = counter + 1
                            indices = np.where(charac_part_map == label_id[l])
                            instance_map[indices] = counter
                            parsing_score = np.mean(charac_score_map[indices])
                            scores.append(len(indices) * (parsing_score))
                    if cur_counter < counter:
                        for o in range(cur_counter, counter):
                            f_out.write(str(char) + ' ' + str(scores[o]) + '\n')


            cv2.imwrite(f'{instance_folder}/{fname}.png', instance_map)

        f_out.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('generate characteristic instance part script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.instance_folder:
        Path(args.instance_folder).mkdir(parents=True, exist_ok=True)
    main(args)


