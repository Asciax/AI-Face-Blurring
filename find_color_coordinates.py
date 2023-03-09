import cv2
import os
import numpy as np
import csv


def find_coordinates(folder_path: str, margin: int, minimum: int, *target_colors: list):

    """Returns a CSV file containing the coordinates corresponding to the colors' position. Each row corresponds to
    the position of one individual object, and a blank row means that it's the coordinates of the next image.
    If multiple target colors are entered, then the contours will treat those overlapping colors as one
    object. Please enter BGR values for the colors. Margin is the variation in BGR value allowed. Minimum is the
    smallest allowed size of a face in pixels"""

    errors = 0

    # Directory
    input_folder = rf"{folder_path}"
    os.makedirs(rf"{input_folder} Head Coordinates", exist_ok=True)
    #

    # Colors
    colors = list(map(np.array, target_colors))
    #

    # Finding Contours of all Images
    for img_name in os.listdir(input_folder):

        try:
            filepath = os.path.join(input_folder, img_name)
            image = cv2.imread(filepath)

            # Turn the Image into a Binary Image Based on the Target Colors
            masks = [cv2.inRange(image, array-margin, array+margin) for array in colors]

            if len(masks) > 1:
                for index in range(len(masks)):
                    if index == 0:
                        mask = cv2.bitwise_or(masks[0], masks[1])

                    elif index < len(masks)-1:
                        mask = cv2.bitwise_or(mask, masks[index+1])

            else:
                mask = masks[0]

            # Turn the Image into a Binary Image. Also smooths out noises and small spots.
            binary_image = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)[1]

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
            binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

            contours = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            ind_contours = [np.array(contour) for contour in contours]

            coordinates = []  # The format of the coordinates will be (xi, xf, yi, yf)
            for contour in ind_contours:
                cx = [row[0][0] for row in contour]
                cy = [row[0][1] for row in contour]
                coordinate = (min(cx), max(cx), min(cy), max(cy))
                if (coordinate[1] - coordinate[0])*(coordinate[3]-coordinate[2]) >= minimum:
                    coordinates.append(coordinate)

            directory = os.path.basename(input_folder)
            with open(rf"{input_folder} Head Coordinates/{directory} Coordinates.csv", "a", newline="") as file:
                obj = csv.writer(file)

                if img_name == os.listdir(input_folder)[0]:
                    obj.writerow(["xi", "xf", "yi", "yf"])

                for row in coordinates:
                    obj.writerow(row)
                obj.writerow([])

            if img_name in os.listdir(input_folder)[0:4]:

                copy1 = image.copy()
                copy2 = image.copy()
                copy3 = image.copy()
                copy3[:, :] = [0, 0, 0]

                for coordinate in coordinates:

                    # cv2 takes ranges of pixels in the format of [yi:yf, xi:xf]
                    region = copy1[coordinate[2]:coordinate[3], coordinate[0]:coordinate[1]]

                    blur = cv2.GaussianBlur(region, (31, 31), 0, 0)
                    copy1[coordinate[2]:coordinate[3], coordinate[0]:coordinate[1]] = blur
                    copy2[coordinate[2]:coordinate[3], coordinate[0]:coordinate[1]] = [255, 255, 255]
                    copy3[coordinate[2]:coordinate[3], coordinate[0]:coordinate[1]] = [255, 255, 255]

                edit = cv2.hconcat([image, copy1, copy2, copy3])

                if edit.shape[1] > 2000:
                    new_height = int(edit.shape[0]/edit.shape[1] * 2000)
                    edit = cv2.resize(edit, (new_height, 2000))

                cv2.imshow("First Edit", edit)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        except ValueError:
            errors += 1

    plural_noun = {True: "", False: "s"}[errors < 1]
    plural_verb = {True: "was", False: "were"}[errors < 1]
    print(f"{len(os.listdir(input_folder)) - errors} image{plural_noun} processed. "
          f"{errors} image{plural_noun} {plural_verb} unsuccessfully processed.")


face_color = [196, 196, 196]
hat_color = [16, 16, 16]
hair_color = [31, 31, 31]
glasses_color = [61, 61, 61]
margin_of_error = 0
minimum_head_size = 40
find_coordinates(########################################################, margin_of_error, minimum_head_size,
                 face_color, glasses_color)
