import cv2
import os
import numpy as np
import csv


def find_coordinates(folder_path, *target_colors):

    """Returns a CSV file containing the coordinates corresponding to the colors' position. Each row corresponds to
    the position of one individual object, and a blank row means that it's the coordinates of the next image.
    If multiple target colors are entered, then the contours will treat those overlapping colors as one
    object. Please enter BGR values for the colors"""

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
            masks = [cv2.inRange(image, array-10, array+10) for array in colors]

            if len(masks) > 1:
                for index in range(len(masks)):
                    if index == 0:
                        mask = cv2.bitwise_or(masks[0], masks[1])

                    elif index < len(masks)-1:
                        mask = cv2.bitwise_or(mask, masks[index+1])

            else:
                mask = masks[0]

            binary_image = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

            contours = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            ind_contours = [np.array(contour) for contour in contours]

            coordinates = []  # The format of the coordinates will be (xi, xf, yi, yf)
            for contour in ind_contours:
                cx = [row[0][0] for row in contour]
                cy = [row[0][1] for row in contour]
                coordinate = (min(cx), max(cx), min(cy), max(cy))
                coordinates.append(coordinate)

            directory = os.path.basename(input_folder)
            with open(rf"{input_folder} Head Coordinates/{directory} Coordinates.csv", "a", newline="") as file:
                obj = csv.writer(file)

                if img_name == os.listdir(input_folder)[0]:
                    obj.writerow(["xi", "xf", "yi", "yf"])

                for row in coordinates:
                    obj.writerow(row)
                obj.writerow([])

            if img_name == os.listdir(input_folder)[0]:
                copy = image.copy()
                for coordinate in coordinates:
                    # cv2 takes ranges of pixels in the format of [yi:yf, xi:xf]
                    region = copy[coordinate[2]:coordinate[3], coordinate[0]:coordinate[1]]

                    blur = cv2.GaussianBlur(region, (31, 31), 0, 0)
                    copy[coordinate[2]:coordinate[3], coordinate[0]:coordinate[1]] = blur

                edit = cv2.hconcat([image, copy])

                if edit.shape[1] > 1000:
                    new_height = int(edit.shape[0]/edit.shape[1] * 1000)
                    edit = cv2.resize(edit, (new_height, 1000))
                cv2.imshow("First Edit", edit)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        except TypeError:
            errors += 1

    plural_noun = {True: "", False: "s"}[errors < 1]
    plural_verb = {True: "was", False: "were"}[errors < 1]
    print(f"Done. {errors} image{plural_noun} {plural_verb} unsuccessfully processed.")
