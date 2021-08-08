import os
import cv2
import ast
import time
import numpy as np
import pandas as pd
from copy import deepcopy


RECTANGLE_THICKNESS = 2
RECTANGLE_COLOR = (0, 0, 0, 0)
WAITING_TIME_BETWEEN_FRAMES = 0.2


def show_tagged_differences_in_frames(images_directory_path: str, csv_paths_to_compare: list):
    """
    Show tagged differences in frames.
    :param csv_paths_to_compare: csv files paths to compare.
    :param images_directory_path: Images directory path.
    """

    detections_dataframes = [pd.read_csv(csv_path) for csv_path in csv_paths_to_compare]

    for frame_index, filename in enumerate(os.listdir(images_directory_path)):
        image = cv2.imread(os.path.join(images_directory_path, filename))
        tags = [dataframe.loc[dataframe["frame_id"] == frame_index] for dataframe in detections_dataframes]
        tagged_images_list = [get_tagged_image(image_detections, image) for image_detections in tags]

        img_concatenated = np.concatenate(tagged_images_list, axis=1)

        cv2.imshow('Tagged frames', img_concatenated)
        cv2.waitKey(delay=1)

        time.sleep(WAITING_TIME_BETWEEN_FRAMES)


def get_tagged_image(tags: pd.DataFrame, image: np.ndarray) -> np.ndarray:
    """
    Get tagged image.
    :param tags: Tags for the image. (middleX, middleY, distanceY, distanceX)
    :param image: Image.
    :return: Tagged image.
    """

    tagged_image = deepcopy(image)

    if len(tags["detections"].values):
        for detections in ast.literal_eval(tags["detections"].values[0]).values():
            for detection in detections:
                (cv2.rectangle(tagged_image,
                               (int(detection[0] - (detection[3] / 2)), int(detection[1] - (detection[2] / 2))),
                               (int(detection[0] + (detection[3] / 2)), int(detection[1] + (detection[2] / 2))),
                               RECTANGLE_COLOR,
                               RECTANGLE_THICKNESS))

    return tagged_image
