import time
import pandas as pd
import numpy as np
import cv2
import os
import ast
from copy import deepcopy


RECTANGLE_COLOR = (0, 0, 0, 0)
RECTANGLE_THICKNESS = 2
WAITING_TIME_BETWEEN_FRAMES = 0.2


def show_tagged_differences_in_frames(images_directory_path: str, tagged_csv_paths_to_compare: list):
    """
    Show tagged differences in frames.
    :param tagged_csv_paths_to_compare:
    :param images_directory_path: Images directory path.
    """

    metadata_arrays = [pd.read_csv(tagged_csv_path) for tagged_csv_path in tagged_csv_paths_to_compare]

    for frame_index, filename in enumerate(os.listdir(images_directory_path)):
        image = cv2.imread(os.path.join(images_directory_path, filename))
        tags = [metadata_array.loc[metadata_array["frame_id"] == frame_index]
                for metadata_array in metadata_arrays]
        tagged_images_list = [get_tagged_image(image_detections, image) for image_detections in tags]

        img_concatenated = np.concatenate(tagged_images_list, axis=1)

        cv2.imshow('Tagged frames', img_concatenated)
        cv2.waitKey(delay=1)

        time.sleep(WAITING_TIME_BETWEEN_FRAMES)


def get_tagged_image(tags: pd.DataFrame, image: np.ndarray):
    """
    Get tagged image.
    :param tags: Tags for the image.
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
