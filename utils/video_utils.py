import time
import pandas as pd
import numpy as np
import cv2
import os
from copy import deepcopy
import ast

RECTANGLE_COLOR = (0, 0, 0, 0)
RECTANGLE_THICKNESS = 2
WAITING_TIME_BETWEEN_FRAMES = 0.2


def show_tagged_differences_in_frames(images_directory_path: str, tagged_csv_path: str, model_detections_csv_path: str):
    """
    Show tagged differences in frames.
    :param images_directory_path: Images directory path.
    :param tagged_csv_path: Tagged csv path.
    :param model_detections_csv_path: Model detections csv path.
    """

    metadata_arrays = pd.read_csv(tagged_csv_path)
    metadata_arrays_new = pd.read_csv(model_detections_csv_path)

    for frame_index, filename in enumerate(os.listdir(images_directory_path)):
        image = cv2.imread(os.path.join(images_directory_path, filename))
        tagged_image = metadata_arrays.loc[metadata_arrays["frame_id"] == frame_index]
        model_tag_image = metadata_arrays_new.loc[metadata_arrays_new["frame_id"] == frame_index]

        tagged_img = get_tagged_image(tagged_image, image)
        model_img = get_tagged_image(model_tag_image, image)

        img_concatenated = np.concatenate([tagged_img, model_img], axis=1)

        cv2.imshow('Tagged frames', img_concatenated)
        cv2.waitKey(delay=1)

        time.sleep(WAITING_TIME_BETWEEN_FRAMES)


def get_tagged_image(tags, image):
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
