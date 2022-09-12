import os
import pandas as pd
import xml.etree.ElementTree as Xet
from pathlib import Path


def split_csv_to_multiple_csv_files(csv_path_to_split: str, target_path: str, column_split_by: str):
    """
    Split csv file to multiple files by specific column.
    :param csv_path_to_split:  The path of csv file to split.
    :param target_path: Path to save into the splitted files.
    :param column_split_by: The column to split by.
    """

   
    dataframe = pd.read_csv(csv_path_to_split)
    grouped_dataframes = dict(iter(dataframe.groupby(column_split_by)))

    for grouped_column, grouped_dataframe in grouped_dataframes.items():
        file_path_to_save = f'{os.path.join(target_path, os.path.basename(grouped_column).split(".")[0])}.csv'
        grouped_dataframe.to_csv(file_path_to_save, index=False)



def convert_xml_to_csv(path: str, path_to_save_csv: str):
    """
    Convert xml file to csv file.
    :param path: Path of xml file.
    :param path_to_save_csv: Path to save csv.
    """

    video_id = f"{Path(path).parent.name}.mp4"
    rows = []

    for index, xml_file in enumerate(os.listdir(path)):
        xml_parse = Xet.parse(os.path.join(path, xml_file))
        root = xml_parse.getroot()
        rows.append(extract_data_from_xml_root(root, video_id, index))

    pd.DataFrame(rows).to_csv(path_to_save_csv, index=False)


def extract_data_from_xml_root(root: Xet, video_id: str, frame_number: int) -> dict:
    """
    Extract data from xml root.
    :param root: The xml root.
    :param video_id: Video id.
    :param frame_number: Frame number.
    :return: The extracted data in csv format.
    """

    detections = {}

    for section in root.findall('object'):
        class_name = section.find('class').text

        if not section.find('class').text in detections.keys():
            detections[class_name] = []

        bndbox = section.find("bndbox")
        detections[class_name].append(extract_coordinates_from_xml_root(bndbox))

    return {"video_id": video_id,
            "frame_id": str(frame_number),
            "homography": [],
            "detections": detections}


def extract_coordinates_from_xml_root(root: Xet) -> list:
    """
    Extract coordinates from xml root.
    :param root: The xml root.
    :return: The extracted coordinates in csv format.
    """

    xmin = int(root.find('xmin').text)
    xmax = int(root.find('xmax').text)
    ymin = int(root.find('ymin').text)
    ymax = int(root.find('ymax').text)

    middleX, distanceX = get_middle_and_distance(xmin, xmax)
    middleY, distanceY = get_middle_and_distance(ymin, ymax)

    return [middleX,
            middleY,
            distanceY,
            distanceX, None, None, None, None, None]


def get_middle_and_distance(x: int, y: int) -> tuple:
    middle = (x + y) / 2
    distance = abs(x) - abs(y)
    return middle, distance
