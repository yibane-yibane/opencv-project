import os
from utils import csv_utils, video_utils


if __name__ == '__main__':
    csv_utils.split_csv_to_multiple_csv_files(os.environ.get('CSV_FILE_PATH'),
                                              os.environ.get('OUTPUT_CSV_DIR_PATH'),
                                              'video_id')
    csv_utils.convert_xml_to_csv(os.environ.get('ANNOTATIONS_PATH'),
                                 os.environ.get('MODEL_DETECTIONS_CSV_PATH'))
    video_utils.save_images_from_path_to_video_file(os.environ.get('IMAGES_DIR_PATH'),
                                                    os.environ.get('OUTPUT_CSV_DIR_PATH'))
    video_utils.show_tagged_differences_in_frames(os.environ.get('IMAGES_DIR_PATH'),
                                                  [os.environ.get('TAGGED_CSV_PATH'),
                                                   os.environ.get('MODEL_DETECTIONS_CSV_PATH')])
