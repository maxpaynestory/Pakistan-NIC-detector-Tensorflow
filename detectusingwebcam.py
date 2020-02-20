import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

from collections import defaultdict
import PIL.Image as Image
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import time


def current_milli_time(): return int(round(time.time() * 1000))


cap = cv2.VideoCapture(0)  # Change only if you have more than one webcams

PATH_TO_LABELS = 'labelmap.pbtxt'

NUM_CLASSES = 1

PATH_TO_CKPT = os.path.join('pknic_trained_model',
                            'exported_model', 'frozen_inference_graph.pb')

loop = True


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while loop:

            # Read frame from camera
            ret, image_np = cap.read()

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Extract image tensor
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Extract detection boxes
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Extract detection scores
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            # Extract detection classes
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            # Extract number of detectionsd
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            threshold = 0.994  # in order to get higher percentages you need to lower this number; usually at 0.01 you get 100% predicted objects

            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)

            # Visualization of the results of a detection.
            returned_image = vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                boxes,
                np.squeeze(classes).astype(np.int32),
                scores,
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8, min_score_thresh=threshold)

            cv2.imshow("Pakistan NIC Detector Tensorflow", returned_image)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                loop = False
                break
