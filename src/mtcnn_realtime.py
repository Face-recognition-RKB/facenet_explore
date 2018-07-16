#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

import sys
import time
import numpy as np
import tensorflow as tf
import cv2

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils_color as vis_util


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './protos/face_label_map.pbtxt'

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
cap = cv2.VideoCapture("./media/test.mp4")
out = None

while True:
  
  ret, image = cap.read()
  
  if ret == 0:
    break

  if out is None:
    [h, w] = image.shape[:2]
    out = cv2.VideoWriter("./media/test_out.avi", 0, 25.0, (w, h))

  if image.ndim == 2: 
    image = facenet.to_rgb(image)
  
  image = image[:,:,0:3]

  start_time = time.time()

  # MTCNN here
  bounding_boxes, box_cord = align.detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)

  elapsed_time = time.time() - start_time

  print('inference time cost: {}'.format(elapsed_time))

#   vis_util.visualize_boxes_and_labels_on_image_array(
# #          image_np,
#       image,
#       np.squeeze(boxes),
#       np.squeeze(23).astype(np.int32),
#       np.squeeze(23),
#       category_index,
#       use_normalized_coordinates=True,
#       line_thickness=4)
  nrof_faces = bounding_boxes.shape[0]
  for rectangle in range(0,nrof_faces):
      cv2.rectangle(image,box_cord[rectangle],(0,255,0),5)
  out.write(image)


  cap.release()
  out.release()
