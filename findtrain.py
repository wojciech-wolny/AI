#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import sys
from threading import Thread

import cv2 as cv
import numpy as np
import tensorflow as tf

sys.path.append("..")

from object_detection.utils import (label_map_util  # todo: dependency to custom_model_object_detection official model
                                    )

MODEL_NAME = 'uic_graph'  # todo: use the folder with the frozen inference graph
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')  # todo: use the label map file

NUM_CLASSES = 2  # todo change to number of classes

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def detect_objects(image_np):
    with detection_graph.as_default():
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]

        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        with tf.Session(graph=detection_graph) as session:
            # Actual detection.
            boxes, scores, classes, num_detections = session.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

    result = []
    for i in range(len(scores)):
        for j in range(len(scores[i])):
            if scores[i][j] > 0.9:

                result.append(classes[i][j])

    return {
        'boxes': tf.squeeze(boxes),
        'scores': scores,
        'classes': result,
        'num_detections': num_detections,
        'image': image_np
    }


def gen_files_paths(*args, train_number, side='left', state):
    for frame in range(*args):
        path = os.path.join(PATH_TO_TEST_IMAGES_DIR, f'{train_number}_{side}_{frame}.jpg')
        state[frame] = {'path': path}
        yield path, frame


def perform_first(image_path, state, frame):
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    objects = detect_objects(image)

    state[frame].update({'edge': 1 in objects.get('classes')})
    state[frame].update({'uic': 2 in objects.get('classes')})
    state[frame].update({'boxes': objects.get('boxes')})


def perform_second(image_path, state, frame):
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    objects = detect_objects(image)

    print(objects.get('num_detections'))


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


def use_first_network(state):
    threads = []
    for image_path, frame in gen_files_paths(10, 20, train_number='0_64', state=state, side='left'):
        process = Thread(target=perform_first, args=(image_path, state, frame))
        process.start()
        threads.append(process)

    for thread in threads:
        thread.join()


def use_second_network(state):
    threads = []
    for frame, values in state.items():
        if not isinstance(values, dict):
            continue
        process = Thread(target=perform_second, args=(values['path'], state, frame))
        process.start()
        threads.append(process)

    for thread in threads:
        thread.join()


def count_wagon(state):
    for key in state:
        state[key]['wagon'] = -1
    keys = list(state.keys())
    keys.sort()

    radius = 2
    index = 3

    previous_check = False

    while index + radius < len(keys):
        sub_key = keys[index - radius: index + radius]
        check = any([state[key].get('edge', False) for key in sub_key])

        if check:
            if not previous_check:

                for key in keys[index:]:
                    state[key]['wagon'] += 1
            else:
                tmp = index - radius

                for key in keys[tmp:]:
                    state[key]['wagon'] += 1
                for key in keys[tmp - radius:tmp - 1]:
                    state[key]['wagon'] -= 1

            previous_check = True
        else:

            previous_check = False
        index += radius * 2 + 1


PATH_TO_TEST_IMAGES_DIR = '0_64_left'
# TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, f'0_28_left_{id}.jpg'.format(i)) for i in range(49, 53)]

threads = []

state = {}

use_first_network(state)
print(state)
count_wagon(state)

MODEL_NAME1 = 'uic_graph'  # todo: use the folder with the frozen inference graph
PATH_TO_CKPT1 = MODEL_NAME1 + '/frozen_inference_graph.pb'
PATH_TO_LABELS1 = os.path.join('training1', 'object-detection.pbtxt')  # todo: use the label map file

NUM_CLASSES1 = 10  # todo change to number of classes

# Loading label map
label_map1 = label_map_util.load_labelmap(PATH_TO_LABELS1)
categories1 = label_map_util.convert_label_map_to_categories(label_map1, max_num_classes=NUM_CLASSES,
                                                             use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT1, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# use_second_network(state)
# print(state)

def export_CSV(state):
    import csv

    with open('names.csv', 'w', newline='') as csvfile:
        fieldnames = ['team_name', 'train_number', 'left_right', 'frame_number', 'wagon', 'uic_0_1', 'uic_label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for key in state:
            tmp = {}
            tmp['team_name'] = 'JanuszeNeuronuw'
            tmp['frame_number'] = key
            tmp['left_right'] = 'left'
            tmp['train_number'] = '0_64'
            tmp['uic_0_1'] = 1 if state[key]['uic'] else 0
            tmp['uic_label'] = 'xxx'
            tmp['wagon'] = 'locomotive' if state[key]['wagon'] == 0 else state[key]['wagon']

            writer.writerow(tmp)


export_CSV(state)
