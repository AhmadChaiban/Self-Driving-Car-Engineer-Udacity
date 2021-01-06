import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import cv2
from styx_msgs.msg import TrafficLight
import time 

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def most_common(lst):
    return max(set(lst), key=lst.count)

class TLClassifier(object):
    def __init__(self):
        # load classifier
        self.num_classes = 3
        self.detection_graph = self.load_model()
        self.img_size = (12, 8)
        self.label_map = label_map_util.load_labelmap('/home/student/Desktop/CarND-Capstone-master/ros/src/tl_detector/light_classification/label_map.pbtxt')
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.num_classes, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)
        self.current_light = TrafficLight.UNKNOWN

    def class_translation(self, light_detections, light_boxes):
        
        new_idx = []
        for i in range(len(light_boxes[0])):
            if any([num!=0.0 for num in light_boxes[0][i]]):
                new_idx.append(i)

        new_detections = [light_detections[i] for i in new_idx]

        print(new_detections)

        if 3 in new_detections:
            self.current_light = TrafficLight.RED
        elif 2 in new_detections:
            self.current_light = TrafficLight.YELLOW
        else:
            self.current_light = TrafficLight.GREEN


    def get_classification(self, image):
        start_time = time.time()
        """Det::ermines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:::
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # implement light color prediction
        image_np = load_image_into_numpy_array(image)
        output_dict = self.infer_single_image(image_np, self.detection_graph)

        light_detections = output_dict['detection_classes']
        light_boxes = output_dict['detection_boxes']

        self.class_translation(light_detections, light_boxes)

        print(time.time() - start_time)

        return self.current_light

    def load_model(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile('/home/student/Desktop/CarND-Capstone-master/ros/src/tl_detector/light_classification/exported-models/frozen_inference_graph.pb', 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph
    
    def infer_single_image(self, image, graph):
        with graph.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {
                    output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in ['detection_boxes', 'detection_classes']:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,
                                    feed_dict={image_tensor: np.expand_dims(image, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
        return output_dict


def load_image_into_numpy_array(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # (im_width, im_height, _) = image_rgb.shape
    # image_np = np.expand_dims(image_rgb, axis=0)
    return image_rgb

if __name__ == '__main__':
    classifier = TLClassifier()

    image_path = './images/left0018.jpg'

    image = Image.open(image_path)
    image_np = load_image_into_numpy_array(image)
    output = classifier.infer_single_image(image_np, classifier.detection_graph)

    print(output['detection_classes'])
    print(output['detection_boxes'])
    print(len(output['detection_boxes']))
