import yaml
import cv2
import os
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import warnings

warnings.filterwarnings('ignore')

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
 
# class that defines and loads the kangaroo dataset
class TrafficLightDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset_dir, is_train=True):
        # define one class
        self.add_class("dataset", 1, "Red")
        self.add_class("dataset", 2, "Green")
        self.add_class("dataset", 3, "Yellow")
        
        images_dir = './dataset-sdcnd-capstone/data/sim_training_data/'
        
        data = self.read_annotations()
        filenames = self.get_filenames(data)
        
        # find all images
        for index, filename in enumerate(filenames):
            image_id = filename[-8:-4]
            # skip all images after 200 if we are building the train set
            if is_train and index >= 200:
                continue
            # skip all images before 200 if we are building the test/val set
            if not is_train and index < 200:
                continue
            img_path = images_dir + filename
            # add to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=data[index])
            
    def read_annotations(self):
        with open("./dataset-sdcnd-capstone/data/sim_training_data/sim_data_annotations.yaml", "r") as yamlfile:
            data = yaml.load(yamlfile, Loader=yaml.FullLoader)
            print("Read successful")
            print(f"Located {len(data)} annotations")
        return data
    
    def get_filenames(self, data):
        filenames = []
        for i in range(len(data)):
            filenames.append(data[i]['filename'])
        return filenames
 
    # extract bounding boxes from an annotation file
    def extract_boxes(self, node):
        # extract each bounding box
        boxes = []
        for annotation in node['annotations']:
            xmin = int(annotation['xmin'])
            ymin = int(annotation['ymin'])
            xmax = int(annotation['x_width']) + xmin
            ymax = int(annotation['y_height']) + ymin
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        # extract image dimensions
        image = cv2.imread(f'./dataset-sdcnd-capstone/data/sim_training_data/{node["filename"]}')
        width = image.shape[1]
        height = image.shape[0]
        return boxes, width, height
 
    # load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        annotation_node = info['annotation']
        # load XML
        boxes, w, h = self.extract_boxes(annotation_node)
        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            if annotation_node['annotations'][i]['class'].lower() == 'green':
                masks[row_s:row_e, col_s:col_e, i] = 1
                class_ids.append(self.class_names.index('Green'))
            elif annotation_node['annotations'][i]['class'].lower() == 'red':
                masks[row_s:row_e, col_s:col_e, i] = 2
                class_ids.append(self.class_names.index('Red'))
            elif annotation_node['annotations'][i]['class'].lower() == 'yellow':
                masks[row_s:row_e, col_s:col_e, i] = 3
                class_ids.append(self.class_names.index('Yellow'))
        return masks, asarray(class_ids, dtype='int32')
 
    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']
 
# define a configuration for the model
class TrafficConfig(Config):
    # define the name of the configuration
    NAME = "traffic_cfg"
    # number of classes (background + kangaroo)
    NUM_CLASSES = 1 + 3
    # number of training steps per epoch
    STEPS_PER_EPOCH = 131