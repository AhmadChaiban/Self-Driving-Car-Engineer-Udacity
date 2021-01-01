import os

# Model
model_name = 'ssd_mobilenet_v2_coco_2018_03_29'
model_path = './models/my_ssd_mobilenet_v2/'
pipeline_file = 'pipeline.config'

# Set Label Map (.pbtxt) path and pipeline.config path
label_map_pbtxt_fname = './annotations/label_map.pbtxt'
pipeline_fname = model_path + pipeline_file

# Set .record path
test_record_fname = '/annotations/test.record'
train_record_fname = '/annotations/train.record'

# Set output directories and clean up
model_dir = './training/'
output_dir = './exported-models/'

os.system(f"rm -rf {model_dir} {output_dir}")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)