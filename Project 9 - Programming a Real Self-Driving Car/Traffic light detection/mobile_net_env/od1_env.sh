%tensorflow_version # check version

# Uninstall 2.x and install 1.14
python -m pip uninstall -y tensorflow tensorboard tensorflow-estimator tensorboard-plugin-wit
python -m pip install tensorflow-gpu==1.14.0 tensorboard==1.14.0 tensorflow-estimator==1.14.0

# Install required libraries
apt-get install -qq protobuf-compiler python-pil python-lxml python-tk
python -m pip install -q pillow lxml jupyter matplotlib cython pandas contextlib2
python -m pip install -q pycocotools tf_slim