# object_detection_flask
This repo is built using modified templates created by [ibrahimokdadov:](https://github.com/ibrahimokdadov). The reference repo can be found [here](https://github.com/ibrahimokdadov/upload_file_python)

## to run
1. clone object_detection_flask repo
2. clone TensorFlow model repo to project directory  (https://github.com/tensorflow/models)
2. update file path at top of requirements.txt to the file path where the TensorFlow folder is saved locally 
3. create a virtual env in project directory (python3 -m venv env)  
4. activate virtual environment (source env/bin/activate)
5. install all requirements to virtual environment (pip install -r requirements.txt)
6. to run modules:  
    - change detection module: python app_change_det.py
    - object detection module: python app_object_det.py
8. to access web interface open link 
