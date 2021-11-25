import os
from flask import Flask, request, render_template, send_from_directory
from PIL import Image
import numpy as np
import base64
import io
import time

__author__ = 'ibininja'

from backend.tf_inference import *

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload_single.html")


@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'files')
    if not os.path.isdir(target):
        os.mkdir(target)

    # load faster-rcnn models 
    vehicle_path = os.path.join(APP_ROOT,'models/modeling_vehicle/faster_rcnn_model/saved_model')
    tree_path = os.path.join(APP_ROOT,'models/modeling_tree/faster_rcnn_model/saved_model')
    building_path = os.path.join(APP_ROOT,'models/modeling_building/faster_rcnn_model/saved_model')
    saved_models = load_faster_rcnn_models(vehicle_path,tree_path,building_path)

    # load the class name mapping file 
    class_name_mapping = load_json_file(os.path.join(APP_ROOT,'ref_files/class_name_mapping.json'))
    label_path = os.path.join(APP_ROOT,'ref_files/labelmap.pbtxt') 

    # single file upload 
    upload = request.files.getlist("file")[0]
    filename = upload.filename
        
    # This is to verify files are supported
    ext = os.path.splitext(filename)[1]
    if (ext == ".jpg") or (ext == ".png") or (ext == ".jpeg"):
        print("File supported moving on...")
    else:
        render_template("Error.html", message="Files uploaded are not supported...")
    
    destination = os.path.join(target, 'object_detection','image',filename)
    # print("Accept incoming file:", filename)
    # print("Save it to:", destination)

    # delete existing image(s)
    files = os.listdir(os.path.join(target, 'object_detection','image'))
    if len(files) > 0:
        for f in files:
            os.remove(os.path.join(target, 'object_detection','image',f)) 
    # upload new image 
    upload.save(destination)
        

    # do object detection 
    upload_dir = os.path.join(target, 'object_detection','image')
    pred_dir = os.path.join(APP_ROOT,'darknet/data/test_after_training')

    # do inference yolov4 
    yolo_results = run_yolo_inference(upload_dir, pred_dir)
    # do inference faster_rcnn 
    faster_rcnn_results = get_model_results_faster_rcnn(destination,saved_models,filename)
    # combine yolov4 and faster_rcnn results together 
    combined_results = combine_results(yolo_results,faster_rcnn_results)
    # ensemble faster_rcnn and yolov4 results together 
    ensembled_df = ensemble_results(combined_results,class_name_mapping)

    # save the visualization of the detectected results 
    out_file_location = os.path.join(APP_ROOT,'files/object_detection/result')

    # remove any old images 
    files = os.listdir(out_file_location)
    if len(files) > 0:
        for f in files: 
            os.remove(os.path.join(out_file_location,f)) 

    # generate the visualized change detection results         
    visualize_detection_results(ensembled_df,out_file_location,label_path,upload_dir,class_name_mapping,filename)

    # return send_from_directory("images", filename, as_attachment=True)
    return render_template("complete_display_image.html", image_name=filename)


@app.route('/upload_single/<filename>')
def send_image(filename):
    return send_from_directory("files/object_detection/result/",filename)


if __name__ == "__main__":
    app.run(port=4555, debug=True)
