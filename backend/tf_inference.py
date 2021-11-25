#import tensorflow as tf
import numpy as np
import pandas as pd
import os
import shutil
import json
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import time as t
from PIL import Image
import re
from ensemble_boxes import *
import tensorflow as tf
import matplotlib.pyplot as plt

root_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))


def test_after_train_generator():
    image_files = []
    test_after_training_path = os.path.join(root_path,"darknet/data/test_after_training")
    for filename in os.listdir(test_after_training_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_files.append("data/test_after_training/" + filename)
    #os.chdir("..")
    out_path = os.path.join(root_path,"darknet/data/test_after_training.txt")
    with open(out_path, "w") as outfile:
        for image in image_files:
            outfile.write(image)
            outfile.write("\n")
        outfile.close()
    #os.chdir("..")

def predict_yolov4(cla,upload_dir, pred_dir):
  '''
  @cla: a string, indicate which class to detect
  @upload_dir: directory where the image is saved 
  @pred_dir: directory to save the predictions to 
  '''
  # copy cfg, obj.data, obj.names, weights to the assigned folders
  
  # delete old files
  cfg = os.path.join(root_path,"darknet/cfg/yolov4-obj.cfg") 
  obj_data = os.path.join(root_path,"darknet/data/obj.data")
  obj_names = os.path.join(root_path,"darknet/data/obj.names")
  weights = os.path.join(root_path,"darknet/data/yolov4-obj_best.weights")

  if os.path.exists(obj_data):
    os.remove(obj_data)
  if os.path.exists(obj_names):
    os.remove(obj_names)
  if os.path.exists(cfg):
    os.remove(cfg)
  if os.path.exists(weights):
    os.remove(weights)
  
  # copy new files
  cfg_to = os.path.join(root_path,"darknet/cfg")
  cfg_from = os.path.join(root_path,f"models/modeling_{cla}/things_put_in_darknet/cfg/yolov4-obj.cfg") 
  shutil.copy(cfg_from, cfg_to)
  
  obj_data_to = os.path.join(root_path,"darknet/data")
  obj_data_from = os.path.join(root_path,f"models/modeling_{cla}/things_put_in_darknet/data/obj.data")
  shutil.copy(obj_data_from, obj_data_to)

  obj_names_to = os.path.join(root_path,"darknet/data")
  obj_names_from = os.path.join(root_path,f"models/modeling_{cla}/things_put_in_darknet/data/obj.names")
  shutil.copy(obj_names_from, obj_names_to)
  
  weights_to = os.path.join(root_path,"darknet/data")
  weights_from = os.path.join(root_path,f"models/modeling_{cla}/backup/yolov4-obj_best.weights")
  shutil.copy(weights_from, weights_to)  
  
  predictions = pd.DataFrame(columns=['image_name', 'object_class', 'x1', 'y1', 'x2', 'y2', 'confidence','model_name','left_x','top_y','width','height'])
  
  for f1 in os.listdir(upload_dir):
    test_after_train_path = os.path.join(root_path,"darknet/data/test_after_training.txt") 
    if os.path.exists(test_after_train_path):
        os.remove(test_after_train_path)
    for f2 in os.listdir(pred_dir): 
        if os.path.exists(os.path.join(pred_dir, f2)):
            os.remove(os.path.join(pred_dir, f2)) # delete old predicted images
    if f1.endswith('.png') or f1.endswith('.jpg'):
      file_path = os.path.join(upload_dir,f1)
      shutil.copy(file_path, pred_dir) # copy images needing prediction to the test_after_training folder

      # generate a txt file for the predicted image which will be used in the detection model
      test_after_train_generator() 

      # predict
      if cla != 'livestock':
        cmd = f'{os.path.join(root_path,"darknet")} detector test {os.path.join(root_path,"darknet/data/obj.data")} {os.path.join(root_path,"darknet/cfg/yolov4-obj.cfg")} {os.path.join(root_path,"darknet/data/yolov4-obj_best.weights")} -dont_show -ext_output < {os.path.join(root_path,"darknet/data/test_after_training.txt")} > {os.path.join(root_path,"darknet/pred_baseline.txt")} -thresh 0.5'
        os.system(cmd)
      else:
        cmd = f'{os.path.join(root_path,"darknet")} detector test {os.path.join(root_path,"darknet/data/obj.data")} {os.path.join(root_path,"darknet/cfg/yolov4-obj.cfg")} {os.path.join(root_path,"darknet/data/yolov4-obj_best.weights")} -dont_show -ext_output < {os.path.join(root_path,"darknet/data/test_after_training.txt")} > {os.path.join(root_path,"darknet/pred_baseline.txt")} -thresh 0.25'
        os.system(cmd)
        
 
  return predictions

def run_yolo_inference(upload_dir, pred_dir):

    '''
    run the inference for yolo (above)
    @upload_dir: directory where the image is saved 
    @pred_dir: directory to save the predictions to 
    '''
    building = predict_yolov4('building',upload_dir, pred_dir)
    tree = predict_yolov4('tree',upload_dir, pred_dir)
    vehicle = predict_yolov4('vehicle',upload_dir, pred_dir)
    livestock = predict_yolov4('livestock',upload_dir, pred_dir)

    combined =  pd.concat([tree, building, vehicle, livestock], ignore_index=True)
    combined = combined.drop(columns=['left_x','top_y','width','height'])
    combined = combined.sort_values(['image_name', 'object_class'], ignore_index=True)

    x1 = [0 if x <0 else x for x in combined.x1.values]
    combined['x1'] = x1
    y1 = [0 if x <0 else x for x in combined.y1.values]
    combined['y1'] = y1
    x2 = [1 if x >1 else x for x in combined.x2.values]
    combined['x2'] = x2
    y2 = [1 if x >1 else x for x in combined.y2.values]
    combined['y2'] = y2

    combined = combined.dropna()

    return combined

def load_faster_rcnn_models(vehicle_model,tree_model,building_model):
  '''
  vehicle_model: path to saved vehicle model
  tree_model: path to saved tree model
  building_model: path to saved building model
  '''
  saved_models = {
    'vehicle' : tf.saved_model.load(vehicle_model),
    'tree' : tf.saved_model.load(tree_model),
    'building' : tf.saved_model.load(building_model)}

  return saved_models

def get_model_results_faster_rcnn(image_path,saved_models,filename):
  '''
  image_path: the path to where the image is saved
  saved_models: dictionary of saved models in format class_name: tensorflow detection function 
  filename: the name of the file uploaded by the user 
  '''
  # load the image and convert it to a numpy array and then an input tensor 
  image_np = np.array(Image.open(image_path))
  input_tensor = tf.convert_to_tensor(image_np)
  input_tensor = input_tensor[tf.newaxis, ...]

  image_name = [] 
  object_class = [] 
  x1s = [] 
  y1s = []  
  x2s = []
  y2s = []  
  scores = [] 
  model_name = []

  for class_type, model in saved_models.items(): 
    detect_fn = model 
    class_val = class_type 

    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
      for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    class_values = [class_val] * num_detections
    image_names = [filename] * num_detections
    model_names = ['faster_rcnn'] * num_detections

    image_name.extend(image_names)
    object_class.extend(class_values)
    model_name.extend(model_names)
    scores.extend(detections['detection_scores'])

    boxes_np = np.array(detections['detection_boxes'])
    ymin = boxes_np[:,0]
    xmin = boxes_np[:,1]
    ymax = boxes_np[:,2]
    xmax = boxes_np[:,3]

    x1s.extend(xmin) 
    y1s.extend(ymin)  
    x2s.extend(xmax)
    y2s.extend(ymax)


  data = {
  'image_name': image_name,
  'object_class' : object_class,
  'x1' : x1s,
  'y1' : y1s, 
  'x2' : x2s,
  'y2' : y2s,
  'confidence': scores,
  'model_name' : model_name}
  
  df = pd.DataFrame(data)

  return df 

def combine_results(df_yolov4,df_faster_rcnn): 
  '''
  combine the yolov4 and faster rcnn detection results into a single dataframe
  '''
  return pd.concat([df_yolov4,df_faster_rcnn])


def generate_lists(df,class_mapping_dict):
  '''
  helper function for the ensemble results function 
  generate the boxes_list, scores_list, labels_list required to ensemble the results together
  for a single image and class
  df is filtered to a single class and image 
  '''
  boxes_list = []
  scores_list = []
  labels_list = []

  models = set(df.model_name.values)

  for model in models:
    annots_m = df[df['model_name']==model]
    num_detections = len(annots_m)

    scores = list(annots_m.confidence.values)
    labels = list(map(lambda x: class_mapping_dict[x], annots_m.object_class.values))
    scores_list.append(scores)
    labels_list.append(labels)
    
    model_boxes = []

    x1s = annots_m.x1.values
    x2s = annots_m.x2.values
    y1s = annots_m.y1.values
    y2s = annots_m.y2.values
      
    for i in range(0,num_detections):
      # x1, y1, x2, y2
      box = [x1s[i], y1s[i], x2s[i], y2s[i]]
      model_boxes.append(box)
      
    boxes_list.append(model_boxes)

  return boxes_list, scores_list, labels_list


def load_json_file(path):
  f = open(path,)
  json_file = json.load(f)
  f.close()
  return json_file


def ensemble_results(detection_results,class_name_mapping):
  '''
  detection_results_location: df containing the combined detection results 
  class_name_map: json file with class name mappings
  ''' 
  
  df = detection_results 

  image_names = set(df.image_name.values)
  classes = set(df.object_class.values) 

  '''
  ensemble methods 
  nms: 
  boxes, scores, labels = nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr)

  soft_nms: 
  boxes, scores, labels = soft_nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)

  nmw: 
  boxes, scores, labels = non_maximum_weighted(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

  wbf:
  boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
  '''

  # ensembling parameters 
  weights_dict = {'building': [1,1],
  'tree': [1,1],
  'vehicle': [1,1]}

  iou_thresh = {'building':0.45,
  'tree':0.25,
  'vehicle':0.5}

  skip_box_thresh = {'building':0.3,
  'tree':0.3,
  'vehicle':0.3}

  ens_methods = {'building': 'wbf',
  'tree': 'wbf',
  'vehicle': 'wbf'}

  image_name = []
  object_class = [] 
  x1 = [] 
  y1 = []
  x2 = [] 
  y2 = []
  confidence = [] 
  model_name = []

  for image in image_names:
    
    for object_type in classes:

      # if livestock predictions then append directly to results and do not ensemble 
      if object_type == 'livestock':
        livestock = df[df['object_class']=='livestock']
        image_name.extend(livestock.image_name.values)
        object_class.extend(livestock.object_class.values)
        x1.extend(livestock.x1.values)
        y1.extend(livestock.y1.values)
        x2.extend(livestock.x2.values)
        y2.extend(livestock.y2.values)
        confidence.extend(livestock.confidence.values)
        model_name.extend(livestock.model_name.values)

      # all other object types  
      else:
        annots = df[(df['image_name']==image)&(df['object_class']==object_type)]
        models = set(annots.model_name.values) 

        boxes_list, scores_list, labels_list = generate_lists(annots,class_name_mapping)

        
        if len(boxes_list)>=1:
          
          # if more than 1 model generated the results then apply a weight
          if len(boxes_list)>1:
          # look up the weighting from the dictionary 
            weights = weights_dict[object_type]
          
          # if just 1 model generated the results then do not apply a weight  
          elif len(boxes_list)==1:
            weights = None 
        
          iou_thr = iou_thresh[object_type]
          skip_box_thr = skip_box_thresh[object_type]
          
          # results are in alphabetical order of models: ['faster_rcnn','yolov4']
          method = ens_methods[object_type]
          
          # get the ensembled deetection results 
          if method == 'nms':
            boxes, scores, labels = nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr)
          elif method == 'soft_nms':
            boxes, scores, labels = soft_nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, thresh=skip_box_thr)
          elif method == 'nmw':
            boxes, scores, labels = non_maximum_weighted(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
          elif method == 'wbf':
            boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)  

          # write the ensembled data to the lists 
          # this data is in the format: x1, y1, x2, y2
          boxes_np = np.array(boxes)
          x1s = boxes_np[:,0]
          y1s = boxes_np[:,1]
          x2s = boxes_np[:,2]
          y2s = boxes_np[:,3]
          x1.extend(x1s)
          y1.extend(y1s)
          x2.extend(x2s)
          y2.extend(y2s)
          confidence.extend(scores)

          # reverse the dictionary so we can map from the int class ids to class names 
          class_int_name_mapping = {value : key for (key, value) in class_name_mapping.items()}
          labels_str = list(map(lambda x: class_int_name_mapping[x], labels))

          num_detections = len(scores)
          image_name_list = [image] * num_detections
          image_name.extend(image_name_list)
          object_class.extend(labels_str)
          model_name_list = ['ensembled'] * num_detections 
          model_name.extend(model_name_list)
  

  # the ensembled data should have this format: 
  ensembled_data = {
      'image_name' : image_name,
      'object_class' : object_class,
      'x1' : x1,
      'y1' : y1,
      'x2' : x2,
      'y2' : y2,
      'confidence' : confidence,
      'model_name' : model_name} 

  ensembled = pd.DataFrame(ensembled_data)

  return ensembled 


def visualize_detection_results(ensembled_df,out_file_location,label_path,data_upload_location,class_name_map,filename): 
  '''
  visualize the ensembled detection results 
  ensembled_df: df containing the ensebled data in format: 'image_name', 'object_class', 'x1', 'y1', 'x2', 'y2', 'confidence', 'model_name' 
  out_file_location: file location to save the result
  label_path: file location for the pbtxt file with the int to str labels 
  data_upload_location: directory where the input images that the user uploads are stored 
  filename: the name of the image file
  '''

  class_name_mapping = class_name_map
  category_index = label_map_util.create_category_index_from_labelmap(label_path)
  detection_results = ensembled_df

  # only 1 image uploaded to each directory 
  #image = os.listdir(data_upload_location)[0]
 
  image_path = os.path.join(data_upload_location,filename)
  image_np = np.array(Image.open(image_path))
  input_tensor = tf.convert_to_tensor(image_np)
  input_tensor = input_tensor[tf.newaxis, ...]

  # load the annotations 
  detections_img = detection_results[detection_results['image_name']==filename]

  # tf bounding boxes are in the format [y_min, x_min, y_max, x_max]
  detection_boxes = np.array(detections_img[['y1','x1','y2','x2']])
  num_detections = len(detections_img)
  detection_scores = detections_img.confidence.values
  detection_classes = list(map(lambda x: class_name_mapping[x], detections_img.object_class.values))

  image_np_with_detections = image_np.copy()

  viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detection_boxes,
            detection_classes,
            detection_scores,
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=0,
            agnostic_mode=False)
    
  plt.figure(figsize=(24, 16))
  plt.imshow(image_np_with_detections)
  img_name = os.path.join(out_file_location,filename)
  plt.imsave(img_name, image_np_with_detections)



def generate_object_count_summary(ensembled_data): 
  '''
  helper function to generate the counts of objects found in the 2 images 
  ensembled_data: df containing the ensembled detection data 
  image_1_location: directory where image 1 is saved 
  image_2_location: directory where image 2 is saved 
  '''
  detections = ensembled_data
  
  object_types = detections.object_class.values
  full_image_names = detections.image_name.values 

  all_objects_count = {} 

  for i in range(0,len(full_image_names)): 
    key_val = full_image_names[i]
    object_name = object_types[i]
    if key_val in all_objects_count.keys(): 
      # values are in format: tree, vehicle, livestock, building 
      object_count_list = all_objects_count[key_val]
      if object_name == 'tree': 
        object_count_list[0] += 1 
      elif object_name == 'vehicle': 
        object_count_list[1] += 1 
      elif object_name == 'livestock': 
        object_count_list[2] += 1 
      elif object_name == 'building': 
        object_count_list[3] += 1 
    else: 
      if object_name == 'tree': 
        object_count_list = [1,0,0,0]
      elif object_name == 'vehicle':
        object_count_list = [0,1,0,0]
      elif object_name == 'livestock':
        object_count_list = [0,0,1,0]
      elif object_name == 'building':
        object_count_list = [0,0,0,1] 
      all_objects_count[key_val] = object_count_list
    
  full_image_sum = [] 
  tree_count_sum = [] 
  livestock_count_sum = [] 
  building_count_sum = [] 
  vehicle_count_sum = []

  for i in range(0,len(all_objects_count)): 
    dict_key = list(all_objects_count.keys())
    key_value = dict_key[i]
    full_image = key_value
    object_count_list = all_objects_count[key_value]
    tree_count = object_count_list[0]
    vehicle_count = object_count_list[1]
    livestock_count = object_count_list[2]
    building_count = object_count_list[3]

    full_image_sum.append(full_image)
    tree_count_sum.append(tree_count)
    livestock_count_sum.append(livestock_count)
    building_count_sum.append(building_count)
    vehicle_count_sum.append(vehicle_count)

  object_count_summary = {
    'image_name' : full_image_sum, 
    'tree_count' :  tree_count_sum,
    'vehicle_count' : vehicle_count_sum,
    'livestock_count' : livestock_count_sum,
    'building_count' : building_count_sum}

  detection_summary = pd.DataFrame(object_count_summary)

  return detection_summary


def get_changes_detected(ensembled_data,image_1,image_2): 
  '''
  function to get the changes detected for between user uploaded images 1 and 2 
  ensembled_data: ensembled detection data df
  image_1: filename for image 1  
  image_2: filename for image 2
  '''
  # get the object counts 
  df = generate_object_count_summary(ensembled_data)
  
  # get the names of image 1 and image 2 from directory (1 file in each location)
  # image_1 = os.listdir(image_1_location)[0]
  # image_2 = os.listdir(image_2_location)[0]


  df_image_1 = df[df['image_name']==image_1]
  df_image_2 = df[df['image_name']==image_2]

  data = {'image_1' : [df_image_1.image_name.values[0]],
        'image_2' : [df_image_2.image_name.values[0]],
        'tree_count_1' : [df_image_1.tree_count.values[0]], 
        'tree_count_2' :  [df_image_2.tree_count.values[0]],
        'tree_change' : [df_image_2.tree_count.values[0] - df_image_1.tree_count.values[0]],
        'vehicle_count_1' : [df_image_1.vehicle_count.values[0]],
        'vehicle_count_2' : [df_image_2.vehicle_count.values[0]],
        'vehicle_change' : [df_image_2.vehicle_count.values[0] - df_image_1.vehicle_count.values[0]],
        'livestock_count_1' : [df_image_1.livestock_count.values[0]],
        'livestock_count_2' : [df_image_2.livestock_count.values[0]],
        'livestock_change' : [df_image_2.livestock_count.values[0] - df_image_1.livestock_count.values[0]],
        'building_count_1' : [df_image_1.building_count.values[0]],
        'building_count_2' : [df_image_2.building_count.values[0]],
        'building_change' : [df_image_2.building_count.values[0] - df_image_1.building_count.values[0]]}
  
  changes_detected = pd.DataFrame(data)
  # out_file_path = out_file_location + '/' + 'changes_detected.csv'
  # changes_detected.to_csv(out_file_path)

  return changes_detected

# method 5: visualize the change detection results in a png image 

def visualize_object_count_changes(changes_detected,image_1_name,image_2_name,out_file_location):
  '''
  visualized the changes detected and save a png file 
  changes_detected: the df containing the changes detected  
  image_1_name: the name of image 1 
  image_2_name: the name of image 2 
  file_location: the directory to save the result image to 
  '''
  # get the counts for image 1 and image 2 and the chages 
  object_counts = changes_detected
  tree_count_1 = object_counts.tree_count_1.values[0]
  tree_count_2 = object_counts.tree_count_2.values[0]
  tree_change = object_counts.tree_change.values[0]
  vehicle_count_1 = object_counts.vehicle_count_1.values[0]
  vehicle_count_2 = object_counts.vehicle_count_2.values[0]
  vehicle_change = object_counts.vehicle_change.values[0]
  livestock_count_1 = object_counts.livestock_count_1.values[0]
  livestock_count_2 = object_counts.livestock_count_2.values[0]
  livestock_change = object_counts.livestock_change.values[0]
  building_count_1 = object_counts.building_count_1.values[0]
  building_count_2 = object_counts.building_count_2.values[0]
  building_change = object_counts.building_change.values[0]

  #image_1_name = os.listdir(image_1_location)[0]
  image_1_path = out_file_location + '/image_1/' + image_1_name
  #image_2_name = os.listdir(image_2_location)[0]
  image_2_path = out_file_location + '/image_2/' + image_2_name

  image_1_title = image_1_name + '\n' + f'Trees: {tree_count_1}' + '\n' + f'Buildings: {building_count_1}' + '\n' + f'Vehicles: {vehicle_count_1}' + '\n' + f'Livestock: {livestock_count_1}'
  image_2_title = image_2_name + '\n' + f'Trees: {tree_count_2}' + '\n' + f'Buildings: {building_count_2}' + '\n' + f'Vehicles: {vehicle_count_2}' + '\n' + f'Livestock: {livestock_count_2}'

  comparison_title = 'Object Count Changes Detected:\n'

  if tree_change == 0: 
    comparison_title += 'Trees: 0\n'
  elif tree_change <0: 
    comparison_title += f'Trees: - {abs(tree_change)}\n'
  else: 
    comparison_title += f'Trees: + {tree_change}\n'
  
  if building_change == 0: 
      comparison_title += 'Buildings: 0\n'
  elif tree_change <0: 
    comparison_title += f'Buildings: - {abs(building_change)}\n'
  else: 
    comparison_title += f'Buildings: + {building_change}\n'

  if vehicle_change == 0: 
    comparison_title += 'Vehicles: 0\n'
  elif vehicle_change <0: 
    comparison_title += f'Vehicles: - {abs(vehicle_change)}\n'
  else: 
    comparison_title += f'Vehicles: + {vehicle_change}\n'
  
  if livestock_change == 0: 
    comparison_title += 'Livestock: 0\n'
  elif livestock_change <0: 
    comparison_title += f'Livestock: - {abs(livestock_change)}\n'
  else: 
    comparison_title += f'Livestock: + {livestock_change}\n'
  
  my_dpi = 100
  fig = plt.figure(figsize=(1200/my_dpi, 800/my_dpi), dpi=my_dpi)
  fig.suptitle(comparison_title)
  
  ax1 = fig.add_subplot(1, 2, 1)
  ax1.set_title(image_1_title)
  ax1.set_xticks([])
  ax1.set_yticks([])
  load_image_1 = Image.open(image_1_path)
  ax1.imshow(load_image_1)

  ax2 = fig.add_subplot(1, 2, 2)
  ax2.set_title(image_2_title)
  ax2.set_xticks([])
  ax2.set_yticks([])
  load_image_2 = Image.open(image_2_path)
  ax2.imshow(load_image_2)

  fig.tight_layout()
  fig.subplots_adjust(top=0.9)

  
  save_image_filepath = out_file_location + '/change_detection/' + 'result.png' 
  fig.savefig(save_image_filepath)

