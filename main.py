import torch, torchvision
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# Inference with a panoptic segmentation model
im = cv2.imread("./roadSmall.jpg")
# cv2_imshow(im)
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
predictor = DefaultPredictor(cfg)
panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
# dat = predictor(im)
# print([(index, cls) for index, cls in enumerate(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).stuff_classes)])
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
cv2_imshow(out.get_image()[:, :, ::-1])
cv2.imwrite("segmentation.jpg", out.get_image()[:, :, ::-1])
print(panoptic_seg)
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
classes = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).stuff_classes

def get_id_by_name(name):
  return classes.index(name)
 
import numpy as np
from PIL import Image
 
 
def add_channel(image):
  try:
      b_channel, g_channel, r_channel = cv2.split(image)
  except:
      return image
  alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255 #creating a dummy alpha channel image.
 
  return cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
 
def mask(segmentation, target_image, effect_image, id, opacity):
  category_id = next(filter(lambda x: x['category_id'] == id, segments_info))['id']
  effect_h, effect_w, _ = effect_image.shape
  print(effect_w, effect_h)
  target_h, target_w, _ = target_image.shape
  crop_effect = effect_image[0:target_h, 0:target_w]
  crop_effect = add_channel(crop_effect)
  prediction = segmentation.cpu().numpy()
  target_prediction = np.array([
   [prediction[j][i] == category_id for i in range(len(prediction[j]))]
   for j in range(len(prediction))
  ])
  crop_effect[~target_prediction, :] = [0, 0, 0, 0]
  cv2_imshow(crop_effect)
  target_image = add_channel(target_image)
  cv2_imshow(crop_effect + target_image)
  dst = cv2.addWeighted(crop_effect, opacity, target_image, 1, 0)
  cv2_imshow(dst)
  cv2_imshow(target_image)
  cv2.imwrite("result.jpg", dst)
  cv2.imwrite("input.jpg", target_image)
  return dst

effect_image = cv2.imread("ice.jpg")
res = mask(panoptic_seg, im, effect_image, get_id_by_name("road"), 0.75) 

effect_image = cv2.imread("polar.jpg")
res = mask(panoptic_seg, res, effect_image, get_id_by_name("sky"), 1)
