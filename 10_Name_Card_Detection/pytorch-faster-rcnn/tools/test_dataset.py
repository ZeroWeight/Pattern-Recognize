import _init_paths

from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse
from PIL import Image
from PIL import ImageDraw
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
import sys
import torch
from xml.etree import ElementTree as ET
from test import img_test

net = resnetv1(num_layers=101)
net.create_architecture(4, tag='default', anchor_scales=[8, 16, 32])

net.load_state_dict(torch.load(os.path.join('../output', 'res101', 'NameCardtrainvalNameCardReal', 'default',
                              'res101_faster_rcnn_iter_200000.pth'), map_location=lambda storage, loc: storage))
net.eval()
if not torch.cuda.is_available():
    net._device = 'cpu'
net.to(net._device)

def IoU(box1,box2):
    cross_box = [
        max(box1[0],box2[0]),
        max(box1[1],box2[1]),
        min(box1[2],box2[2]),
        min(box1[3],box2[3])
    ]
    Sbox1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    Sbox2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    SboxCross = max((cross_box[2] - cross_box[0]),0) * max((cross_box[3] - cross_box[1]),0)
    return SboxCross / (Sbox1 + Sbox2 - SboxCross)

def parse_xml(xml_file):
    per = ET.parse(xml_file)
    p = per.findall('./object')

    chn_boxes = []
    num_boxes = []
    eng_boxes = []
    for obj in p:
        xmin = int(obj.findall('./bndbox/xmin')[0].text)
        ymin = int(obj.findall('./bndbox/ymin')[0].text)
        xmax = int(obj.findall('./bndbox/xmax')[0].text)
        ymax = int(obj.findall('./bndbox/ymax')[0].text)
        box = [xmin,ymin,xmax,ymax]
        cls = obj.findall('./name')[0].text
        if cls == 'Chinese':
            chn_boxes.append(box)
        elif cls == 'English':
            eng_boxes.append(box)
        elif cls == 'Number':
            num_boxes.append(box)
    return [chn_boxes,eng_boxes,num_boxes]

iou_sum = np.zeros(3)
acc = np.zeros(3)
mAP_sum = np.zeros(3)
F1_sum = np.zeros(3)
mAP_count = np.zeros(3)

if __name__ == '__main__':
  if sys.argv[1] == 'train':
    txt_file = 'trainval.txt'
  elif sys.argv[1] == 'test':
    txt_file = 'test.txt'
  else:
    txt_file = 'error'

with open(os.path.join('..','data','NameCard','NameCardReal','ImageSets','Main',txt_file)) as f:
    for idx_with_n in f:
        idx = idx_with_n[:-1]
        local_acc = np.zeros(3)
        local_exist = np.zeros(3)
        local_report = np.zeros(3)

        xml_file = os.path.join('..','data','NameCard','NameCardReal','Annotations',(idx + '.xml'))
        jpg_file = os.path.join('..','data','NameCard','NameCardReal','JPEGImages',(idx + '.jpg'))
        gt = parse_xml(xml_file)
        boxes = img_test(net,jpg_file)

        for cls in range(3):
            local_exist[cls] += len(gt[cls])
            local_report[cls] += len(boxes[cls])
            for box in boxes[cls]:
                for gt_box in gt[cls]:
                    score = IoU(box,gt_box)
                    if score > 0.6:
                        local_acc[cls] += 1
                        acc[cls] += 1
                        iou_sum[cls] += score
                        break
        for temp_idx in range(3):
            if local_exist[temp_idx] != 0:
                mAP_count[temp_idx] += 1
                mAP_sum[temp_idx] += local_acc[temp_idx] / local_exist[temp_idx]
                if local_acc[temp_idx] != 0:
                    F1_sum[temp_idx] += 2 * local_acc[temp_idx] \
                    / (local_report[temp_idx] + local_exist[temp_idx])
                
total_iou = 100 * np.sum(iou_sum) / np.sum(acc)
iou = 100 * iou_sum / acc
print ('---------- Average IoU ----------')
print (' Chinese: {:.2f}%, English: {:.2f}%, Number: {:.2f}%, Total: {:.2f}%'.format(
    iou[0], iou[1], iou[2], total_iou))

mAP = mAP_sum / mAP_count * 100
print ('-------------------- mean Average Precision  --------------------')
print (' Chinese: {:.2f}%, English: {:.2f}%, Number: {:.2f}%, Total: {:.2f}%'.format(
    mAP[0], mAP[1], mAP[2], np.mean(mAP)))

F1 = F1_sum / mAP_count * 100
print ('---------- mean F-1 score  ----------')
print (' Chinese: {:.2f}%, English: {:.2f}%, Number: {:.2f}%, Total: {:.2f}%'.format(
    F1[0], F1[1], F1[2], np.mean(F1)))



