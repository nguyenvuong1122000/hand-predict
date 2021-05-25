import torch
import json
import os
from PIL import Image
import torch.hub
from sklearn.preprocessing import  LabelEncoder
import numpy as np
from xml.etree import ElementTree as ET
from torch.utils.data import Dataset
from utils import *
PATH = "/home/vuong/PycharmProjects/hand-predict/hand-sign/data_ver2/data"
c = 0
def read_content(xml_file):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []
    labels = []
    difficult_list = []
    filename = root.find("filename").text
    for boxes in root.iter('object'):

        filename = root.find('filename').text
        difficult = boxes.find("difficult").text

        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)
        label = boxes.find("name").text

        difficult_list.append(difficult)
        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)
        labels.append(label)

    return  filename, list_with_all_boxes, labels, difficult_list

def one_hot_encode(folder):
    X = []
    X = X + ["."]
    for filename in os.listdir(folder):
        if not filename.__contains__(".xml"):
            continue
        filename, list_with_all_boxes, labels, difficult = read_content(os.path.join(folder,filename))
        X = X + labels
        if labels == ["."]:
            print(filename)

    onehot = LabelEncoder()
    onehot.fit(np.array(X).reshape(-1,1))
    return onehot


class DataLoader(Dataset):
    def __init__(self, folder, split):
        self.annotations = []
        self.folder = os.path.join(os.path.join(folder, split))
        self.split = split
        for filename in os.listdir(os.path.join(folder, split)):
            if filename.__contains__(".xml"):
                self.annotations.append(filename)
        self.enc = one_hot_encode(self.folder)
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, item):
        annotation  = self.annotations[item]

        image, boxes, labels, difficult = read_content(os.path.join(self.folder, annotation))
        image = Image.open(os.path.join(self.folder, image))
        boxes = torch.FloatTensor(boxes)
        labels = torch.FloatTensor(self.enc.transform(np.array(labels).reshape(-1,1)))


        # if (boxes.nelement() != 4):
        #     print(annotation)
        image, boxes, labels, difficulties = transform(image, boxes, labels, difficult, split=self.split)

        return image, boxes, labels,difficulties, annotation
    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each

data = DataLoader(PATH, "train")
