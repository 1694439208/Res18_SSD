#coding=utf-8
"""
生成XML格式的label文件
"""
from pycocotools.coco import COCO
import numpy as np
import json
import os
from xml.etree.ElementTree import Element, SubElement, ElementTree

datadir = "/mnt/mfs3/lizhilong/data/coco"
datatype = "val2014"
annfile = "{}/annotations/instances_{}.json".format(datadir, datatype)

coco = COCO(annfile)

needs = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
catIds = coco.getCatIds(catNms=needs)
labelDict = dict(zip(catIds, needs))

imageIds = []

for catId in catIds:
    imgIds = coco.getImgIds(catIds=catId)
    imageIds.extend(imgIds)

imagesIds = list(set(imageIds))


def write2xml(img, ann):
    annotation = Element("annotation")
    folder = SubElement(annotation, "folder")
    filename = SubElement(annotation, "filename")
    size = SubElement(annotation, "size")
    h = img[0]["height"]
    w = img[0]["width"]
    d = 3
    folder.text = "train2014"
    filename.text = img[0]["file_name"]
    H = SubElement(size, "height")
    W = SubElement(size, "width")
    D = SubElement(size, "depth")
    H.text = str(h)
    W.text = str(w)
    D.text = str(d)
    for p in range(len(ann)):
        object = SubElement(annotation, "object")
        name = SubElement(object, "name")
        name.text = labelDict[ann[p]["category_id"]]
        bndbox = SubElement(object, "bndbox")
        xmin = SubElement(bndbox, "xmin")
        xmin.text = str(int(ann[p]["bbox"][0]))
        ymin = SubElement(bndbox, "ymin")
        ymin.text = str(int(ann[p]["bbox"][1]))
        xmax = SubElement(bndbox, "xmax")
        xmax.text = str(int(ann[p]["bbox"][0] + ann[p]["bbox"][2]))
        ymax = SubElement(bndbox, "ymax")
        ymax.text = str(int(ann[p]["bbox"][1] + ann[p]["bbox"][3]))
    tree = ElementTree(annotation)
    NAME = img[0]["file_name"].split(".")[0]
    tree.write("val_xml_annotations/{}.xml".format(NAME))
    return NAME, h, w


with open("val_name_size.txt", "w") as v:
    for Id in imagesIds:
        annIds = coco.getAnnIds(imgIds=Id, catIds=catIds, iscrowd=False)
        ann = coco.loadAnns(ids=annIds)
        img = coco.loadImgs(ids=Id)
        result = write2xml(img=img, ann=ann)
        v.write("%s %s %s\n" % result)
