#coding=utf-8
"""
load cocodataset annotations
"""
from pycocotools.coco import COCO
import numpy as np
import json
import os

datadir = "/mnt/mfs3/lizhilong/data/coco"
datatype = "train2014"
annfile = "{}/annotations/instances_{}.json".format(datadir, datatype)

coco = COCO(annfile)

needs = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
catIds = coco.getCatIds(catNms=needs)
labelDict = dict(zip(needs, catIds))

imageIds = []

for catId in catIds:
    imgIds = coco.getImgIds(catIds=catId)
    imageIds.extend(imgIds)

imagesIds = set(imageIds)
images = coco.loadImgs(ids=imageIds)

# creat name_size.txt
with open("train_name_size.txt", "w") as n:
    for Id in imagesIds:
        annIds = coco.getAnnIds(imgIds=Id, catIds=catIds, iscrowd=False)
        ann = coco.loadAnns(ids=annIds)
        img = coco.loadImgs(ids=Id)

        file_name = img[0]["file_name"]
        width = img[0]["width"]
        height = img[0]["height"]

        img[0]["filename"] = file_name
        for i in [
                "file_name", "license", "coco_url", "date_captured",
                "flickr_url", "id"
        ]:
            img[0].pop(i)
        new_data = {}
        new_data["image"] = img[0]

        for i in range(len(ann)):
            for j in ["segmentation", "area", "image_id", "id"]:
                ann[i].pop(j)
        new_data["annotation"] = ann
        NAME = file_name.split(".")[0]
        # create per json file
        with open("annotations/{}.json".format(NAME), "w") as j:
            json.dump(new_data, j, indent=3)

        n.write("{} {} {}\n".format(NAME, str(height), str(width)))
