#coding=utf-8


import sys
reload(sys)
sys.setdefaultencoding('utf8')
import argparse
import numpy as np
import json
import caffe
import cv2
import time
import os

class Detection:
    def __init__(self, gpu_id, model_def, model_weights, conf_thresh):
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()
        self.net = caffe.Net(model_def,
                             model_weights,
                             caffe.TEST)
        self.conf_thresh = conf_thresh

    def _transform(self, image):
        image = cv2.resize(image, (512, 512))
        image = image.transpose((2, 0, 1))
        image = image.astype('float32') - np.array([104., 117., 123]).reshape(3, 1, 1)
        return image

    def _draw_save(self, image, image_file, result):
        save_name = "detect_result/" + image_file.split('/')[-1]
        label_name = {1: "person",
                      2: "bicycle",
                      3: "car",
                      4: "motorcycle",
                      5: "bus",
                      6: "trunk"}
        height, width = image.shape[:2]
        for item in result:
            xmin = int(round(item[0] * width))
            ymin = int(round(item[1] * height))
            xmax = int(round(item[2] * width))
            ymax = int(round(item[3] * height))
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0))
            cv2.putText(image, label_name[item[-2]] +': '+str(round(item[-1], 2)), (xmin,ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            print item
            print("{} score is {}".format(label_name[item[-2]], item[-1]))
        cv2.imwrite(save_name, image)


    def detect(self, image_file):
        start = time.time()
        image = cv2.imread(image_file.encode('utf-8'))
        transformed_image = self._transform(image)
        self.net.blobs['data'].data[...] = transformed_image
        detections = self.net.forward()['detection_out']
        cost = time.time() - start
        print "load transform forward cost {:.3f}s".format(cost)

        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_yamx = detections[0,0,:,6]

        index = [i for i, conf in enumerate(det_conf) if conf >= self.conf_thresh]
        top_conf = det_conf[index]
        top_label = det_label[index]
        top_xmin = det_xmin[index]
        top_ymin = det_ymin[index]
        top_xmax = det_xmax[index]
        top_ymax = det_yamx[index]

        set_label = set(top_label)
        result = [[top_xmin[i],top_ymin[i],top_xmax[i],top_ymax[i],top_label[i],top_conf[i]] for i in xrange(top_conf.shape[0])]

        self._draw_save(image, image_file, result=result)


def main(args):
    caffessd_root = "/home/zhongchong/repos/caffe-ssd"
    detection = Detection(args.gpu_id, args.model_def, args.model_weights, args.conf_thresh)
    if os.path.isdir(args.image):
        image_list = os.listdir(args.image)
        for image_file in image_list:
            image_file = os.path.join(caffessd_root, args.image, image_file)
            detection.detect(image_file)
    if os.path.isfile(args.image):
        image_file = os.path.join(caffessd_root, args.image)
        detection.detec(image_file)
    print "result saved in caffe-ssd/detect_result"

def parse_args():
    "参数"
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=3, help='gpu id')
    parser.add_argument("--conf_thresh", type=float, default=0.5, help='threshold')
    parser.add_argument("--model_def", default="models/resnet18/coco/deploy.prototxt", help='deploy')
    parser.add_argument("--model_weights", default='models/resnet18/coco/'
                        'res18_coco_SSD_512x512_iter_160000.caffemodel')
    parser.add_argument('--image', default='test_images', help='image dir or image file')
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())

