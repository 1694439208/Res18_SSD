cd /home/zhongchong/repos/caffe-ssd
./build/tools/caffe train \
--solver="/home/zhongchong/repos/caffe-ssd/models/resnet18/own/solver.prototxt" \
--weights="/home/zhongchong/repos/caffe-ssd/models/resnet18/resnet-18.caffemodel" \
--gpu 1 2>&1 | tee /home/zhongchong/repos/caffe-ssd/models/resnet18/own/job/res18_coco_SSD_512x512.log
