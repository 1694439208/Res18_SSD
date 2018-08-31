root_dir="/home/zhongchong/repos/caffe-ssd"


cd $root_dir

data_root_dir="/mnt/mfs3/zhongchong/coco"
mapfile="/mnt/mfs3/zhongchong/coco/labelmap.prototxt"
anno_type="detection"
label_type="xml"
db="lmdb"
min_dim=512
max_dim=512
width=512
height=512

extra_cmd="--encode-type=jpg --encoded"

python $root_dir/scripts/create_annoset.py --anno-type=$anno_type --label-type=$label_type \
                                           --label-map-file=$mapfile --min-dim=512 --max-dim=512 --resize-width=512 --resize-height=512 \
                                           --check-label $extra_cmd $data_root_dir $data_root_dir/"train.txt" $data_root_dir/$db/"train2014_"$db $data_root_dir 2>&1 | tee $data_root_dir/train.log


