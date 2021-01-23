DATA_FOLDER=/home/xuewyang/Xuewen/Research/data/COCO
OUTPUT_FOLDER=/home/xuewyang/Xuewen/Research/model/captioning

CUDA_VISIBLE_DEVICES='1' python test_r.py --batch_size 10 --workers 1 --n_layers 4 \
--features_path $DATA_FOLDER/detection_features_relation.hdf5 \
--annotation_folder $DATA_FOLDER/annotations --model_path $OUTPUT_FOLDER/Relational-Transformer-1/Relational-Transformer_best.pth
