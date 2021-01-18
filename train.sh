DATA_FOLDER=/home/xuewyang/Xuewen/Research/data/COCO
OUTPUT_FOLDER=/home/xuewyang/Xuewen/Research/model/captioning

CUDA_VISIBLE_DEVICES='1' python train_r.py --exp_name 'Meshed-Memory-Transformer-1st' --batch_size 10 \
--workers 1 --m 40 --head 8 --warmup 10000 --features_path $DATA_FOLDER/detection_features_relation.hdf5 \
--annotation_folder $DATA_FOLDER/annotations --logs_folder $OUTPUT_FOLDER/tensorboard_logs \
--model_folder $OUTPUT_FOLDER/Meshed-Memory-Transformer-3st
