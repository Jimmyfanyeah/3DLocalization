set -x
nohup python3 main.py           --device='cuda:3'  \
                                --param_file='/home/lingjia/Documents/rPSF/NN/param_v2.yaml'  \
                                --data_path_override='/media/hdd/rPSF_data/rPSF/train/0620_uniformFlux'  \
                                --img_size_override=96 \
                                > /home/lingjia/Documents/rPSF/log/decode_impl_v1.log 2>&1 &
set +x


python3 main.py           --device='cuda:2' \
                          --param_file='/home/lingjia/Documents/rPSF/NN/param_v2.yaml'\
                          --data_path_override='/media/hdd/rPSF/data/decode/decode_impl_train_plain_30k'\
                          --img_size_override=96 \
                          --log_comment='30k_nobg'


"""
DATA:
/media/hdd/rPSF/data/decode/decode_impl_train
/media/hdd/rPSF/data/plain/train/0620_uniformFlux
/media/hdd/rPSF/data/decode/decode_impl_train_plain_20k
/media/hdd/rPSF/data/decode/decode_impl_train_plain_30k
""""