#!/bin/bash
#git clone https://github.com/ducha-aiki/wxbs-descriptors-benchmark.git
#chmod +x wxbs-descriptors-benchmark/data/download_W1BS_dataset.sh
#./wxbs-descriptors-benchmark/data/download_W1BS_dataset.sh
#mv W1BS wxbs-descriptors-benchmark/data/
python HardNet_provenance.py --fliprot=False --n-triplets=1000000 --epochs 5 --alpha=1
#python HardNet.py --fliprot=True --experiment-name=/liberty_train_with_aug/  | tee -a log_HardNetPlus_Lib.log
#python HardNet_provenance.py --fliprot=False --experiment-name=/provenance/ --resume=./logs/provenance_1.0/checkpoint_4.pth --n-triplets=1000000 --epochs 5
