#!/bin/bash
#git clone https://github.com/ducha-aiki/wxbs-descriptors-benchmark.git
#chmod +x wxbs-descriptors-benchmark/data/download_W1BS_dataset.sh
#./wxbs-descriptors-benchmark/data/download_W1BS_dataset.sh
#mv W1BS wxbs-descriptors-benchmark/data/
python HardNet_provenance_test.py --test-set=MSCOCO_synthesized --alpha=0.0 --beta=0.0 --loss_type=0 --start-epoch=2 
#python HardNet.py --fliprot=True --experiment-name=/liberty_train_with_aug/  | tee -a log_HardNetPlus_Lib.log
