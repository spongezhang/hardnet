#!/bin/bash
#git clone https://github.com/ducha-aiki/wxbs-descriptors-benchmark.git
#chmod +x wxbs-descriptors-benchmark/data/download_W1BS_dataset.sh
#./wxbs-descriptors-benchmark/data/download_W1BS_dataset.sh
#mv W1BS wxbs-descriptors-benchmark/data/
#python triplet.py | tee ./log/log_hardNet.log
python triplet.py --Loss_Type=1| tee ./log/log_hardNet_gor.log
#python triplet.py --Loss_Type=2| tee ./log/log_triplet.log
alpha=(0.01 0.1 1.0 2.0 5.0 10.0 100.0)

for margin_0 in "${margin[@]}"
do
    echo "Decay step $margin_0"
    python patch_network_train_new.py --training notredame --test liberty --decay_step $margin_0| tee ../log/decay_step/$margin_0.txt
done
