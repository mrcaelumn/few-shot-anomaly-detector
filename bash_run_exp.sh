#! /bin/bash

listDataset=(
    # "mura_sobelx_ori" "mura_sobelx_ori_v2" "mura_sobelx_8" "mura_sobelx_8_v2" "mura_sobelx_16" "mura_sobelx_16_v2" "mura_sobelx_32" "mura_sobelx_32_v2" "mura_sobelx_64" "mura_sobelx_64_v2"
    "mura_sobely_ori" "mura_sobely_ori_v2" "mura_sobely_8" "mura_sobely_8_v2" "mura_sobely_16" "mura_sobely_16_v2" "mura_sobely_32" "mura_sobely_32_v2" "mura_sobely_64" "mura_sobely_64_v2"
    "mura_sobelxy_ori" "mura_sobelxy_ori_v2" "mura_sobelxy_8" "mura_sobelxy_8_v2" "mura_sobelxy_16" "mura_sobelxy_16_v2" "mura_sobelxy_32" "mura_sobelxy_32_v2" "mura_sobelxy_64" "mura_sobelxy_64_v2"
    )

listShots=(5 10 15 20)
listNoDataset=(0 1 2 3 4)
for t in ${listDataset[@]}; do
    echo "Start Program $t"
#     run programming
    python3 few-shot-train-seresnet50.py -dn $t -s 20 -nd 0 > output_$t.log
    # python3 few-shot-train-seresnet50.py --DATASET_NAME $t 
    # echo "echo 3 >  /proc/sys/vm/drop_caches"
    sleep 60
    echo "Oops! I fell asleep for a couple seconds!"
done