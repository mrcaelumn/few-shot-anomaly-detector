#! /bin/bash



### create folders
listFolder=("result" "samples" "saved_model" "plot_output", "text_output")

for t in ${listFolder[@]}; do
    echo "Create folder $t if not exists"
    mkdir -p $t
done



### run experiments
# listDataset=(
#     # "mura_sobelx_ori" "mura_sobelx_ori_v2" 
#     # "mura_sobelx_8" "mura_sobelx_8_v2" 
#     # "mura_sobelx_16" "mura_sobelx_16_v2" 
#     # "mura_sobelx_32" "mura_sobelx_32_v2" 
#     # "mura_sobelx_64" "mura_sobelx_64_v2"
#     # "mura_sobely_ori" "mura_sobely_ori_v2"
#     # "mura_sobely_8" "mura_sobely_8_v2"
#     "mura_sobely_16" # "mura_sobely_16_v2"
#     # "mura_sobely_32" "mura_sobely_32_v2"
#     # "mura_sobely_64" "mura_sobely_64_v2"
#     # "mura_sobelxy_ori" 
#     "mura_sobelxy_ori_v2"
#     # "mura_sobelxy_8" "mura_sobelxy_8_v2"
#     "mura_sobelxy_16" "mura_sobelxy_16_v2"
#     # "mura_sobelxy_32" "mura_sobelxy_32_v2"
#     # "mura_sobelxy_64" "mura_sobelxy_64_v2"
# )

listDataset=(
    "mura_scharr_x_8_v3" "mura_scharr_x_16_v3" "mura_scharr_x_32_v3" "mura_scharr_x_64_v3" "mura_scharr_x_ori_v3"
    "mura_scharr_y_8_v3" "mura_scharr_y_16_v3" "mura_scharr_y_32_v3" "mura_scharr_y_64_v3" "mura_scharr_y_ori_v3"
)

# listDataset=(
#     "mura_sobelx_8" "mura_sobelx_8_v2" 
#     "mura_sobelx_16" "mura_sobelx_32" 
#     "mura_sobelx_32_v2" "mura_sobelx_64" 
#     "mura_sobelx_64_v2" "mura_sobelx_ori" 
#     "mura_sobelx_ori_v2" "mura_sobelxy_8"
#     "mura_sobelxy_8_v2" "mura_sobelxy_32"
#     "mura_sobelxy_32_v2" "mura_sobelxy_64_v2"
#     "mura_sobelxy_ori"
# )

listShots=(5 10 15 20)
listNoDataset=(0 1 2 3 4)
for t in ${listDataset[@]}; do
    echo "Start Program $t"
#     run programming
    # python3 few-shot-train-seresnet50.py -dn $t -s 20 -nd 0 -bb "seresnet50" > output_$t.log
    python3 main.py -dn $t -s 20 -nd 0 -bb "seresnext50" > output_$t.log
    # echo "echo 3 >  /proc/sys/vm/drop_caches"
    sleep 60
    echo "Oops! I fell asleep for a couple seconds!"
done