#! /bin/bash



### create folders
listFolder=("result" "samples" "saved_model" "plot_output" "text_output")

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

# listDataset=(
#     "mura_scharr_x_8_v3" "mura_scharr_x_16_v3" "mura_scharr_x_32_v3" "mura_scharr_x_64_v3" "mura_scharr_x_ori_v3"
#     "mura_scharr_y_8_v3" "mura_scharr_y_16_v3" "mura_scharr_y_32_v3" "mura_scharr_y_64_v3" "mura_scharr_y_ori_v3"
# )

# listDataset=(
#    "mura_scharr_x_8_v3" "mura_sobel_xy_32_v5" "mura_sobel_xy_64_v3" "mura_sobel_xy_ori_v5" "mura_april" 
# )

listDataset=(
    "mura_sobel_xy_32_v5" "mura_sobel_xy_64_v3"
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

echo "start training process for v1 dataset"
for t in ${listDataset[@]}; do
    version=1
    echo "Start Program $t of version $version"
    res_dir="result/$t""_v$version/"
    saved_model_dir="saved_model/$t""_v$version/" 
    # echo $res_dir
    mkdir -p $res_dir
    mkdir -p $saved_model_dir
    # run programming
    python3 main.py -dn $t -s 20 -nd 0 -bb "seresnext50" -m true -rd $res_dir -ted "test_data_v1" -trd "train_data" -eld "eval_data" -smd $saved_model_dir > "output_$t""_v$version.log"
    
    sleep 5
    echo "Oops! I fell asleep for a couple seconds!"
done


# echo "start training process for v2 dataset"
# for t in ${listDataset[@]}; do
#     version=2
#     echo "Start Program $t of version $version"
#     res_dir="result/$t""_v$version/"
#     saved_model_dir="saved_model/$t""_v$version/" 
#     # echo $res_dir
#     mkdir -p $res_dir
#     mkdir -p $saved_model_dir
#     # run programming

#     # python3 few-shot-train-seresnet50.py -dn $t -s 20 -nd 0 -bb "seresnet50" > output_$t.log
#     python3 main.py -dn $t -s 20 -nd 0 -bb "seresnext50" -m true -rd $res_dir -ted "test_data_v2" -trd "train_data_v2" -eld "eval_data_v2" -smd $saved_model_dir > "output_$t""_v$version.log"
    
#     sleep 5
#     echo "Oops! I fell asleep for a couple seconds!"
# done













# echo "start testing process for v1 dataset with test data v2"
# for t in ${listDataset[@]}; do
#     version=1
#     echo "Start Program $t of version $version"
#     res_dir="result/$t""_v3/"
#     saved_model_dir="saved_model/$t""_v$version" 
#     # echo $res_dir
#     mkdir -p $res_dir
#     mkdir -p $saved_model_dir
#     # run programming

#     # python3 few-shot-train-seresnet50.py -dn $t -s 20 -nd 0 -bb "seresnet50" > output_$t.log
#     python3 main.py -dn $t -s 20 -nd 0 -bb "seresnext50" -m false -rd $res_dir -ted "test_data_v2" -trd "train_data_v1" -eld "eval_data_v1" -smd $saved_model_dir > "output_$t""_v$version.log"
    
#     sleep 5
#     echo "Oops! I fell asleep for a couple seconds!"
# done


# echo "start testing process for v2 dataset with test data v1"
# for t in ${listDataset[@]}; do
#     version=2
#     echo "Start Program $t of version $version"
#     res_dir="result/$t""_v4/"
#     saved_model_dir="saved_model/$t""_v$version" 
#     # echo $res_dir
#     mkdir -p $res_dir
#     mkdir -p $saved_model_dir
#     # run programming

#     # python3 few-shot-train-seresnet50.py -dn $t -s 20 -nd 0 -bb "seresnet50" > output_$t.log
#     python3 main.py -dn $t -s 20 -nd 0 -bb "seresnext50" -m false -rd $res_dir -ted "test_data_v1" -trd "train_data_v2" -eld "eval_data_v2" -smd $saved_model_dir > "output_$t""_v$version.log"
    
#     sleep 5
#     echo "Oops! I fell asleep for a couple seconds!"
# done







echo "start testing process for v1 dataset with test data v2"
for t in ${listDataset[@]}; do
    version=1
    echo "Start Program $t of version $version"
    res_dir="result/$t""_v2/"
    saved_model_dir="saved_model/$t""_v$version" 
    # echo $res_dir
    mkdir -p $res_dir
    mkdir -p $saved_model_dir
    # run programming

    # python3 few-shot-train-seresnet50.py -dn $t -s 20 -nd 0 -bb "seresnet50" > "output_$t""_v$version.log"
    python3 main.py -dn $t -s 20 -nd 0 -bb "seresnext50" --MODE=false -rd $res_dir -ted "test_data_v2" -trd "train_data" -eld "eval_data" -smd $saved_model_dir 
    
    sleep 5
    echo "Oops! I fell asleep for a couple seconds!"
done