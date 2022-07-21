#! /bin/bash



### create folders
listFolder=("result" "samples" "saved_model")

for t in ${listFolder[@]}; do
    echo "Create folder $t if not exists"
    mkdir -p $t
done



### run experiments

# listDataset=(
#    "mura_scharr_x_8_v3" "mura_sobel_xy_32_v5" "mura_sobel_xy_64_v3" "mura_sobel_xy_ori_v5" "mura_april" 
# )


# listDataset=(
#     "mura_sobel_xy_64_v3" "mura_sobel_xy_ori_v5"
# )

listDataset=(
    "mura_sobel_xy_ori_v5"
)


listShots=(5 10 15 20)
listNoDataset=(0 1 2 3 4)

echo "start testing process for linear with gaussion filter 5x5 dataset"
for t in ${listDataset[@]}; do
    version=1
    echo "Start Program $t of version $version"
    res_dir="result/$t""_v5/"
    saved_model_dir="saved_model_0712/$t""_v1/"
    # echo $res_dir
    mkdir -p $res_dir
    mkdir -p $saved_model_dir
    # run programming
    python3 main.py -dn $t -s 20 -nd 0 -bb "seresnext50" -m false -rd $res_dir -ted "test_data_v1" -trd "train_data" -eld "eval_data" -smd $saved_model_dir -rtd "target_data_new_smoothed"

    sleep 5
    echo "Oops! I fell asleep for a couple seconds!"
done

echo "start training process for linear with gaussion filter 3x3 dataset"
for t in ${listDataset[@]}; do
    version=1
    echo "Start Program $t of version $version"
    res_dir="result/$t""_v6/"
    saved_model_dir="saved_model_0712/$t""_v1/"
    # echo $res_dir
    mkdir -p $res_dir
    mkdir -p $saved_model_dir
    # run programming
    python3 main.py -dn $t -s 20 -nd 0 -bb "seresnext50" -m false -rd $res_dir -ted "test_data_v1" -trd "train_data" -eld "eval_data" -smd $saved_model_dir -rtd "target_data_new_smoothed_v2"

    sleep 5
    echo "Oops! I fell asleep for a couple seconds!"
done











echo "start training process for linear with gaussion filter 5x5 dataset"
for t in ${listDataset[@]}; do
    version=1
    echo "Start Program $t of version $version"
    res_dir="result/$t""_v$version/"
    saved_model_dir="saved_model/$t""_v$version/"
    # echo $res_dir
    mkdir -p $res_dir
    mkdir -p $saved_model_dir
    # run programming
    python3 main.py -dn $t -s 20 -nd 0 -bb "seresnext50" -m true -rd $res_dir -ted "test_data_v1" -trd "train_data" -eld "eval_data" -smd $saved_model_dir -rtd "target_data_new_smoothed"

    sleep 5
    echo "Oops! I fell asleep for a couple seconds!"
done


echo "start training process for linear with gaussion filter 5x5 dataset"
for t in ${listDataset[@]}; do
    version=1
    echo "Start Program $t of version $version"
    res_dir="result/$t""_v$version/"
    saved_model_dir="saved_model/$t""_v$version/"
    # echo $res_dir
    mkdir -p $res_dir
    mkdir -p $saved_model_dir
    # run programming
    python3 main.py -dn $t -s 20 -nd 0 -bb "seresnext50" -m false -rd $res_dir -ted "test_data_v2" -trd "train_data" -eld "eval_data" -smd $saved_model_dir -rtd "target_data_new_smoothed"

    sleep 5
    echo "Oops! I fell asleep for a couple seconds!"
done








echo "start training process for linear with gaussion filter 3x3 dataset"
for t in ${listDataset[@]}; do
    version=2
    echo "Start Program $t of version $version"
    res_dir="result/$t""_v$version/"
    saved_model_dir="saved_model/$t""_v$version/"
    # echo $res_dir
    mkdir -p $res_dir
    mkdir -p $saved_model_dir
    # run programming
    python3 main.py -dn $t -s 20 -nd 0 -bb "seresnext50" -m true -rd $res_dir -ted "test_data_v1" -trd "train_data" -eld "eval_data" -smd $saved_model_dir -rtd "target_data_new_smoothed_v2"

    sleep 5
    echo "Oops! I fell asleep for a couple seconds!"
done


echo "start training process for linear with gaussion filter 3x3 dataset"
for t in ${listDataset[@]}; do
    version=2
    echo "Start Program $t of version $version"
    res_dir="result/$t""_v$version/"
    saved_model_dir="saved_model/$t""_v$version/"
    # echo $res_dir
    mkdir -p $res_dir
    mkdir -p $saved_model_dir
    # run programming
    python3 main.py -dn $t -s 20 -nd 0 -bb "seresnext50" -m false -rd $res_dir -ted "test_data_v2" -trd "train_data" -eld "eval_data" -smd $saved_model_dir -rtd "target_data_new_smoothed_v2"

    sleep 5
    echo "Oops! I fell asleep for a couple seconds!"
done