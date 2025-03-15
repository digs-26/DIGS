# 定义一个函数来处理SIGINT信号
handle_sigint() {
    echo "脚本已被Ctrl+C终止"
    exit 1  # 退出脚本
}

# 使用trap命令来捕获SIGINT信号，并调用handle_sigint函数
trap handle_sigint SIGINT



#!/bin/bash

# 初始化参数
datasets=()
trainsets=()
partsets=()
part_indices=()
testsets1=()
testsets2=()
sf=()


# 使用getopts处理命名参数
while (( "$#" )); do
  case "$1" in
    --datasets)
      shift
      while (( "$#" )) && [[ "$1" != --* ]]; do
        datasets+=("$1")
        shift
      done
      ;;
    --trainsets)
      shift
      while (( "$#" )) && [[ "$1" != --* ]]; do
        trainsets+=("$1")
        shift
      done
      ;;
    --partsets)
      shift
      while (( "$#" )) && [[ "$1" != --* ]]; do
        partsets+=("$1")
        shift
      done
      ;;
    --part_indices)
      shift
      while (( "$#" )) && [[ "$1" != --* ]]; do
        part_indices+=("$1")
        shift
      done
      ;;
    --testsets1)
      shift
      while (( "$#" )) && [[ "$1" != --* ]]; do
        testsets1+=("$1")
        shift
      done
      ;;
    --testsets2)
      shift
      while (( "$#" )) && [[ "$1" != --* ]]; do
        testsets2+=("$1")
        shift
      done
      ;;
    --sf)
      shift
      while (( "$#" )) && [[ "$1" != --* ]]; do
        sf+=("$1")
        shift
      done
      ;;
    --) # 结束参数处理
      shift
      break
      ;;
    -*|--*=) # 不支持的参数
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
  esac
done

# 在这里，你可以使用$datasets和$testsets数组
echo "Datasets: ${datasets[@]}"
echo "Trainsets: ${trainsets[@]}"
echo "Partsets: ${partsets[@]}"
echo "Testsets1: ${testsets1[@]}"
echo "Testsets2: ${testsets2[@]}"
echo "Part Indices: ${part_indices[@]}"
echo "SF: ${sf[@]}"


seeds=(1000)
for sf in "${sf[@]}"; do
  for part_index in "${part_indices[@]}"; do
    for seed in "${seeds[@]}"; do
      for dataset in "${datasets[@]}"; do
        for trainset in "${trainsets[@]}"; do
          for partset in "${partsets[@]}"; do
            for testset1 in "${testsets1[@]}"; do
              for testset2 in "${testsets2[@]}"; do
                output_root="../../devign_storage/$dataset"
                if [ ! -d "$output_root" ]; then
                  mkdir -p "$output_root"
                fi
                processed_train_path=../../devign_storage/shard/$trainset
                processed_part_path=../../devign_storage/shard/$partset
                processed_test1_path=../../devign_storage/shard/$testset1
                processed_test2_path=../../devign_storage/shard/$testset2
                trains=$(find ../../devign_storage/shard/$trainset -type f -name "*shard*")
                parts=$(find ../../devign_storage/shard/$partset -type f -name "*shard*")
                tests1=$(find ../../devign_storage/shard/$testset1 -type f -name "*shard*")
                tests2=$(find ../../devign_storage/shard/$testset2 -type f -name "*shard*")
                if [ "$partset" == "none" ]; then
                  exec python -u ../code/main.py \
                  --mode train \
                  --dataset_root ../../devign_storage \
                  --train_mode step_2000 \
                  --dataset $dataset \
                  --seed "$seed" \
                  --sf "$sf" \
                  --model_type devign \
                  --train_src $trains \
                  --test1_src $tests1 \
                  --test2_src $tests2 \
                  --processed_train_path $processed_train_path \
                  --processed_test1_path $processed_test1_path \
                  --processed_test2_path $processed_test2_path \
                  2>&1 | tee "$output_root/${trainset}_${partset}_${testset1}_${testset2}_${seed}_${sf}.log"
                else
                  exec python -u ../code/main.py \
                  --mode train \
                  --dataset_root ../../devign_storage \
                  --train_mode step_2000 \
                  --dataset $dataset \
                  --seed "$seed" \
                  --sf "$sf" \
                  --model_type devign \
                  --train_src $trains \
                  --part_src $parts \
                  --part_indices $part_index \
                  --test1_src $tests1 \
                  --test2_src $tests2 \
                  --processed_train_path $processed_train_path \
                  --processed_part_path $processed_part_path \
                  --processed_test1_path $processed_test1_path \
                  --processed_test2_path $processed_test2_path \
                  2>&1 | tee "$output_root/${trainset}_${partset}_${testset1}_${testset2}_${seed}_${sf}.log"
                fi
              done
            done
          done
        done
      done
    done
  done
done





#train_srcs="/root/my_eval/RQ1/dataset/reveal_shard/reveal.json.shard1 \
#/root/my_eval/RQ1/dataset/reveal_shard/reveal.json.shard2 \
#/root/my_eval/RQ1/dataset/reveal_shard/reveal.json.shard3 \
#/root/my_eval/RQ1/dataset/reveal_shard/reveal.json.shard4 "
#
#test_srcs="/root/my_eval/RQ1/dataset/reveal_shard/devign_test.shard1"
#
#declare -A extra_train_srcs=(
#    ["baseline"]=""
#    ["gen"]="/root/my_eval/RQ1/dataset/reveal_shard/reveal_gen.json.shard1"
#)
#
#dataset_root="/root/my_eval/RQ1/devign/data_storage_generalization"
#
#seeds=(1000)
#
#for seed in "${seeds[@]}"; do
#  for dataset in "${!extra_train_srcs[@]}"; do
#    mkdir -p "$dataset_root/$dataset/models-seed$seed"
#    printf "processing $dataset seed $seed\n"
#    cmd="python -u main.py --mode train --dataset_root $dataset_root --train_mode step_2000 --dataset $dataset \
#    --seed $seed --model_type devign \
#    --train_src $train_srcs ${extra_train_srcs[$dataset]} --test_src $test_srcs \
#    > $dataset_root/$dataset/models-seed$seed/$dataset.out \
#    2> $dataset_root/$dataset/models-seed$seed/$dataset.err"
#    echo $cmd
#    eval $cmd
#  done
#done


#train_srcs="/root/my_eval/RQ1/dataset/shard_same_set/train_baseline.json.shard1 \
#/root/my_eval/RQ1/dataset/shard_same_set/train_baseline.json.shard2 \
#/root/my_eval/RQ1/dataset/shard_same_set/train_baseline.json.shard3 \
#/root/my_eval/RQ1/dataset/shard_same_set/train_baseline.json.shard4 "
#train_srcs="/root/my_eval/RQ1/dataset/shard_same_set/train_my_gen.json.shard1 \
#/root/my_eval/RQ1/dataset/shard_same_set/train_my_gen.json.shard2 \
#/root/my_eval/RQ1/dataset/shard_same_set/train_my_gen.json.shard3 \
#/root/my_eval/RQ1/dataset/shard_same_set/train_my_gen.json.shard4 \
#/root/my_eval/RQ1/dataset/shard_same_set/train_my_gen.json.shard5 "


#test_srcs="/root/my_eval/RQ1/dataset/shard_same_set/test_baseline.json.shard1"
#test_srcs="/root/my_eval/RQ1/dataset/shard_same_set/test_my_gen.json.shard1"

#declare -A same_set_train_srcs=(
#    ["baseline"]="/root/my_eval/RQ1/dataset/shard_same_set/train_baseline.shard1 \
#                  /root/my_eval/RQ1/dataset/shard_same_set/train_baseline.shard2 \
#                  /root/my_eval/RQ1/dataset/shard_same_set/train_baseline.shard3 \
#                  /root/my_eval/RQ1/dataset/shard_same_set/train_baseline.shard4 "
#    ["my_gen"]="/root/my_eval/RQ1/dataset/shard_same_set/train_my_gen.shard1 \
#                /root/my_eval/RQ1/dataset/shard_same_set/train_my_gen.shard2 \
#                /root/my_eval/RQ1/dataset/shard_same_set/train_my_gen.shard3 \
#                /root/my_eval/RQ1/dataset/shard_same_set/train_my_gen.shard4 \
#                /root/my_eval/RQ1/dataset/shard_same_set/train_my_gen.shard5 "
#)
#declare -A same_set_test_srcs=(
#    ["baseline"]="/root/my_eval/RQ1/dataset/shard_same_set/test_baseline.shard1"
#    ["my_gen"]="/root/my_eval/RQ1/dataset/shard_same_set/test_my_gen.shard1"
#)
#
#dataset_root="/root/my_eval/RQ1/devign/data_storage_same_set"

#dataset=$1
#shift
#subsets=($@)
#
#seeds=(1000)
#for seed in "${seeds[@]}"; do
#  for subset in "${subsets[@]}"; do
#    output_root="../$dataset/data_storage_$(echo "$subset" | sed s@/@-@g)"
#    if [ ! -d "$output_root" ]; then
#      mkdir -p "$output_root"
#    fi
#
#    trains=$(find ../"$dataset"/data/"$subset"/ -type f -name "*train*")
#    tests=$(find ../"$dataset"/data/"$subset"/ -type f -name "*test*")
##    echo "$output_root"
##    echo "$trains"
#    exec python  ../code/main.py \
#    --mode train \
#    --dataset_root "$output_root" \
#    --train_mode step_2000 \
#    --dataset "$subset" \
#    --seed "$seed" \
#    --model_type devign \
#    --train_src $trains \
#    --test_src $tests 2>&1 | tee "$output_root/$(echo "$subset" | sed s@/@-@g)_$seed.log"
#
#  done
#done