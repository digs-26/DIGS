# 定义一个函数来处理SIGINT信号
handle_sigint() {
    echo "脚本已被Ctrl+C终止"
    exit 1  # 退出脚本
}

# 使用trap命令来捕获SIGINT信号，并调用handle_sigint函数
trap handle_sigint SIGINT

# 初始化参数
datasets=()
trainsets=()
partsets=()
validsets=()
testsets1=()
testsets2=()
ratio=()
handlesets=()
under=()
over=()

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
    --validsets)
      shift
      while (( "$#" )) && [[ "$1" != --* ]]; do
        validsets+=("$1")
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
    --ratio)
      shift
      while (( "$#" )) && [[ "$1" != --* ]]; do
        ratio+=("$1")
        shift
      done
      ;;
    --handlesets)
      shift
      while (( "$#" )) && [[ "$1" != --* ]]; do
        handlesets+=("$1")
        shift
      done
      ;;
    --under)
      shift
      while (( "$#" )) && [[ "$1" != --* ]]; do
        under+=("$1")
        shift
      done
      ;;
    --over)
      shift
      while (( "$#" )) && [[ "$1" != --* ]]; do
        over+=("$1")
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
echo "Validsets: ${validsets[@]}"
echo "Testsets1: ${testsets1[@]}"
echo "Testsets2: ${testsets2[@]}"
echo "Ratio: ${ratio[@]}"
echo "Handlesets: ${handlesets[@]}"
echo "Under: ${under[@]}"
echo "Over: ${over[@]}"

seeds=(123456)
for dataset in "${datasets[@]}"; do
  for trainset in "${trainsets[@]}"; do
    for partset in "${partsets[@]}"; do
      for validset in "${validsets[@]}"; do
        for testset1 in "${testsets1[@]}"; do
          for testset2 in "${testsets2[@]}"; do
            for r in "${ratio[@]}"; do
              for handleset in "${handlesets[@]}"; do
                for under in "${under[@]}"; do
                  for over in "${over[@]}"; do
                    for seed in "${seeds[@]}"; do
                      output_root="../../linevul_storage/$dataset"
                      if [ ! -d "$output_root" ]; then
                        mkdir -p "$output_root"
                      fi
                      trains=$(find ../../linevul_storage/json/$trainset -type f -name "*jsonl*")
                      parts=$(find ../../linevul_storage/json/$partset -type f -name "*jsonl*")
                      valids=$(find ../../linevul_storage/json/$validset -type f -name "*jsonl*")
                      tests1=$(find ../../linevul_storage/json/$testset1 -type f -name "*jsonl*")
                      tests2=$(find ../../linevul_storage/json/$testset2 -type f -name "*jsonl*")
                      if [ "$partset" == "none" ]; then
                        exec python -u ../code/linevul/linevul_main.py \
                          --output_dir=$output_root/saved_models \
                          --model_type=roberta \
                          --tokenizer_name=microsoft/codebert-base \
                          --model_name_or_path=microsoft/codebert-base \
                          --do_train \
                          --train_data_file=$trains \
                          --eval_data_file=$valids \
                          --test1_data_file=$tests1 \
                          --test2_data_file=$tests2 \
                          --epochs 10 \
                          --block_size 512 \
                          --train_batch_size 32 \
                          --eval_batch_size 32 \
                          --learning_rate 5e-5 \
                          --max_grad_norm 1.0 \
                          --evaluate_during_training \
                          --ratio $r \
                          --handlesets $handleset \
                          --under $under \
                          --over $over \
                          --seed $seed 2>&1 | tee "$output_root/${trainset}_${partset}_${testset1}_${testset2}_${handleset}_under${under}_over_${over}_${seed}.log"
                      else
                        exec python -u ../code/linevul/linevul_main.py \
                          --output_dir=$output_root/saved_models \
                          --model_type=roberta \
                          --tokenizer_name=microsoft/codebert-base \
                          --model_name_or_path=microsoft/codebert-base \
                          --do_train \
                          --train_data_file=$trains \
                          --part_data_file=$parts \
                          --eval_data_file=$valids \
                          --test1_data_file=$tests1 \
                          --test2_data_file=$tests2 \
                          --epochs 10 \
                          --block_size 512 \
                          --train_batch_size 32 \
                          --eval_batch_size 32 \
                          --learning_rate 5e-5 \
                          --max_grad_norm 1.0 \
                          --evaluate_during_training \
                          --ratio $r \
                          --handlesets $handleset \
                          --under $under \
                          --over $over \
                          --seed $seed 2>&1 | tee "$output_root/${trainset}_${partset}_${testset1}_${testset2}_${handleset}_under${under}_over_${over}_${seed}.log"
                      fi
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

