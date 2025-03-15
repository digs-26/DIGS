# 定义一个函数来处理SIGINT信号
handle_sigint() {
    echo "脚本已被Ctrl+C终止"
    exit 1  # 退出脚本
}

# 使用trap命令来捕获SIGINT信号，并调用handle_sigint函数
trap handle_sigint SIGINT
export CUDA_VISIBLE_DEVICES=0
echo "Using GPU(s): $CUDA_VISIBLE_DEVICES"

#trainsets="test1"
#testsets2="test2"
#handlesets="123456"
#over="SASS"
#partsets="test3"
#cd ./scripts

#under_list=("0.2")
#for under in "${under_list[@]}"; do
#  datasets="${trainsets}_${partsets}_${handlesets}_under${under}_over_${over}"
#  bash run.sh --datasets $datasets --trainsets $trainsets --partsets $partsets --ratio "1.0" --handlesets $handlesets --under $under --over $over
#done

trainsets="reveal"
testsets2="reveal"
handlesets="123470"
over="moderate"
partsets="vulscriber"
cd ./scripts
#
under_list=("0.6")
#
for under in "${under_list[@]}"; do
  datasets="${trainsets}_${partsets}_${handlesets}_under${under}_over_${over}"
  bash run.sh --datasets $datasets --trainsets $trainsets --validsets "none" --partsets $partsets  --ratio "1.0" --handlesets $handlesets --under $under --over $over
done
