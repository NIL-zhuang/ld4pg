checkpoint=$1
tgt_name=$2

for seed in 42 43 44 45 46; do
  echo "processing with seed $seed, checkpoint $checkpoint, tgt file name $tgt_name"
  python main.py --config conf/config.yaml --seed "$seed" --mode eval --ckpt "$checkpoint" --tgt "results/$tgt_name/$seed.txt"
done
