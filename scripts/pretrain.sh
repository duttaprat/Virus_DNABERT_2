export CUDA_VISIBLE_DEVICES=0,1,2,3,6
export TRITON_LOG=1



export TIMESTAMP=$(date +%Y%m%d_%H%M%S)


cd ..

python run_mlm_wandb.py \
  --model_name_or_path "zhihan1996/DNABERT-2-117M" \
  --train_file /home/pdutta/Data/DNABERT_2_data/pretraining_data/VirusHuman/Sequences_1000bp_overlap_500/70_30/train.txt \
  --validation_file /home/pdutta/Data/DNABERT_2_data/pretraining_data/VirusHuman/Sequences_1000bp_overlap_500/70_30/dev.txt \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
   --eval_accumulation_steps=500 \
  --do_train \
  --do_eval \
  --max_seq_length 1000 \
  --num_train_epochs 3000 \
  --learning_rate 3e-5 \
  --weight_decay 0.01 \
  --warmup_steps 100 \
  --gradient_accumulation_steps 2 \
  --save_steps 1000 \
  --logging_steps 1000 \
  --evaluation_strategy steps \
  --eval_steps 1000 \
  --fp16 \
  --line_by_line True \
  --report_to wandb \
  --output_dir /home/pdutta/DNABERT_2/pretrained_model/Sequences_1000bp_overlap_500_${TIMESTAMP} \
  --trust_remote_code True
