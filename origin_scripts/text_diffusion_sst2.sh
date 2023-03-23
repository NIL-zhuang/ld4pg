python train_text_diffusion.py \
 --learning_rate 0.0001 --adam_weight_decay 0.01 \
 --dataset_name sst \
 --num_train_steps 100000 --train_batch_size 64 --tx_dim 768 --tx_depth 12 \
 --objective pred_x0 \
 --enc_dec_model /home/data_91_d/zhuangzy/latent_diffusion/huggingface/bart-base \
 --num_samples 1000 --normalize_latent --scale_shift --beta_schedule linear \
 --loss_type l1 --class_conditional --self_condition