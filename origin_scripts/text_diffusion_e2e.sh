python train_text_diffusion.py --dataset_name e2e --adam_weight_decay 0.01 --learning_rate 1e-4 --num_train_steps 100000 --train_batch_size 64 --tx_dim 768 --tx_depth 12 --objective pred_x0 --enc_dec_model facebook/bart-base --num_samples 1000 --self_condition --normalize_latent --scale_shift --loss_type l1 --beta_schedule linear