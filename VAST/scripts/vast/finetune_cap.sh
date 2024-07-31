##### VIDEO-CAP

nohup python ./run.py \
--learning_rate \
5e-5 \
--train_batch_size \
4 \
--train_epoch \
20 \
--checkpointing \
true \
--save_best \
true \
--config \
./config/vast/finetune_cfg/caption-medqa.json \
--pretrain_dir \
./vision_captioner \
--beam_size \
3 \
--first_eval \
false \
--output_dir \
./output/results/fine-tune \
> t1.out &