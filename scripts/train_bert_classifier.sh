MODEL_NAME_OR_PATH="bert-base-uncased"
TOKENIZER_NAME="bert-base-uncased"
TASK_NAME="biasbios"
OUTPUT_DIR="models/bert-base-uncased-biasbios"
TRAIN_FILE="datasets/Biasbios/train.json"
VALID_FILE="datasets/Biasbios/valid.json"

accelerate launch train/run_glue_no_trainer.py \
    --tokenizer_name $TOKENIZER_NAME \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --task_name $TASK_NAME \
    --train_file $TRAIN_FILE \
    --validation_file $VALID_FILE \
    --max_length 512 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --num_train_epochs 10 \
    --output_dir $OUTPUT_DIR \
    --with_tracking
    