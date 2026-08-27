Write-Host "======================================"
Write-Host "Training with PKT method"
Write-Host "======================================"

$env:CUDA_VISIBLE_DEVICES = "0,1"
$env:TOKENIZERS_PARALLELISM = "false"

$METHOD = "pkt"
$TRAIN_DATA = "..\data\train_set\merged_3_data_5k_each.csv"
$STUDENT_MODEL = "google-bert/bert-base-uncased"
$TEACHER_MODEL = "Qwen/Qwen3-Embedding-4B"
$BATCH_SIZE = 32
$EPOCHS = 5
$LR = 2e-5
$MAX_LENGTH = 256
$SAVE_DIR = "checkpoints/pkt"

# Passalis & Tefas transfer without a supervised term (their loop defaults to
# supervised_weight=0), so w_task stays at 0 here.
python ../main.py `
    --method $METHOD `
    --train_data $TRAIN_DATA `
    --student_model $STUDENT_MODEL `
    --teacher_model $TEACHER_MODEL `
    --batch_size $BATCH_SIZE `
    --epochs $EPOCHS `
    --lr $LR `
    --max_length $MAX_LENGTH `
    --save_dir $SAVE_DIR `
    --w_task 0.0 `
    --w_pkt 1.0 `
    --pkt_kernel cosine `
    --num_workers 2

Write-Host "======================================"
Write-Host "Training completed!"
Write-Host "======================================"
