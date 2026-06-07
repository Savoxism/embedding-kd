Write-Host "======================================"
Write-Host "Training with CDM method"
Write-Host "======================================"

$env:CUDA_VISIBLE_DEVICES = "0,1"
$env:TOKENIZERS_PARALLELISM = "false"

$METHOD = "cdm"
$TRAIN_DATA = "..\data\test_debug.csv"
$STUDENT_MODEL = "..\model_hub\MiniLMv2-L6-H384-distilled-from-BERT-Base\MiniLM-L6-H384-distilled-from-BERT-Base"
$TEACHER_MODEL = "Qwen/Qwen3-Embedding-0.6B"
$BATCH_SIZE = 32
$EPOCHS = 5
$LR = 2e-5
$MAX_LENGTH = 256
$SAVE_DIR = "checkpoints/cdm"

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
    --w_task 0.5 `
    --alpha_dtw 0.5 `
    --num_workers 2

Write-Host "======================================"
Write-Host "Training completed!"
Write-Host "======================================"
