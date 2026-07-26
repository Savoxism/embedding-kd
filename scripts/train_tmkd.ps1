$ErrorActionPreference = "Stop"

Write-Host "======================================"
Write-Host "Training with TMKD method"
Write-Host "======================================"

$env:CUDA_VISIBLE_DEVICES = if ($env:CUDA_VISIBLE_DEVICES) { $env:CUDA_VISIBLE_DEVICES } else { "0,1" }
$env:TOKENIZERS_PARALLELISM = "false"

$TRAIN_DATA = if ($env:TRAIN_DATA) { $env:TRAIN_DATA } else { "..\data\test_debug.csv" }
$STUDENT_MODEL = if ($env:STUDENT_MODEL) { $env:STUDENT_MODEL } else { "bert-base-uncased" }
$TEACHER_MODEL = if ($env:TEACHER_MODEL) { $env:TEACHER_MODEL } else { "Qwen/Qwen3-Embedding-0.6B" }
$BATCH_SIZE = if ($env:BATCH_SIZE) { $env:BATCH_SIZE } else { 4 }
$EPOCHS = if ($env:EPOCHS) { $env:EPOCHS } else { 5 }
$LR = if ($env:LR) { $env:LR } else { "1e-5" }
$MAX_LENGTH = if ($env:MAX_LENGTH) { $env:MAX_LENGTH } else { 256 }
$SAVE_DIR = if ($env:SAVE_DIR) { $env:SAVE_DIR } else { "checkpoints/tmkd" }
$LAMBDA_TMKD = if ($env:LAMBDA_TMKD) { $env:LAMBDA_TMKD } else { 1.0 }
$TMKD_BLOCK_SIZE = if ($env:TMKD_BLOCK_SIZE) { $env:TMKD_BLOCK_SIZE } else { 512 }
$TMKD_MODE = if ($env:TMKD_MODE) { $env:TMKD_MODE } else { "full" }

python ../main.py `
    --method tmkd `
    --train_data $TRAIN_DATA `
    --student_model $STUDENT_MODEL `
    --teacher_model $TEACHER_MODEL `
    --batch_size $BATCH_SIZE `
    --epochs $EPOCHS `
    --lr $LR `
    --max_length $MAX_LENGTH `
    --save_dir $SAVE_DIR `
    --lambda_tmkd $LAMBDA_TMKD `
    --tmkd_block_size $TMKD_BLOCK_SIZE `
    --tmkd_mode $TMKD_MODE
