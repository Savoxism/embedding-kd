param(
    [ValidateSet(
        "qwen3_0_6b_to_minilmv2_h384",
        "bge_m3_to_minilmv2_h768",
        "qwen3_4b_to_bert_base"
    )]
    [string]$Pair = "qwen3_0_6b_to_minilmv2_h384",
    [int]$Seed = 42,
    [string]$Gpu = "0",
    [switch]$PrepareCache,
    [switch]$DryRun,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PythonBin = if ($env:PYTHON_BIN) {
    $env:PYTHON_BIN
} else {
    Join-Path $RepoRoot ".venv\Scripts\python.exe"
}
$TrainData = if ($env:TRAIN_DATA) {
    $env:TRAIN_DATA
} else {
    Join-Path $RepoRoot "data\train_set\merged_3_data_5k_each.csv"
}
$CacheRoot = if ($env:CACHE_ROOT) {
    $env:CACHE_ROOT
} else {
    Join-Path $RepoRoot "cache\talas"
}

$Pairs = @{
    "qwen3_0_6b_to_minilmv2_h384" = @{
        Teacher = "Qwen/Qwen3-Embedding-0.6B"
        Student = "nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base"
        Pooling = "last_token"
    }
    "bge_m3_to_minilmv2_h768" = @{
        Teacher = "BAAI/bge-m3"
        Student = "nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Base"
        Pooling = "cls"
    }
    "qwen3_4b_to_bert_base" = @{
        Teacher = "Qwen/Qwen3-Embedding-4B"
        Student = "google-bert/bert-base-uncased"
        Pooling = "last_token"
    }
}
$Selected = $Pairs[$Pair]

if (-not (Test-Path -Path $PythonBin -PathType Leaf)) {
    throw "Project virtual-environment Python not found: $PythonBin"
}
if (-not (Test-Path -Path $TrainData -PathType Leaf)) {
    throw "TALAS training data not found: $TrainData"
}
if ($Seed -lt 0) {
    throw "Seed must be non-negative, got: $Seed"
}

$CachePath = Join-Path (Join-Path $CacheRoot $Pair) "teacher_train.pt"
$RunDir = if ($env:RUN_DIR) {
    $env:RUN_DIR
} else {
    Join-Path $RepoRoot "checkpoints\talas\$Pair\seed_$Seed"
}
$WeightsDir = if ($env:WEIGHTS_DIR) {
    $env:WEIGHTS_DIR
} else {
    Join-Path $RunDir "weights"
}

$env:CUDA_VISIBLE_DEVICES = $Gpu
$env:TOKENIZERS_PARALLELISM = "false"

$CommandArgs = @(
    (Join-Path $RepoRoot "main.py"),
    "--method", "talas",
    "--talas_pair", $Pair,
    "--train_data", $TrainData,
    "--student_model", $Selected.Student,
    "--teacher_model", $Selected.Teacher,
    "--task_type", "pair_cls",
    "--pooling_method", $Selected.Pooling,
    "--cache_path", $CachePath,
    "--batch_size", "32",
    "--epochs", "5",
    "--lr", "2e-5",
    "--max_length", "256",
    "--seed", "$Seed",
    "--save_dir", $RunDir,
    "--no_wandb"
)
if ($PrepareCache) {
    $CommandArgs += "--prepare_cache_only"
} else {
    $CommandArgs += @("--weights_dir", $WeightsDir, "--final_weights_only")
}
if ($ExtraArgs) {
    $CommandArgs += $ExtraArgs
}

Write-Host "TALAS pair=$Pair seed=$Seed gpu=$Gpu"
Write-Host "Teacher: $($Selected.Teacher)"
Write-Host "Student: $($Selected.Student)"
Write-Host "Teacher cache: $CachePath"
Write-Host "Run directory: $RunDir"

if ($DryRun) {
    Write-Output ((@($PythonBin) + $CommandArgs) -join " ")
    exit 0
}

& $PythonBin @CommandArgs
exit $LASTEXITCODE
