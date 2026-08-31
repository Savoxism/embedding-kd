import argparse
import sys

from config import (
    TALAS_PAPER_PAIRS,
    BaseConfig,
    CDMConfig,
    DSKDConfig,
    EMOConfig,
    HeatGeoConfig,
    RKDConfig,
    StellaConfig,
    TALASConfig,
    get_talas_paper_pair,
)
from distiller import KnowledgeDistiller


def parse_args():
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation for Embeddings Model"
    )

    parser.add_argument(
        "--method",
        type=str,
        default="cdm",
        choices=["cdm", "dskd", "emo", "stella", "talas", "heatgeo", "rkd"],
        help="Distillation method to use",
    )

    parser.add_argument(
        "--train_data", type=str, default=None, help="Path to training data CSV file"
    )
    parser.add_argument(
        "--eval_data", type=str, default=None, help="Path to evaluation data CSV file"
    )

    parser.add_argument(
        "--student_model", type=str, default=None, help="Student model name or path"
    )
    parser.add_argument(
        "--teacher_model", type=str, default=None, help="Teacher model name or path"
    )
    parser.add_argument(
        "--talas_pair",
        choices=sorted(TALAS_PAPER_PAIRS),
        default=None,
        help="Canonical TALAS teacher-student pair from the paper",
    )

    parser.add_argument(
        "--batch_size", type=int, default=None, help="Training batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument(
        "--max_length", type=int, default=None, help="Maximum sequence length"
    )

    # HeatGeo experiment overrides. Keeping these on the CLI lets concurrent
    # teacher-student pairs use isolated caches without rewriting the shared
    # default config.
    parser.add_argument("--graph_k", type=int, default=None)
    parser.add_argument(
        "--perplexity",
        type=float,
        default=None,
        help="Target transition-row perplexity; 0 uses the fixed-bandwidth baseline",
    )
    parser.add_argument("--truncation_tolerance", type=float, default=None)
    parser.add_argument("--row_weight", type=float, default=None)
    parser.add_argument("--row_start_epoch", type=int, default=None)
    parser.add_argument("--direct_temp", type=float, default=None)
    parser.add_argument("--diffusion_quota", type=int, default=None)
    parser.add_argument("--hard_neg_k", type=int, default=None)
    parser.add_argument("--random_neg_k", type=int, default=None)
    parser.add_argument("--cache_path", type=str, default=None)
    parser.add_argument("--heatgeo_cache_path", type=str, default=None)
    parser.add_argument("--heatgeo_log_dir", type=str, default=None)
    parser.add_argument(
        "--pooling_method",
        choices=["last_token", "mean", "cls"],
        default=None,
    )

    parser.add_argument("--w_task", type=float, default=None, help="Task loss weight")
    parser.add_argument(
        "--alpha_dtw", type=float, default=None, help="DTW KD loss weight"
    )
    parser.add_argument(
        "--rkd_distance_weight",
        type=float,
        default=None,
        help="RKD distance-wise loss weight",
    )
    parser.add_argument(
        "--rkd_angle_weight",
        type=float,
        default=None,
        help="RKD angle-wise loss weight",
    )
    parser.add_argument(
        "--task_type",
        choices=["single_cls", "pair_cls", "pair_reg"],
        default=None,
        help="Training task contract",
    )

    parser.add_argument(
        "--save_dir", type=str, default=None, help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default=None,
        help="Optional durable directory for per-epoch student weights",
    )
    parser.add_argument(
        "--final_weights_only",
        action="store_true",
        help="Save only the final student state dict, not training checkpoints",
    )
    parser.add_argument(
        "--prepare_cache_only",
        action="store_true",
        help="Prepare and validate a cached-teacher method, then exit before training",
    )

    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--num_workers", type=int, default=None, help="Number of dataloader workers"
    )
    parser.add_argument(
        "--no_wandb", action="store_true", help="Disable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_project", type=str, default=None, help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default=None, help="Weights & Biases run name"
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default=None,
        choices=["online", "offline", "disabled"],
        help="Weights & Biases mode",
    )

    return parser.parse_args()


def get_config(method: str, args):
    if method == "cdm":
        config = CDMConfig()
    elif method == "dskd":
        config = DSKDConfig()
    elif method == "emo":
        config = EMOConfig()
    elif method == "stella":
        config = StellaConfig()
    elif method == "talas":
        config = TALASConfig()
    elif method == "heatgeo":
        config = HeatGeoConfig()
    elif method == "rkd":
        config = RKDConfig()
    else:
        config = BaseConfig()

    if args.talas_pair is not None:
        if method != "talas":
            raise ValueError("--talas_pair is supported only with --method talas")
        preset = get_talas_paper_pair(args.talas_pair)
        config.paper_pair = args.talas_pair
        config.teacher_model_name = preset["teacher"]
        config.student_model_name = preset["student"]
        config.pooling_method = preset["pooling_method"]

    if args.train_data is not None:
        config.train_data_path = args.train_data
    if args.eval_data is not None:
        config.eval_data_path = args.eval_data

    if args.student_model is not None:
        config.student_model_name = args.student_model
    if args.teacher_model is not None:
        config.teacher_model_name = args.teacher_model

    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.max_length is not None:
        config.max_length = args.max_length

    heatgeo_overrides = (
        "graph_k",
        "truncation_tolerance",
        "row_weight",
        "row_start_epoch",
        "direct_temp",
        "diffusion_quota",
        "hard_neg_k",
        "random_neg_k",
        "cache_path",
        "heatgeo_cache_path",
        "heatgeo_log_dir",
        "pooling_method",
    )
    for name in heatgeo_overrides:
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)
    # None already means "leave the config unchanged", so 0 disables the
    # entropic-affinity bandwidth and selects the fixed-bandwidth baseline.
    if args.perplexity is not None:
        config.perplexity = None if args.perplexity <= 0 else args.perplexity

    if args.w_task is not None:
        config.w_task = args.w_task
    if args.alpha_dtw is not None:
        config.alpha_dtw = args.alpha_dtw
    if args.rkd_distance_weight is not None:
        config.rkd_distance_weight = args.rkd_distance_weight
    if args.rkd_angle_weight is not None:
        config.rkd_angle_weight = args.rkd_angle_weight
    if args.task_type is not None:
        config.task_type = args.task_type

    if args.save_dir is not None:
        config.save_dir = args.save_dir
    if args.weights_dir is not None:
        config.weights_dir = args.weights_dir
    if args.final_weights_only:
        config.final_weights_only = True

    if args.seed is not None:
        config.seed = args.seed
    if args.debug:
        config.debug_align = True
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.no_wandb:
        config.use_wandb = False
    if args.wandb_project is not None:
        config.wandb_project = args.wandb_project
    if args.wandb_run_name is not None:
        config.wandb_run_name = args.wandb_run_name
    if args.wandb_mode is not None:
        config.wandb_mode = args.wandb_mode

    return config


def main():
    args = parse_args()

    if args.prepare_cache_only and args.method not in {"talas", "rkd"}:
        raise SystemExit(
            "--prepare_cache_only is supported only with --method talas or rkd"
        )

    config = get_config(args.method, args)

    print("\n" + "=" * 70)
    print(f"Configuration for {args.method.upper()} method:")
    print("=" * 70)
    for k, v in config.to_dict().items():
        print(f"  {k:25s} : {v}")
    print("=" * 70 + "\n")

    try:
        distiller = KnowledgeDistiller(config)
    except Exception as e:
        print(f"Error creating distiller: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    if args.prepare_cache_only:
        print(
            f"{args.method.upper()} teacher cache prepared and validated: {config.cache_path}"
        )
        return

    try:
        distiller.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
