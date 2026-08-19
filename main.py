import argparse
import sys
from config import (
    BaseConfig,
    CDMConfig,
    DSKDConfig,
    EMOConfig,
    HeatGeoConfig,
    StellaConfig,
    TALASConfig,
)
from distiller import KnowledgeDistiller


def parse_args():
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation for Embeddings Model"
    )
    
    parser.add_argument(
        '--method',
        type=str,
        default='cdm',
        choices=['cdm', 'dskd', 'emo', 'stella', 'talas', 'heatgeo'],
        help='Distillation method to use'
    )
    
    parser.add_argument(
        '--train_data',
        type=str,
        default=None,
        help='Path to training data CSV file'
    )
    parser.add_argument(
        '--eval_data',
        type=str,
        default=None,
        help='Path to evaluation data CSV file'
    )
    
    parser.add_argument(
        '--student_model',
        type=str,
        default=None,
        help='Student model name or path'
    )
    parser.add_argument(
        '--teacher_model',
        type=str,
        default=None,
        help='Teacher model name or path'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Training batch size'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=None,
        help='Maximum sequence length'
    )

    # HeatGeo experiment overrides. Keeping these on the CLI lets concurrent
    # teacher-student pairs use isolated caches without rewriting the shared
    # default config.
    parser.add_argument('--graph_k', type=int, default=None)
    parser.add_argument('--graph_temp', type=float, default=None)
    parser.add_argument(
        '--temp_exponent', type=float, default=None,
        help='Alpha in the temperature law tau_r = graph_temp * r^alpha '
             '(default 1.0 = heat-kernel value; 0.5 = historical ladder)'
    )
    parser.add_argument('--pool_size', type=int, default=None)
    parser.add_argument('--num_walks', type=int, default=None)
    parser.add_argument('--walk_length', type=int, default=None)
    parser.add_argument('--walk_weight', type=float, default=None)
    # --walk_temp was removed: the criterion ties it to --graph_temp.
    parser.add_argument('--walk_start_epoch', type=int, default=None)
    parser.add_argument('--walk_topk', type=int, default=None)
    parser.add_argument('--candidate_size', type=int, default=None)
    parser.add_argument('--diffusion_quota', type=int, default=None)
    parser.add_argument('--hard_neg_k', type=int, default=None)
    parser.add_argument('--random_neg_k', type=int, default=None)
    parser.add_argument('--cache_path', type=str, default=None)
    parser.add_argument('--heatgeo_cache_path', type=str, default=None)
    parser.add_argument('--heatgeo_log_dir', type=str, default=None)
    parser.add_argument(
        '--pooling_method',
        choices=['last_token', 'mean', 'cls'],
        default=None,
    )
    
    parser.add_argument(
        '--w_task',
        type=float,
        default=None,
        help='Task loss weight'
    )
    parser.add_argument(
        '--alpha_dtw',
        type=float,
        default=None,
        help='DTW KD loss weight'
    )
    parser.add_argument(
        '--task_type',
        choices=['single_cls', 'pair_cls', 'pair_reg'],
        default=None,
        help='Training task contract'
    )
    
    parser.add_argument(
        '--save_dir',
        type=str,
        default=None,
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--weights_dir',
        type=str,
        default=None,
        help='Optional durable directory for per-epoch student weights'
    )
    parser.add_argument(
        '--final_weights_only',
        action='store_true',
        help='Save only the final student state dict, not training checkpoints',
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='Number of dataloader workers'
    )
    parser.add_argument(
        '--no_wandb',
        action='store_true',
        help='Disable Weights & Biases logging'
    )
    parser.add_argument(
        '--wandb_project',
        type=str,
        default=None,
        help='Weights & Biases project name'
    )
    parser.add_argument(
        '--wandb_run_name',
        type=str,
        default=None,
        help='Weights & Biases run name'
    )
    parser.add_argument(
        '--wandb_mode',
        type=str,
        default=None,
        choices=['online', 'offline', 'disabled'],
        help='Weights & Biases mode'
    )
    
    return parser.parse_args()


def get_config(method: str, args):
    if method == 'cdm':
        config = CDMConfig()
    elif method == 'dskd':
        config = DSKDConfig()
    elif method == 'emo':
        config = EMOConfig()
    elif method == 'stella':
        config = StellaConfig()
    elif method == 'talas':
        config = TALASConfig()
    elif method == 'heatgeo':
        config = HeatGeoConfig()
    else:
        config = BaseConfig()
    
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
        'graph_k',
        'graph_temp',
        'temp_exponent',
        'pool_size',
        'num_walks',
        'walk_length',
        'walk_weight',
        'walk_start_epoch',
        'walk_topk',
        'candidate_size',
        'diffusion_quota',
        'hard_neg_k',
        'random_neg_k',
        'cache_path',
        'heatgeo_cache_path',
        'heatgeo_log_dir',
        'pooling_method',
    )
    for name in heatgeo_overrides:
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)
    
    if args.w_task is not None:
        config.w_task = args.w_task
    if args.alpha_dtw is not None:
        config.alpha_dtw = args.alpha_dtw
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
    
    config = get_config(args.method, args)
    
    print("\n" + "="*70)
    print(f"Configuration for {args.method.upper()} method:")
    print("="*70)
    for k, v in config.to_dict().items():
        print(f"  {k:25s} : {v}")
    print("="*70 + "\n")
    
    try:
        distiller = KnowledgeDistiller(config)
    except Exception as e:
        print(f"Error creating distiller: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
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


if __name__ == '__main__':
    main()
