import argparse
import sys

from config import (
    BaseConfig,
    CDMConfig,
    DSKDConfig,
    EMOConfig,
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
        choices=['cdm', 'dskd', 'emo', 'stella', 'talas'],
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
        '--teacher_pooling',
        choices=['last_token', 'mean', 'cls'],
        default=None,
        help='Pooling used when caching teacher sentence embeddings'
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
        '--save_every',
        type=int,
        default=None,
        help='Save a periodic checkpoint every N epochs (must be positive)'
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
        '--cache_path',
        type=str,
        default=None,
        help='Teacher embedding cache path'
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
    if args.teacher_pooling is not None:
        config.pooling_method = args.teacher_pooling
    
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.save_every is not None:
        if args.save_every <= 0:
            raise ValueError("--save_every must be a positive integer")
        config.save_every = args.save_every
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.max_length is not None:
        config.max_length = args.max_length
    
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
    if args.cache_path is not None:
        config.cache_path = args.cache_path
    
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
    finally:
        distiller.close()


if __name__ == '__main__':
    main()
