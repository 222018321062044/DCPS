import argparse

_TRAINER_ALIASES = {
    "CPrompt": "DCPS",
}

_VALID_TRAINERS = ("DCPS",)


def _split_csv(value):
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_trainer(value):
    trainer = _TRAINER_ALIASES.get(value, value)
    if trainer not in _VALID_TRAINERS:
        valid = ", ".join(_VALID_TRAINERS)
        raise argparse.ArgumentTypeError(f"Invalid trainer '{value}'. Expected one of: {valid}.")
    return trainer


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="DCPS training and evaluation entry point."
    )

    # Core experiment setup
    parser.add_argument("--model", type=str, default="ViT-B/16")
    parser.add_argument(
        "--train-mode",
        type=str,
        default="prompt",
        choices=["whole", "text", "image", "prompt"],
        help="Training mode.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="finetune",
        choices=["finetune"],
        help="Training method.",
    )
    parser.add_argument(
        "--trainer",
        type=_parse_trainer,
        default="DCPS",
        choices=_VALID_TRAINERS,
        help="Trainer variant.",
    )
    parser.add_argument("--train-dataset", type=str, default="Aircraft")
    parser.add_argument(
        "--eval-datasets",
        "--eval-dataset",
        dest="eval_datasets",
        type=_split_csv,
        default=["Aircraft"],
        help="Comma-separated evaluation dataset names.",
    )
    parser.add_argument(
        "--data-location",
        "--data_location",
        dest="data_location",
        type=str,
        default="dataset",
        help="Root directory containing prepared datasets.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=str,
        default="results",
        help="Directory for evaluation outputs and analysis artifacts.",
    )
    parser.add_argument(
        "--eval-names",
        "--eval_names",
        dest="eval_names",
        type=str,
        default="exp_dcps",
        help="Experiment name used by evaluation scripts.",
    )

    # Optimization
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--batch-size-eval", type=int, default=32, help="Evaluation batch size.")
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate.")
    parser.add_argument("--wd", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--warmup_length", type=int, default=100)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--meta_net_lr", type=float, default=2e-4, help="Meta network learning rate.")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--loss-interval", type=int, default=1000)
    parser.add_argument("--few-shot", type=int, default=None)
    parser.add_argument("--hard_sample_ratio", type=float, default=0.3)
    parser.add_argument("--threshold_percentile", type=float, default=0.05)
    parser.add_argument("--hard_loss_weight", type=float, default=0.3)

    # Runtime behavior
    parser.add_argument("--eval-every-epoch", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--save", type=str, default=None, help="Checkpoint output directory.")
    parser.add_argument("--load", type=str, default=None, help="Checkpoint path to resume from.")
    parser.add_argument("--dataset_order", default=None, type=_split_csv)
    parser.add_argument("--template", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for cached features.",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze the image encoder during evaluation or fine-tuning.",
    )
    parser.add_argument("--image_loss", action="store_true")
    parser.add_argument("--text_loss", action="store_true")

    # Prompt settings
    parser.add_argument("--prompt_width", type=int, default=2, help="Number of prompt tokens.")
    parser.add_argument("--ctx_init", type=str, default=None, help="Optional prompt initialization string.")
    parser.add_argument("--prompt_depth_vision", type=int, default=12, help="Prompt depth in the vision branch.")
    parser.add_argument("--prompt_depth_text", type=int, default=12, help="Prompt depth in the text branch.")
    parser.add_argument("--n_ctx_vision", type=int, default=2, help="Context tokens in the vision branch.")
    parser.add_argument("--n_ctx_text", type=int, default=2, help="Context tokens in the text branch.")
    parser.add_argument(
        "--input-size",
        nargs=2,
        type=int,
        default=(224, 224),
        metavar=("HEIGHT", "WIDTH"),
    )
    parser.add_argument("--optimizer", type=str, default="adam")

    args = parser.parse_args(argv)

    import torch

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.input_size = tuple(args.input_size)
    return args
