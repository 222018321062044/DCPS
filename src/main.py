import os
import sys

from .args import parse_arguments

if "-h" in sys.argv or "--help" in sys.argv:
    parse_arguments()
    raise SystemExit(0)

from custom_clip import custom_clip as clip

from . import utils
from .models import evaluate, prompt_tune


def main(args):
    print(args)
    utils.seed_all(args.seed)

    if args.eval_only:
        model, _, val_preprocess = clip.load(args.model, args, jit=False)
        if args.load:
            utils.torch_load(model, args.load)
        elif args.save:
            checkpoint_pth = os.path.join(
                args.save, f"clip_zeroshot_{args.train_dataset}.pth"
            )
            utils.torch_save(model, checkpoint_pth)
        evaluate(model, args, val_preprocess)
    else:
        if args.train_mode == "prompt":
            model = prompt_tune(args)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
