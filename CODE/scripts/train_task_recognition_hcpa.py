"""HCP-A four-task recognition with DeepHoloBrain (Sec. 4.2, Table 1).

Five-fold cross-validation over FACENAME, VISMOTOR, CARIT and REST.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepholobrain.data import FC_SCDataset
from deepholobrain.models import DeepHoloBrain
from scripts.common import run_kfold_classification


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fc_path', required=True, help='Dir of per-subject FC .csv files (AAL116).')
    parser.add_argument('--sc_path', required=True, help='Dir of per-subject SC .mat folders.')
    parser.add_argument('--output_path', type=str, default='hcpa_deepholobrain')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--number_cls', type=int, default=4)
    parser.add_argument('--spd', type=int, default=1, help='1: manifold path (Fig. 1/3); 0: plain-MLP ablation.')
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--scale_regularization_beta', type=float, default=1.0,
                        help='Beta in Eq. 4 (the paper does not report its numerical value).')
    parser.add_argument('--use_conda', type=str, default='cuda:0', help='torch device string.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    device = torch.device(args.use_conda if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)

    dataset = FC_SCDataset(args.fc_path, args.sc_path)
    print(f'num_sample={len(dataset)} num_folds={args.num_folds}')
    run_kfold_classification(
        dataset=dataset, model_fn=DeepHoloBrain, num_classes=args.number_cls,
        num_folds=args.num_folds, epochs=args.epochs, lr=args.lr, spd=args.spd,
        device=device, output_dir_name=args.output_path, batch_size=args.batch_size,
        num_workers=args.num_workers, seed=args.seed,
        scale_regularization_beta=args.scale_regularization_beta,
    )
