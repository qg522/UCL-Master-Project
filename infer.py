"""Classify the supplied test partition with a tensor-only state_dict."""
import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataloader import TestCineDataset
from heartmirror.classification import DiseaseClassifier
from heartmirror.config import Config


BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=BASE/'data')
    parser.add_argument('--weights', type=Path, default=BASE/'weights/ACDC.pt')
    parser.add_argument('--config', type=Path, required=True, help='Path to your model-compatible configuration')
    parser.add_argument('--classes', type=Path, default=BASE/'classes.json')
    parser.add_argument('--out', type=Path, default=BASE/'outputs/predictions.csv')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    torch.set_num_threads(2)
    cfg = Config.load(args.config)
    classes = json.loads(args.classes.read_text())
    dataset = TestCineDataset(args.data, cfg, classes)
    model = DiseaseClassifier(cfg, len(classes))
    state = torch.load(args.weights, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    device = torch.device(args.device)
    model.to(device).eval()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    correct, count = 0, 0
    with args.out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['patient_key', 'label', 'prediction'] + ['prob_'+c for c in classes])
        writer.writeheader()
        with torch.inference_mode():
            for batch in DataLoader(dataset, batch_size=args.batch_size, shuffle=False):
                logits = model(batch['cine'].to(device), batch['reference'].to(device),
                               batch['difference'].to(device))[0]
                probabilities = logits.float().softmax(-1).cpu()
                for key, label, probs in zip(batch['key'], batch['label'].tolist(), probabilities.tolist()):
                    pred = max(range(len(classes)), key=lambda i: probs[i])
                    writer.writerow(dict(patient_key=key, label=classes[label], prediction=classes[pred],
                                         **{'prob_'+c: probs[i] for i, c in enumerate(classes)}))
                    correct += pred == label
                    count += 1
                print(f'Inferred {count}/{len(dataset)} test-partition patients', flush=True)
    print(json.dumps(dict(patients=count, partition='seed0_test', accuracy=correct/count,
                          output=str(args.out.resolve())), indent=2))


if __name__ == '__main__':
    main()
