# Three-stage training and healthy-reference generation

Install the additional training/evaluation dependencies in an appropriate environment:

```bash
python -m pip install -r requirements-training.txt
```

## Entry points

| Entry | Purpose |
| --- | --- |
| `step1_train_acdc.py` | Cardiac tokenizer training |
| `step2_train_acdc.py` | Healthy-reference generator training |
| `step3_train_acdc.py` | Disease classifier training |
| `infer_acdc.py` | Generate healthy-reference cine from the first volume |
| `eval_acdc.py` | Evaluate generated references |
| `heartmirror_cli.py` | Prepare manifests/splits, run stages and evaluate |

Use `python <entry> --help` for full arguments. Stage execution requires your own manifest, patient partitions and relevant checkpoints. No additional experiment configuration files are supplied with this change. No root configuration JSON is supplied; classification inference also requires an explicit model-compatible --config path. The Config Python class remains required model code.

## Workflow templates

Replace placeholders with locally prepared inputs. The following commands are templates, not bundled runnable training experiments.

```bash
python step1_train_acdc.py --manifest PATH_TO_MANIFEST --splits PATH_TO_SPLITS --config YOUR_CONFIG --domain BU4DFE --scratch --perceptual-checkpoint PATH_TO_VGG --out runs/tokenizer_pretrain
python step1_train_acdc.py --manifest PATH_TO_MANIFEST --splits PATH_TO_SPLITS --config YOUR_CONFIG --domain MMs --init runs/tokenizer_pretrain/best.pt --perceptual-checkpoint PATH_TO_VGG --out runs/tokenizer_adapt
python step2_train_acdc.py --manifest PATH_TO_MANIFEST --splits PATH_TO_SPLITS --config YOUR_CONFIG --tokenizer runs/tokenizer_adapt/best.pt --out runs/generator
python infer_acdc.py --manifest PATH_TO_MANIFEST --splits PATH_TO_SPLITS --tokenizer runs/tokenizer_adapt/best.pt --checkpoint runs/generator/best.pt --out runs/references
python step3_train_acdc.py --manifest PATH_TO_MANIFEST --splits PATH_TO_SPLITS --config YOUR_CONFIG --references runs/references --out runs/classifier
python eval_acdc.py --manifest PATH_TO_MANIFEST --splits PATH_TO_SPLITS --references runs/references --out runs/generation_metrics.json
```

Patient partitions, domains and checkpoint provenance must be consistent across these operations. The runner checks subject membership and reference provenance. Full OOF generation/classification requires separate appropriate fold checkpoints, rather than reusing one training fold for every patient.

Root `infer.py` remains the lightweight classifier that reads the bundled tensor-only weights. Training and `infer_acdc.py` use full stage checkpoints with configuration/provenance. `weights/ACDC.pt` cannot replace tokenizer or generator checkpoints, and cannot be used as a full training resume checkpoint.

Perceptual training requires separately acquired VGG weights. FID/FVD evaluation requires suitable feature backbones; optional LPIPS and plotting dependencies are documented by their CLI options. This source addition does not include the missing datasets, generation-stage weights, baseline implementations or completed paper experiments.
