# Historical ECHOPulse code

The original method and training source is retained in its original import layout:

- `EchoPulse_pytorch/cvivit.py`, `attention.py`: video tokenizer and attention components.
- `EchoPulse_pytorch/phenaki_pytorch*.py`: conditional sequence models.
- `EchoPulse_pytorch/*trainer*.py`: tokenizer and conditional-model trainers.
- `step1_train.py`, `step2_train.py`, `EchoPulse_train.py`: historical training entries.
- `EchoPulse_pytorch/STMEM/`: ECG representation models, training code, configurations and license.
- `EchoPulse_pytorch/frozen_models/pytorch_i3d/`: I3D source and license.
- `eval/`: historical evaluation components.

`requirements-echopulse.txt` is a historical dependency snapshot, separate from root `requirements.txt` for classification inference. Local file URLs in the snapshot were replaced with unpinned package names; it is not a newly validated compatible dependency lock.

Training entries execute training at module import and contain original placeholder paths/checkpoint assumptions. Inspect them and supply the relevant data and pretrained models before execution. Some imports refer to the installed `phenaki_pytorch` package, others to local `EchoPulse_pytorch`. No training was launched as part of publishing this source. Historical I3D `train_i3d.py` uses Python 2 syntax, and STMEM may require its historical PyTorch APIs/environment.

Datasets, pretrained backbones, example binary files and new generator checkpoints are not included. Existing third-party licenses and notices are preserved. These historical files are separate from the current classification method.
