# PhysioDiff-4D

**Physics-Informed Normal Reference Generation for Cardiac Disease Detection and Pathological Sequence Synthesis**

PhysioDiff-4D is a generation-to-understanding framework for cardiac cine MRI analysis. The framework, referred to as **HeartMirror** in the thesis, learns healthy cardiac dynamics and generates a subject-specific healthy 4D cine MRI sequence from only the first 3D cardiac volume. The generated sequence serves as a personalized normal reference, allowing cardiac disease recognition to be formulated through the deviation between observed pathological motion and expected healthy motion.

This repository accompanies the UCL MSc Machine Learning project by **Qifan Gao**, supervised by **Dr. Yipeng Hu**.

## Motivation

Cardiac diseases are expressed not only through anatomical abnormalities but also through altered contraction, relaxation, and regional wall motion. Direct disease classification from pathological cine MRI is challenging because pathological motion is heterogeneous across patients and labelled 4D cardiac MRI data are limited.

PhysioDiff-4D addresses this problem by estimating how the same subject would move under a healthy cardiac dynamic prior. Given an observed pathological cine MRI sequence \(X^p\) and its generated healthy reference \(\hat{X}^h\), the normal-reference deviation is defined as


$\Delta X = X^p - \hat{X}^h.$


The deviation captures departures from expected subject-specific cardiac motion and is used together with the original pathological sequence for downstream disease classification.

## Framework

The framework consists of three stages.

### 1. Depth-Preserving Cardiac Video Tokenizer

Each short-axis slice at every cardiac phase is encoded into a compact discrete token. A spatial Transformer extracts in-plane anatomical information, while a causal temporal Transformer models motion independently at each depth position. Lookup-Free Quantisation (LFQ) converts the continuous latent features into binary representations without maintaining a learned codebook.

For a cine MRI sequence with 30 cardiac phases and 10 short-axis slices, the tokenizer produces 300 ordered tokens while preserving temporal and through-plane correspondence.

### 2. Position-Aligned Healthy Cine Generation

A position-aligned autoregressive Transformer generates the healthy token trajectory conditioned on the first 3D cardiac volume. For each phase, tokens are generated in short-axis depth order. Each prediction uses the temporal history of the corresponding depth position, previously generated slices in the current volume, and a global subject-specific anatomical condition.

To reduce accumulated autoregressive errors, a spatio-temporal anomaly detector identifies uncertain or locally inconsistent tokens. A lightweight corrector selectively resamples at most \(K\) suspicious positions using temporal neighbours, adjacent depth tokens, and the anatomical condition. Temporal, depth, cycle-closure, reconstruction, and token-level objectives encourage anatomically and physiologically coherent generation.

### 3. Generation-to-Understanding Classification

The observed pathological cine MRI and the normal-reference deviation are processed by two independent spatio-temporal Vision Transformer encoders. The classifier combines:

- disease-aware adaptive patch reweighting;
- supervised cross-stream contrastive learning;
- cross-attention from deviation tokens to pathological tokens; and
- patient-level disease prediction.

The difference stream highlights abnormal motion, while the pathological stream retains the anatomical context required to interpret it.

## Main Contributions

- A generation-to-understanding formulation that uses a personalized healthy cine MRI sequence as a normal dynamic reference for cardiac disease recognition.
- A depth-preserving LFQ cardiac video tokenizer that maintains the ordered short-axis organization of 4D cine MRI.
- A position-aligned autoregressive generator that preserves the temporal identity of corresponding depth locations.
- A spatio-temporal anomaly-triggered corrector that selectively resamples uncertain tokens using temporal and through-plane context.
- A dual-stream classifier that integrates pathological cine MRI and healthy-reference deviations through adaptive patch weighting, contrastive learning, and cross-attention.
- Evaluation of generation quality, few-shot disease classification, ablation, sensitivity, robustness, interpretability, and failure cases on ACDC and M&Ms.

## Datasets

The project uses the following datasets:

| Dataset | Role | Description |
| --- | --- | --- |
| ACDC | Healthy-reference generation and disease classification | 150 short-axis cine MRI studies across normal, DCM, HCM, MINF, and abnormal-RV categories. The available ED/ES segmentation masks are not used as generation conditions or classifier inputs. |
| M&Ms | Tokenizer adaptation and heterogeneous evaluation | 375 multi-centre, multi-vendor, and multi-disease cardiac MRI studies. A subset of 25 unlabelled cine sequences is used exclusively for tokenizer adaptation. |
| BU-4DFE | Backbone initialization | Temporally continuous non-rigid 3D facial sequences used to initialize the spatio-temporal tokenizer backbone before cardiac-domain adaptation. |

The generation result are available through Dropbox:

**[Download the final result](https://www.dropbox.com/scl/fo/f8h12x79vri8wgx61miay/AI4wc1VaOFeJk9OstDzqq6g?rlkey=2aghsgidpxbld5zae7dcn0uuj&st=0imzptlv&dl=0)**

This dropbox contains the counter-factual result to represent normal cardiac dynamic cine video in both M&Ms and ACDC.

Use of the datasets remains subject to the licenses, access conditions, and citation requirements of their original providers.

## Pre-processing

All cine MRI studies are converted to a common tensor representation:

\[
X \in \mathbb{R}^{30 \times 10 \times 240 \times 240},
\]

where the dimensions correspond to cardiac phase, short-axis depth, height, and width. The pipeline includes:

- temporal resampling to 30 cardiac phases;
- depth adjustment to 10 ordered short-axis positions;
- centre cropping or symmetric padding to \(240 \times 240\);
- study-level intensity normalization; and
- temporally consistent sequence-level augmentation for training data.

The first end-diastolic 3D volume is used as the generation condition.

## Evaluation Protocol

### Healthy-reference generation

Generation performance is assessed using:

- Fréchet Inception Distance (FID);
- Fréchet Video Distance (FVD);
- Structural Similarity Index (SSIM); and
- Learned Perceptual Image Patch Similarity (LPIPS).

The 30 healthy ACDC subjects are evaluated using five-fold patient-level cross-validation. Each fold contains 20 training, 4 validation, and 6 held-out test subjects.

### Few-shot cardiac disease classification

Classification follows a supervised 4-shot protocol, using four labelled training subjects per diagnostic category. Performance is reported at the patient level using accuracy, Macro-AUC, Macro-F1, and abnormal-right-ventricle recall. Experiments are repeated using five predefined random seeds with identical support sets and patient partitions across comparison methods.

## Results

### Healthy cine MRI generation

| Dataset | FID ↓ | FVD ↓ | SSIM ↑ | LPIPS ↓ |
| --- | ---: | ---: | ---: | ---: |
| ACDC | **9.62 ± 0.31** | **73.88 ± 1.76** | **0.964 ± 0.001** | **0.021 ± 0.001** |
| M&Ms | **12.50 ± 0.46** | **96.22 ± 2.14** | **0.814 ± 0.005** | 0.031 ± 0.002 |

### Supervised 4-shot disease classification on ACDC

| ACC ↑ | Macro-AUC ↑ | Macro-F1 ↑ | RV Recall ↑ |
| ---: | ---: | ---: | ---: |
| **85.7 ± 0.9%** | **90.1 ± 0.7%** | **86.3 ± 1.0%** | **63.3 ± 3.8%** |

The results support the use of personalized healthy-reference generation as an intermediate reasoning mechanism for label-efficient cardiac disease recognition.

## Citation

If this project supports your research, please cite the accompanying thesis:

```bibtex
@mastersthesis{gao2026physiodiff4d,
  author  = {Gao, Qifan},
  title   = {PhysioDiff-4D: Physics-Informed Normal Reference Generation for Cardiac Disease Detection and Pathological Sequence Synthesis},
  school  = {University College London},
  year    = {2026},
  type    = {MSc thesis}
}
```

## Acknowledgements

This work was completed as part of the MSc Machine Learning programme at University College London under the supervision of Dr. Yipeng Hu. We acknowledge the organizers and contributors of ACDC, M&Ms, and BU-4DFE for making their datasets available to the research community.


## Runnable ACDC classification inference

The repository root contains the runnable classifier, tensor-only weights, model configuration, data loader, and implementation-level [method description](METHOD.md). This release uses precomputed aligned healthy references; it does not include healthy-reference generator weights.

Download `heartmirror-inference-test-data.zip` from the repository Releases and extract it into the repository root so that `data/patients/` is present. The supplied data contains only the seed0 test partition: 30 patients, six per diagnosis.

```bash
python -m pip install -r requirements.txt
python infer.py
```

The entry point processes only `data/test.csv` and verifies `data/test_ids.json`. Outputs are per-patient class probabilities and predictions. Partition names and tensor-only weights do not establish training-data independence; this inference bundle makes no independent-generalization or paper-reproduction claim. ACDC terms and mandatory citation are included under `data/`.


## 分类推理使用说明

本包包含分类推理入口、纯参数权重、模型源码、dataloader、完整推理方法说明和seed0测试分区的30例配对MRI。每类6例，类别顺序固定为NOR、DCM、HCM、MINF、RV。

### 使用

```bash
python -m pip install -r requirements.txt
python infer.py
```

默认使用可用的CUDA，否则使用CPU。可以明确选择：

```bash
python infer.py --device cpu --batch-size 1
python infer.py --device cuda --batch-size 2 --out outputs/predictions.csv
```

推理只读取`data/test.csv`，并核对`data/test_ids.json`中的患者顺序；不遍历训练或验证分区。输出每例标签、预测类别和五类概率。标签只用于输出及计算分区准确率，不参与模型前向。

| 文件 | 用途 |
| --- | --- |
| `infer.py` | 严格加载参数并执行分类推理 |
| `dataloader.py` | 读取配对MRI并计算有符号差异 |
| `weights/ACDC.pt` | 127项张量组成的纯state_dict |
| `config.json`、`classes.json` | 参数对应的配置与类别顺序 |
| `heartmirror/` | 推理所需模型及Transformer实现 |
| `METHOD.md` | 数据、结构、公式、训练目标与推理方法 |
| `data/` | 测试分区清单、30例输入和数据许可说明 |

当前权重为4,538,375个参数。本包不包含参考生成器权重，使用已提供的生成参考，不能只输入原始MRI运行完整生成流程。测试分区名称和纯参数文件均不能单独证明训练时的数据独立性；本包输出用于执行验证，不作独立泛化或论文成绩声明。

本机验证记录可见`VALIDATION.json`。源码与权重包、测试影像包分开提供；若下载两份ZIP，将测试影像包解压到本包根目录，使`data/patients/`就位。

### 数据与许可

ACDC影像按原数据条款使用，许可和引用见`data/LICENSE_TERMS.md`及`data/MANDATORY_CITATION.md`。


## Generation method source and historical ECHOPulse

The current generation method is provided in [`heartmirror/tokenizer.py`](heartmirror/tokenizer.py) and [`heartmirror/generation.py`](heartmirror/generation.py), with the algorithm described in [GENERATION_METHOD.md](GENERATION_METHOD.md). This addition contains no generation training or inference CLI, and no generator checkpoints.

The original ECHOPulse method and training code is available under [`EchoPulse_pytorch/`](EchoPulse_pytorch/), with `step1_train.py`, `step2_train.py` and `EchoPulse_train.py` at the repository root. See [ECHOPULSE_LEGACY.md](ECHOPULSE_LEGACY.md) for layout, dependencies and historical execution assumptions. Root classification inference continues to use `requirements.txt`; legacy dependencies are recorded separately in `requirements-echopulse.txt`.
