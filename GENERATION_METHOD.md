# Healthy reference generation: method implementation

This source release provides method modules, not a generation training or inference CLI. Generator/tokenizer checkpoints and perceptual-backbone weights are not bundled. The existing root `infer.py` remains the disease-classification entry point.

## Modules

- `heartmirror/tokenizer.py`: depth-preserving CardiacTokenizer, binary LFQ, differentiable token decoding and frozen perceptual features.
- `heartmirror/generation.py`: position-aligned autoregressive prediction, anomaly scoring, selective correction and objective definitions.
- `heartmirror/config.py`: shared architecture and objective settings.

## Tokenization

Inputs are B,T,Z,H,W. Each 2D slice is split into spatial patches, projected, processed by a spatial Transformer and averaged into a slice feature. Causal temporal attention is applied independently at each depth; depth is not merged with time. Projection produces `bits` binary dimensions. LFQ takes signs, uses a straight-through estimator and maps bit vectors to discrete IDs. The vocabulary has 2^bits entries; the current configuration uses 10 bits (1024 tokens).

The decoder projects binary vectors back to the hidden dimension, applies per-depth temporal decoding and spatial patch decoding, then reconstructs the cine. The reconstruction objective combines voxel L1, perceptual loss and LFQ entropy regularization using the configured weights. Frozen perceptual VGG features require their separately supplied checkpoint; they are not needed for classification inference.

## Position-aligned autoregression

The first-volume token embeddings are pooled and passed through an anatomy Transformer. Each query is a time/depth position embedding; memory consists of this anatomy token and raw token embeddings with position embeddings. Queries can attend only to prior phases at the same depth and earlier depth positions in the current phase. Targets cannot see their own token or future tokens indirectly through contextual memory. Query states are updated across ContextBlocks while the raw memory remains fixed.

Generation keeps the first-volume IDs unchanged, then predicts phase-by-phase and depth-by-depth. Tokens are sampled with configured temperature or selected greedily. Uncertainty is one minus the largest untempered softmax probability. Default architecture uses dimension 128, four heads and four autoregressive blocks.

## Detection and correction

The detector combines uncertainty with temporal and depth second differences of token embeddings, normalized as implemented in `risk_logits`. A two-layer MLP outputs a risk logit. At most `max_corrections` locations above `risk_threshold` are selected; the conditioning volume is excluded.

A separate corrector uses prior same-depth history, the next same-depth phase, adjacent depths in the current volume, and anatomy memory. It excludes its own target token. Corrections are computed from the same original rollout and applied once; the module does not repeatedly correct until convergence. The corrected IDs are decoded, and the first output volume is replaced with the supplied condition.

## Objective

`HealthyGenerator.objective` combines teacher-forced autoregressive cross entropy, detector binary cross entropy, corrector cross entropy, voxel L1 reconstruction, temporal/depth first-difference matching and cycle closure. Error locations plus random masks supervise correction. Discrete rollout is performed without gradients; conditional logits are recomputed and a straight-through token estimator carries voxel-loss gradients through the frozen decoder. Gradient propagation through a frozen decoder does not imply updating its parameters.

Temporal/depth residuals sum over H,W and average the remaining dimensions; cycle closure compares the final reconstruction to the first observation. Weights are read from `generator_weights`. Ablation flags control consistency terms and resampling.

## Scope

The root configuration is the published classifier-compatible 30-phase, 16-slice, 128-pixel setting. Instantiating generation modules with it is supported; this does not supply a trained generator or certify reproduction of another resolution. Method implementation is available without new stage-training or generation-inference entry points.
