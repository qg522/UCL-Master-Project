"""Reproducible training, OOF reference construction, evaluation and table export."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import Config
from .data import (CLASSES, CineDataset, acdc_manifest, load_cine, make_splits,
                   read_manifest)
from .tokenizer import CardiacTokenizer, FrozenPerceptual
from .generation import HealthyGenerator
from .classification import DiseaseClassifier
from .baselines import DirectRegression
from .metrics import paired_metrics, classification_metrics, frechet, aggregate_runs
from .experiments import EXPERIMENTS
from .paired_data import PairedCineDataset, audit_paired_data


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_hash(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            sha.update(chunk)
    return sha.hexdigest()


def load_checkpoint(path, device='cpu'):
    # Only load user-owned local checkpoints; all checkpoints contain plain data/tensors.
    return torch.load(path, map_location=device, weights_only=True)


def tokenizer_from(path, device):
    state = load_checkpoint(path)
    if state['stage'] != 'tokenizer':
        raise ValueError('A HeartMirror LFQ checkpoint is required (VQ checkpoints incompatible)')
    cfg = Config(**state['config']).validate()
    tok = CardiacTokenizer(cfg).to(device)
    tok.load_state_dict(state['model'], strict=True)
    tok.eval().requires_grad_(False)
    return tok, cfg, state


def prepare(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.acdc_root:
        acdc_manifest(args.acdc_root, out/'manifest.csv')
        manifest = out/'manifest.csv'
    elif args.manifest:
        manifest = args.manifest
    else:
        raise ValueError('Provide --acdc-root or --manifest')
    rows = read_manifest(manifest)
    if args.dataset == 'MMs':
        mm = [r for r in rows if r['dataset'] == 'MMs']
        if len(mm) != 375 or sum(r['tokenizer_only'] for r in mm) != 25:
            raise ValueError('Paper M&Ms protocol requires 375 subjects, 25 marked tokenizer_only')
    cfg = Config.load(args.config)
    cfg.save(out/'config.json')
    classes = tuple(args.classes.split(','))
    for seed in cfg.seeds:
        write_json(out/f'splits_seed{seed}.json', make_splits(rows, args.dataset, seed, classes))
    write_json(out/'experiments.json', EXPERIMENTS)
    print(f'Prepared {len(rows)} subjects, five seeds x five folds in {out}')


def verify_partitions(rows, split):
    index = {r['key']: r for r in rows}
    for stage in ('generator', 'classifier'):
        lists = [split[stage][s] for s in ('train', 'val', 'test')]
        flat = sum(lists, [])
        if len(set(flat)) != len(flat):
            raise ValueError(f'Patient leakage in {stage}')
        for key in flat:
            if key not in index or index[key]['tokenizer_only']:
                raise ValueError(f'Unknown/tokenizer-only subject in {stage}: {key}')
            if stage == 'generator' and index[key]['group'] != 'NOR':
                raise ValueError('Generator may only train/evaluate paired healthy cine')


def audit_reference_method(rows, ids, root, seed):
    """Ensure one comparison row never mixes different reference semantics."""
    methods = set()
    for row in rows:
        if row['key'] not in ids:
            continue
        path = Path(root)/row['dataset']/(row['patient_id']+'.npz')
        with np.load(path, allow_pickle=False) as f:
            meta = json.loads(str(f['metadata']))
        if meta['seed'] != seed or meta['patient_key'] != row['key']:
            raise ValueError(f'Reference patient/seed mismatch: {path}')
        methods.add(meta['method'])
    if len(methods) != 1:
        raise ValueError(f'References must use a single method, found {methods}')
    return next(iter(methods))


def train(args):
    cfg = Config.load(args.config)
    seed_all(args.seed)
    device = torch.device(args.device)
    rows = read_manifest(args.manifest)
    tok = perceptual = None
    provenance = {}
    paired_root = getattr(args, 'paired_data', None)
    if paired_root and args.stage != 'classifier':
        raise ValueError('--paired-data is only for classifier training')
    if args.stage == 'tokenizer':
        if not args.train_ids or not args.val_ids or not args.domain:
            raise ValueError('Tokenizer requires explicit --train-ids, --val-ids and --domain')
        train_ids, val_ids = read_json(args.train_ids), read_json(args.val_ids)
        if not train_ids or not val_ids or set(train_ids) & set(val_ids):
            raise ValueError('Tokenizer needs nonempty, disjoint patient-level train/val')
        model = CardiacTokenizer(cfg)
        if args.init:
            initial = load_checkpoint(args.init)
            if initial['stage'] != 'tokenizer':
                raise ValueError('Tokenizer initialization must be LFQ tokenizer')
            model.load_state_dict(initial['model'], strict=True)
            provenance = dict(initial.get('provenance', {}))
        elif args.domain != 'BU4DFE' and not args.scratch:
            raise ValueError('Cross-domain run needs --init; --scratch explicitly selects pretraining ablation')
        by_key = {r['key']: r for r in rows}
        for key in train_ids+val_ids:
            if key not in by_key or by_key[key]['dataset'] != args.domain:
                raise ValueError(f'Tokenizer data must belong to --domain: {key}')
        seen = set(provenance.get('tokenizer_seen_ids', [])) | set(train_ids) | set(val_ids)
        provenance.update(tokenizer_seen_ids=sorted(seen), domain=args.domain,
                          bu_pretrained=args.domain == 'BU4DFE' or provenance.get('bu_pretrained', False),
                          scratch=args.scratch, perceptual_checkpoint=args.perceptual_checkpoint)
        if cfg.tokenizer_weights['per']:
            if not args.perceptual_checkpoint:
                raise ValueError('Paper LPer requires --perceptual-checkpoint; no random feature substitution')
            perceptual = FrozenPerceptual(args.perceptual_checkpoint).to(device)
        stage_idx, classes = 0, CLASSES
    else:
        if not args.splits:
            raise ValueError('Generator/classifier need --splits')
        protocol = read_json(args.splits)
        if protocol['seed'] != args.seed:
            raise ValueError('--seed must match saved split seed')
        split = protocol['folds'][args.fold]
        verify_partitions(rows, split)
        classes = protocol['classes']
        part = split['classifier' if args.stage == 'classifier' else 'generator']
        train_ids, val_ids = part['train'], part['val']
        if args.stage == 'classifier':
            if any(sum(r['key'] in train_ids and r['group'] == c for r in rows) != cfg.shots for c in classes):
                raise ValueError('Classifier support must contain exactly four patients per class')
            if args.variant != 'pathological_only' and not (args.references or paired_root):
                raise ValueError('Classifier needs audited out-of-fold --references')
            model = DiseaseClassifier(cfg, len(classes), args.variant)
            stage_idx = 2
        elif args.stage == 'direct_regression':
            model, stage_idx = DirectRegression(cfg), 1
        else:
            if not args.tokenizer:
                raise ValueError('Generator requires --tokenizer checkpoint')
            tok, tok_cfg, tok_state = tokenizer_from(args.tokenizer, device)
            if asdict(tok_cfg) != asdict(cfg):
                raise ValueError('Use identical config for tokenizer and generator')
            provenance = tok_state['provenance']
            # Adaptation/initialization may never have seen held-out evaluation patients.
            heldout = set(split['generator']['test']+split['generator']['val']+
                          split['classifier']['test']+split['classifier']['val'])
            overlap = heldout & set(provenance['tokenizer_seen_ids'])
            if overlap:
                raise ValueError(f'Tokenizer leakage into held-out fold: {sorted(overlap)}')
            model = HealthyGenerator(cfg, not args.no_consistency, not args.no_resampling)
            stage_idx = 1
    model = model.to(device)
    views = cfg.views if args.stage in ('generator', 'direct_regression') else 1
    refs = args.references if args.stage == 'classifier' and args.variant != 'pathological_only' else None
    if paired_root:
        if args.references:
            raise ValueError('Choose --paired-data or --references, not both')
        provenance, _ = audit_paired_data(paired_root, rows, train_ids+val_ids, cfg, tuple(classes))
        reference_method = provenance['method']
        cache_pairs = getattr(args, 'cache_paired', False)
        if cache_pairs and args.workers:
            raise ValueError('--cache-paired requires --workers 0 to avoid duplicated caches')
        train_ds = PairedCineDataset(rows, train_ids, cfg, paired_root, classes, cache=cache_pairs)
        val_ds = PairedCineDataset(rows, val_ids, cfg, paired_root, classes, cache=cache_pairs)
    else:
        reference_method = audit_reference_method(rows, train_ids+val_ids, refs, args.seed) if refs else None
        train_ds = CineDataset(rows, train_ids, cfg, views, args.seed, refs, classes)
        val_ds = CineDataset(rows, val_ids, cfg, 1, args.seed, refs, classes)
    batch_size = cfg.batch_sizes[stage_idx]
    micro = args.micro_batch or batch_size
    if args.stage == 'classifier' and micro != batch_size:
        raise ValueError('Contrastive learning requires full batch=8; microbatching changes Eq. (21)')
    if batch_size % micro:
        raise ValueError('Effective batch must be divisible by micro-batch')
    accum = batch_size//micro
    dl = DataLoader(train_ds, batch_size=micro, shuffle=True, num_workers=args.workers)
    vl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.workers)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    amp = args.amp and device.type == 'cuda'
    # Patch softmax shrinks features at the native external grid; a high initial
    # FP16 scale can overflow the first backward pass with summed focal loss.
    # Dynamic scaling remains enabled; this changes no loss or batch semantics.
    scaler = torch.cuda.amp.GradScaler(enabled=amp, init_scale=128. if paired_root else 65536.)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    start, best = 0, float('inf')
    if args.resume:
        state = load_checkpoint(args.resume)
        if state['stage'] != args.stage or state['train_ids'] != train_ids or state['config'] != asdict(cfg):
            raise ValueError('Resume stage/config/patient split mismatch')
        if (state['val_ids'] != val_ids or state['seed'] != args.seed or state['fold'] != args.fold
                or state['variant'] != args.variant or state['consistency'] != (not args.no_consistency)
                or state['resampling'] != (not args.no_resampling)
                or state.get('reference_method') != reference_method):
            raise ValueError('Resume validation/seed/ablation/reference mismatch')
        if state.get('paired_data') != (str(Path(paired_root).resolve()) if paired_root else None):
            raise ValueError('Resume paired dataset mismatch')
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        start, best = state['epoch'], state['best_val']
        torch.set_rng_state(state['torch_rng'])
        if device.type == 'cuda' and state.get('cuda_rng'):
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    alpha = torch.ones(len(classes), device=device)  # four-shot: all classes have equal support

    def objective(batch):
        x = batch['cine'].to(device)
        if args.stage == 'tokenizer':
            recon, _, reg = model(x)
            losses = {'rec': nn.functional.l1_loss(recon, x), 'lfq': reg,
                      'per': perceptual(recon, x) if perceptual is not None else recon.sum()*0}
            return sum(v*cfg.tokenizer_weights[k] for k, v in losses.items()), losses
        if args.stage == 'generator':
            return model.objective(tok, x)
        if args.stage == 'direct_regression':
            loss = nn.functional.l1_loss(model(x[:, :1])[:, 1:], x[:, 1:])
            return loss, {'mae': loss}
        ref = batch.get('reference')
        diff = batch.get('difference')
        return model.objective(x, ref.to(device) if ref is not None else None, batch['label'].to(device), alpha,
                               difference=diff.to(device) if diff is not None else None)

    epochs = args.epochs or cfg.epochs[stage_idx]
    for epoch in range(start, epochs):
        model.train(); opt.zero_grad(set_to_none=True)
        epoch_loss, subjects = 0., 0
        for i, batch in enumerate(dl):
            # Correctly weight partial final accumulation groups by actual subjects.
            group_start = (i//accum)*accum*micro
            group_count = min(batch_size, len(train_ds)-group_start)
            n = len(batch['cine'])
            with torch.autocast(device_type=device.type, enabled=amp):
                loss, parts = objective(batch)
                scaled_loss = loss if args.stage == 'classifier' else loss*n/group_count
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at epoch {epoch+1}, batch {i}')
            scaler.scale(scaled_loss).backward()
            if (i+1) % accum == 0 or i+1 == len(dl):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            epoch_loss += float(loss.detach())*(1 if args.stage == 'classifier' else n)
            subjects += n
            if i % 10 == 0:
                print(f'epoch={epoch+1}/{epochs} batch={i+1}/{len(dl)} loss={float(loss.detach()):.5f}', flush=True)
        model.eval()
        with torch.no_grad():
            if args.stage == 'classifier':
                yy, pp = [], []
                for batch in vl:
                    ref = batch.get('reference')
                    diff = batch.get('difference')
                    logits, _, _ = model(batch['cine'].to(device), ref.to(device) if ref is not None else None,
                                         diff.to(device) if diff is not None else None)
                    yy.extend(batch['label'].tolist()); pp.extend(logits.softmax(-1).cpu().tolist())
                validation = 1-classification_metrics(yy, pp, classes)['macro_f1']
            else:
                val_losses = [float(objective(batch)[0]) for batch in vl]
                validation = float(np.mean(val_losses))
        if not np.isfinite(validation):
            raise FloatingPointError('Non-finite validation score; refusing to select checkpoint')
        improved = validation < best
        best = min(best, validation)
        state = {'stage': args.stage, 'config': asdict(cfg), 'model': model.state_dict(),
                 'optimizer': opt.state_dict(), 'scaler': scaler.state_dict(), 'epoch': epoch+1,
                 'best_val': best, 'train_ids': train_ids, 'val_ids': val_ids, 'seed': args.seed,
                 'fold': args.fold, 'classes': list(classes), 'variant': args.variant,
                 'consistency': not args.no_consistency, 'resampling': not args.no_resampling,
                 'tokenizer_checkpoint': args.tokenizer,
                 'tokenizer_sha256': file_hash(args.tokenizer) if args.tokenizer else None,
                 'provenance': provenance, 'torch_rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
                 'references': str(Path(refs).resolve()) if refs else None, 'reference_method': reference_method,
                 'paired_data': str(Path(paired_root).resolve()) if paired_root else None}
        torch.save(state, out/'last.pt')
        if improved:
            torch.save(state, out/'best.pt')
        report = {'epoch': epoch+1, 'train_loss': epoch_loss/subjects, 'validation': validation}
        with (out/'history.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(report)+'\n')
        print(report, flush=True)


@torch.no_grad()
def infer(args):
    rows = read_manifest(args.manifest)
    protocol = read_json(args.splits)
    split = protocol['folds'][args.fold]
    verify_partitions(rows, split)
    device = torch.device(args.device)
    seed_all(protocol['seed'])
    state, tok = {}, None
    if args.method == 'heartmirror':
        state = load_checkpoint(args.checkpoint)
        if state['stage'] != 'generator':
            raise ValueError('Expected generator checkpoint')
        tok, cfg, _ = tokenizer_from(args.tokenizer, device)
        if asdict(cfg) != state['config'] or file_hash(args.tokenizer) != state['tokenizer_sha256']:
            raise ValueError('Tokenizer/config mismatch with generator training')
        model = HealthyGenerator(cfg, state['consistency'], state['resampling']).to(device)
        model.load_state_dict(state['model'], strict=True); model.eval()
    elif args.method == 'direct_regression':
        state = load_checkpoint(args.checkpoint)
        if state['stage'] != 'direct_regression':
            raise ValueError('Expected direct-regression checkpoint')
        cfg = Config(**state['config'])
        model = DirectRegression(cfg).to(device)
        model.load_state_dict(state['model']); model.eval()
    else:
        cfg = Config.load(args.config)
    if state and (state['train_ids'] != split['generator']['train'] or state['val_ids'] != split['generator']['val']
                  or state['seed'] != protocol['seed'] or state['fold'] != args.fold):
        raise ValueError('Checkpoint belongs to a different split/seed/fold')
    method_tag = args.method
    if args.method == 'heartmirror':
        ablations = []
        if not state.get('provenance', {}).get('bu_pretrained', False):
            ablations.append('no_bu_pretraining')
        if not state['consistency']:
            ablations.append('no_consistency')
        if not state['resampling']:
            ablations.append('no_resampling')
        method_tag += '/' + ('+'.join(ablations) or 'full')
    # Each patient appears exactly once across five invocations. All references
    # can then be shared by classifier variants with the exact same support set.
    ids = split['classifier']['test']
    selected = [r for r in rows if r['key'] in ids]
    for r in selected:
        x, spatial = load_cine(r, (cfg.phases, cfg.slices, cfg.image_size, cfg.image_size))
        first = torch.from_numpy(x[:1])[None].to(device)
        if args.method == 'heartmirror':
            ref, _, _ = model.sample(tok, first, args.greedy)
        elif args.method == 'direct_regression':
            ref = model(first)
        else:
            ref = first.expand(-1, cfg.phases, -1, -1, -1)
        ref = ref[0].cpu().numpy().astype(np.float32)
        metadata = dict(spatial, patient_key=r['key'], group=r['group'], seed=protocol['seed'], fold=args.fold,
                        method=method_tag, config=asdict(cfg), generator_train_ids=state.get('train_ids', []),
                        generator_val_ids=state.get('val_ids', []), checkpoint=args.checkpoint,
                        checkpoint_sha256=file_hash(args.checkpoint) if args.checkpoint else None,
                        tokenizer_seen_ids=state.get('provenance', {}).get('tokenizer_seen_ids', []))
        folder = Path(args.out)/r['dataset']; folder.mkdir(parents=True, exist_ok=True)
        dest = folder/(r['patient_id']+'.npz')
        if dest.exists() and not args.overwrite:
            raise FileExistsError(f'{dest}: use --overwrite intentionally')
        np.savez_compressed(dest, cine=ref, metadata=json.dumps(metadata))
        if args.nifti:
            import nibabel as nib
            image = nib.Nifti1Image(ref.transpose(2, 3, 1, 0), np.asarray(spatial['affine']))
            zooms = list(image.header.get_zooms()); zooms[3] = spatial['temporal_spacing']
            image.header.set_zooms(zooms)
            image.header.set_xyzt_units(*spatial['xyzt_units'])
            nib.save(image, folder/(r['patient_id']+'_gen.nii.gz'))
        print(f'Saved {r["key"]}', flush=True)


@torch.no_grad()
def evaluate(args):
    if args.task != 'classification' and (getattr(args, 'paired_data', None) or not args.references):
        raise ValueError('Generation evaluation requires --references; --paired-data is classification-only')
    rows = read_manifest(args.manifest)
    protocol = read_json(args.splits)
    device = torch.device(args.device)
    if args.task == 'classification':
        state = load_checkpoint(args.checkpoint)
        if state['stage'] != 'classifier' or state['seed'] != protocol['seed'] or state['fold'] != args.fold:
            raise ValueError('Classifier checkpoint/split mismatch')
        cfg = Config(**state['config'])
        classes = state['classes']
        split = protocol['folds'][args.fold]; verify_partitions(rows, split)
        if state['train_ids'] != split['classifier']['train'] or state['val_ids'] != split['classifier']['val']:
            raise ValueError('Classifier support/validation mismatch')
        model = DiseaseClassifier(cfg, len(classes), state['variant']).to(device)
        model.load_state_dict(state['model']); model.eval()
        paired_root = getattr(args, 'paired_data', None) or state.get('paired_data')
        refs = None if state['variant'] == 'pathological_only' else args.references
        if state.get('paired_data'):
            if args.references or str(Path(paired_root).resolve()) != state['paired_data']:
                raise ValueError('Evaluation must use the training paired dataset')
            ds = PairedCineDataset(rows, split['classifier']['test'], cfg, paired_root, classes)
        else:
            if paired_root:
                raise ValueError('Checkpoint was not trained with --paired-data')
            if refs and audit_reference_method(rows, split['classifier']['test'], refs, protocol['seed']) != state['reference_method']:
                raise ValueError('Test reference method differs from classifier training')
            ds = CineDataset(rows, split['classifier']['test'], cfg, seed=protocol['seed'], reference_dir=refs, classes=classes)
        labels, probs, keys = [], [], []
        for batch in DataLoader(ds, batch_size=1):
            ref = batch.get('reference')
            diff = batch.get('difference')
            logits, _, _ = model(batch['cine'].to(device), ref.to(device) if ref is not None else None,
                                 diff.to(device) if diff is not None else None)
            probs.extend(logits.softmax(-1).cpu().tolist()); labels.extend(batch['label'].tolist()); keys.extend(batch['key'])
        report = {'seed': protocol['seed'], 'fold': args.fold, 'task': args.task, 'classes': classes,
                  'dataset': protocol['dataset'], 'method': state['reference_method'], 'variant': state['variant'],
                  'patient_ids': keys, 'labels': labels, 'probabilities': probs,
                  'metrics': classification_metrics(labels, probs, classes),
                  'paired_data': state.get('paired_data'), 'provenance': state.get('provenance', {})}
    else:
        expected = {k for f in protocol['folds'] for k in f['generator']['test']}
        reference_method = audit_reference_method(rows, expected, args.references, protocol['seed'])
        per_patient, real_fid, gen_fid, real_fvd, gen_fvd = [], [], [], [], []
        lpips = None
        if args.lpips:
            import lpips as lpips_module
            lpips = lpips_module.LPIPS(net=args.lpips_backbone).to(device).eval()
        fid_model = torch.jit.load(args.fid_model, map_location=device).eval() if args.fid_model else None
        fvd_model = torch.jit.load(args.fvd_model, map_location=device).eval() if args.fvd_model else None
        # Feature adapters must implement documented [0,1] RGB contracts.
        for r in rows:
            if r['key'] not in expected:
                continue
            path = Path(args.references)/r['dataset']/(r['patient_id']+'.npz')
            with np.load(path, allow_pickle=False) as f:
                gen, meta = f['cine'], json.loads(str(f['metadata']))
            if meta['patient_key'] != r['key'] or meta['seed'] != protocol['seed']:
                raise ValueError('Reference identity/seed mismatch')
            fold = protocol['folds'][meta['fold']]['generator']
            if r['key'] not in fold['test']:
                raise ValueError('Reference not generated by held-out fold')
            if meta['method'] != 'ed_repeat' and (meta['generator_train_ids'] != fold['train'] or meta['generator_val_ids'] != fold['val']):
                raise ValueError('Reference generator patient provenance mismatch')
            if r['key'] in meta.get('tokenizer_seen_ids', []):
                raise ValueError('Evaluation patient seen by tokenizer')
            cfg = Config(**meta['config'])
            real, _ = load_cine(r, (cfg.phases, cfg.slices, cfg.image_size, cfg.image_size))
            if not np.allclose(real[0], gen[0], atol=1e-5):
                raise ValueError('Conditioning ED changed')
            score = paired_metrics(real, gen)
            score.update(patient_id=r['key'], fold=meta['fold'], lpips=None)
            a, b = (torch.from_numpy(v[1:].copy()).to(device) for v in (real, gen))
            af, bf = (v.flatten(0, 1)[:, None].expand(-1, 3, -1, -1) for v in (a, b))
            if lpips is not None:
                vals = [lpips(aa*2-1, bb*2-1).flatten().cpu() for aa, bb in zip(af.split(16), bf.split(16))]
                score['lpips'] = float(torch.cat(vals).mean())
            if fid_model is not None:
                for src, dest in ((af, real_fid), (bf, gen_fid)):
                    features = torch.cat([fid_model(c).flatten(1).cpu() for c in src.split(16)]).numpy()
                    if features.shape[1] != 2048:
                        raise ValueError('FID adapter must return pool3 2048-D features')
                    dest.append(features)
            if fvd_model is not None:
                for src, dest in ((a, real_fvd), (b, gen_fvd)):
                    clips = src.permute(1, 0, 2, 3)[:, None].expand(-1, 3, -1, -1, -1)
                    features = fvd_model(clips).flatten(1).cpu().numpy()
                    if features.shape[1] != 1024:
                        raise ValueError('FVD adapter must return I3D 1024-D features, not frame Inception')
                    dest.append(features)
            per_patient.append(score)
        if len(per_patient) != len(expected) or not per_patient:
            raise ValueError('Missing OOF healthy patients')
        metrics = {k: float(np.mean([r[k] for r in per_patient])) for k in ('ssim', 'mse')}
        metrics.update(lpips=float(np.mean([r['lpips'] for r in per_patient])) if lpips else None,
                       fid=frechet(np.concatenate(real_fid), np.concatenate(gen_fid)) if real_fid else None,
                       fvd=frechet(np.concatenate(real_fvd), np.concatenate(gen_fvd)) if real_fvd else None)
        report = {'seed': protocol['seed'], 'task': args.task, 'per_patient': per_patient, 'metrics': metrics,
                  'dataset': protocol['dataset'], 'method': reference_method, 'variant': 'generation',
                  'feature_models': {k: {'path': v, 'sha256': file_hash(v)} if v else None
                                     for k, v in [('fid', args.fid_model), ('fvd', args.fvd_model)]},
                  'missing_metrics': [k for k, v in metrics.items() if v is None],
                  'lpips_backbone': args.lpips_backbone if lpips is not None else None,
                  'protocol': 'ED excluded; paired metrics averaged within patient then across patients; slice-wise I3D FVD'}
    write_json(args.out, report)
    print(json.dumps(report['metrics'], indent=2))


def aggregate(args):
    reports = [read_json(p) for p in args.reports]
    identity = {k: reports[0].get(k) for k in ('dataset', 'method', 'variant', 'task')}
    if any(any(r.get(k) != v for k, v in identity.items()) for r in reports):
        raise ValueError('Cannot aggregate different methods/datasets/ablation variants/tasks')
    if args.folds:
        if len(reports) != 5 or {r['fold'] for r in reports} != set(range(5)) or len({r['seed'] for r in reports}) != 1:
            raise ValueError('Need all five folds of one seed')
        keys = sum([r['patient_ids'] for r in reports], [])
        if len(keys) != len(set(keys)):
            raise ValueError('Patient appears in multiple test folds')
        classes = reports[0]['classes']
        if any(r['classes'] != classes for r in reports):
            raise ValueError('Class order mismatch')
        result = {'seed': reports[0]['seed'], 'metrics': classification_metrics(
            sum([r['labels'] for r in reports], []), sum([r['probabilities'] for r in reports], []), classes)}
    else:
        result = aggregate_runs(reports)
    result.update(identity)
    write_json(args.out, result)


def visualize(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = read_manifest(args.manifest)
    row = next(r for r in rows if r['key'] == args.patient)
    with np.load(Path(args.references)/row['dataset']/(row['patient_id']+'.npz'), allow_pickle=False) as f:
        gen, meta = f['cine'], json.loads(str(f['metadata']))
    cfg = Config(**meta['config'])
    real, _ = load_cine(row, (cfg.phases, cfg.slices, cfg.image_size, cfg.image_size))
    z = cfg.slices//2
    fig, ax = plt.subplots(2, 4, figsize=(12, 6))
    for j, phase in enumerate((9, 19)):
        delta = np.abs(real[phase, z]-gen[phase, z])
        for k, (v, name) in enumerate(zip((real[phase, z], gen[phase, z], delta, delta),
                                         ('Observed', 'Healthy reference', 'Absolute deviation', 'Heatmap'))):
            ax[j, k].imshow(v, cmap='hot' if k == 3 else 'gray', vmin=0, vmax=1)
            ax[j, k].set_title(f'{name}, t={phase+1}'); ax[j, k].axis('off')
    fig.tight_layout(); Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='HeartMirror paper implementation')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.set_defaults(func=prepare)
    p.add_argument('--acdc-root'); p.add_argument('--manifest'); p.add_argument('--dataset', default='ACDC')
    p.add_argument('--classes', default=','.join(CLASSES)); p.add_argument('--config'); p.add_argument('--out', required=True)
    p = sub.add_parser('train'); p.set_defaults(func=train)
    p.add_argument('stage', choices=['tokenizer', 'generator', 'classifier', 'direct_regression'])
    p.add_argument('--manifest', required=True); p.add_argument('--config'); p.add_argument('--splits')
    p.add_argument('--fold', type=int, choices=range(5), default=0); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', required=True); p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--train-ids'); p.add_argument('--val-ids'); p.add_argument('--domain')
    p.add_argument('--init'); p.add_argument('--scratch', action='store_true'); p.add_argument('--perceptual-checkpoint')
    p.add_argument('--tokenizer'); p.add_argument('--references'); p.add_argument('--paired-data'); p.add_argument('--resume')
    p.add_argument('--variant', default='full', choices=['full', 'pathological_only', 'no_reweighting', 'constant_weighting', 'no_contrastive', 'no_cross_attention'])
    p.add_argument('--no-consistency', action='store_true'); p.add_argument('--no-resampling', action='store_true')
    p.add_argument('--epochs', type=int); p.add_argument('--micro-batch', type=int); p.add_argument('--workers', type=int, default=0)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--cache-paired', action='store_true', help='Cache validated observed/reference tensors in RAM')
    p = sub.add_parser('infer'); p.set_defaults(func=infer)
    p.add_argument('--manifest', required=True); p.add_argument('--splits', required=True); p.add_argument('--fold', type=int, choices=range(5), default=0)
    p.add_argument('--method', choices=['heartmirror', 'ed_repeat', 'direct_regression'], default='heartmirror')
    p.add_argument('--checkpoint'); p.add_argument('--tokenizer'); p.add_argument('--config'); p.add_argument('--out', required=True)
    p.add_argument('--greedy', action='store_true'); p.add_argument('--nifti', action='store_true'); p.add_argument('--overwrite', action='store_true')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p = sub.add_parser('evaluate'); p.set_defaults(func=evaluate)
    p.add_argument('task', choices=['generation', 'classification'])
    p.add_argument('--manifest', required=True); p.add_argument('--splits', required=True); p.add_argument('--references'); p.add_argument('--paired-data')
    p.add_argument('--checkpoint'); p.add_argument('--fold', type=int, choices=range(5), default=0); p.add_argument('--out', required=True)
    p.add_argument('--lpips', action='store_true'); p.add_argument('--fid-model'); p.add_argument('--fvd-model')
    p.add_argument('--lpips-backbone', choices=['alex', 'vgg', 'squeeze'], default='vgg')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p = sub.add_parser('aggregate'); p.set_defaults(func=aggregate)
    p.add_argument('reports', nargs='+'); p.add_argument('--folds', action='store_true'); p.add_argument('--out', required=True)
    p = sub.add_parser('visualize'); p.set_defaults(func=visualize)
    p.add_argument('--manifest', required=True); p.add_argument('--references', required=True)
    p.add_argument('--patient', required=True); p.add_argument('--out', required=True)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
