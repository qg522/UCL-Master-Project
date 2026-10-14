"""Explicit table-to-run mapping. External methods require their own implementation."""
EXPERIMENTS = {
    'table1': {
        'datasets': ['ACDC', 'MMs'],
        'methods': ['ed_repeat', 'direct_regression', 'Mo-Diff', 'VTG', 'EchoVDiff', 'heartmirror'],
        'generation_metrics': ['fid', 'fvd', 'ssim', 'lpips'],
        'classification_metrics': ['accuracy', 'macro_auc', 'macro_f1', 'rv_recall'],
        'runs': 5, 'shots': 4,
    },
    'table2': {
        'dataset': 'ACDC',
        'variants': {'no_bu_pretraining': {'tokenizer_init': 'scratch_on_MMs'},
                     'no_consistency': {'consistency': False},
                     'no_resampling': {'resampling': False}, 'full': {}},
    },
    'table3': {
        'dataset': 'ACDC',
        'variants': ['pathological_only', 'no_reweighting', 'constant_weighting',
                     'no_contrastive', 'no_cross_attention', 'full'],
    },
    'table4': {
        'methods': ['GenBiomarker', 'AR-SIVAE', 'CPGG', 'CineMA', 'MeshHeart', 'heartmirror'],
        'external_methods_required': True,
    },
    'table5': {
        'references': ['Mo-Diff_pathological', 'Mo-Diff_healthy', 'heartmirror_healthy'],
        'same_classifier_and_splits': True,
    },
    'figure2': {'phases_one_based': [10, 20], 'outputs': ['observed', 'reference', 'absolute_difference', 'heatmap']},
}
