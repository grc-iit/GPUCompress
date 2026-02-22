"""Configuration space constants and action encoding for the compression predictor."""

ALGORITHM_NAMES = ['lz4', 'snappy', 'deflate', 'gdeflate',
                   'zstd', 'ans', 'cascaded', 'bitcomp']

SHUFFLE_OPTIONS = [0, 4]

QUANT_OPTIONS = [
    (False, 0.0),      # lossless
    (True, 0.1),
    (True, 0.01),
    (True, 0.001),
]

NUM_ALGORITHMS = 8
NUM_CONFIGS_NN = 32    # 8 algo × 2 quant × 2 shuffle (NN action space)
NUM_CONFIGS_FULL = 64  # 8 algo × 2 shuffle × 4 quant (training space)


def decode_action(action_id: int) -> dict:
    """Decode NN action ID (0-31) to config dict."""
    return {
        'algorithm': action_id % 8,
        'quantization': (action_id // 8) % 2,
        'shuffle': (action_id // 16) % 2,
    }


def build_all_config_features(entropy, mad, second_derivative,
                               data_size, error_bounds):
    """Build feature vectors for all 64 training configs.

    Returns list of (feature_dict, algo_name, shuffle, quant_str, error_bound) tuples.
    Used by predict.py and xgb_predict.py to avoid duplicating feature construction.
    """
    import numpy as np
    import math

    configs = []
    for algo_idx, algo_name in enumerate(ALGORITHM_NAMES):
        for shuffle in SHUFFLE_OPTIONS:
            for quant, eb in QUANT_OPTIONS:
                features = {}
                for i, a in enumerate(ALGORITHM_NAMES):
                    features[f'alg_{a}'] = 1.0 if i == algo_idx else 0.0
                features['quant_enc'] = 1.0 if quant else 0.0
                features['shuffle_enc'] = 1.0 if shuffle > 0 else 0.0
                eb_val = eb if quant else (error_bounds if error_bounds > 0 else 1e-7)
                features['error_bound_enc'] = math.log10(max(eb_val, 1e-7))
                features['data_size_enc'] = math.log2(max(data_size, 1))
                features['entropy'] = entropy
                features['mad'] = mad
                features['second_derivative'] = second_derivative
                quant_str = 'linear' if quant else 'none'
                configs.append((features, algo_name, shuffle, quant_str, eb))
    return configs
