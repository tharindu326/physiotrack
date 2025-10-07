import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_default_config() -> Dict[str, Any]:
    # Get the configs directory relative to this file
    current_dir = Path(__file__).parent.parent
    configs_dir = current_dir / "configs"
    base_config_path = configs_dir / "base_config.yaml"
    
    if base_config_path.exists():
        return load_config(str(base_config_path))
    else:
        # Fallback default config if file doesn't exist
        return {
            'model': {
                'input_joints': 17,
                'encoder_type': 'mlp',
                'rotation_type': '6d',
                'hidden_dim': 512,
                'encoder_output_dim': 256,
                'dropout': 0.1
            },
            'training': {
                'learning_rate': 1e-4,
                'batch_size': 32,
                'num_epochs': 100,
                'pose_weight': 1.0,
                'rotation_weight': 1.0
            }
        }


def load_model_config(encoder_type: str) -> Dict[str, Any]:
    base_config = get_default_config()
    
    # Get model-specific config
    current_dir = Path(__file__).parent.parent
    configs_dir = current_dir / "configs"
    model_config_path = configs_dir / f"{encoder_type}_config.yaml"
    
    if model_config_path.exists():
        model_config = load_config(str(model_config_path))
        return merge_configs(base_config, model_config)
    else:
        # Return base config if specific config doesn't exist
        base_config['model']['encoder_type'] = encoder_type
        return base_config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def validate_config(config: Dict[str, Any]) -> bool:
    required_fields = {
        'model': ['input_joints', 'encoder_type', 'rotation_type'],
        'training': ['learning_rate', 'batch_size']
    }
    for section, fields in required_fields.items():
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
        
        for field in fields:
            if field not in config[section]:
                raise ValueError(f"Missing required field: {section}.{field}")
    
    # Validate encoder type
    valid_encoders = ['mlp', 'gcn', 'transformer', 'hybrid', 'hybrid_simple']
    if config['model']['encoder_type'] not in valid_encoders:
        raise ValueError(f"Invalid encoder_type. Must be one of: {valid_encoders}")
    
    # Validate rotation type
    valid_rotations = ['6d', 'quaternion', 'matrix']
    if config['model']['rotation_type'] not in valid_rotations:
        raise ValueError(f"Invalid rotation_type. Must be one of: {valid_rotations}")
    
    # Validate predict_mode if present
    if 'predict_mode' in config.get('model', {}):
        valid_predict_modes = ['rotation_only', 'rotation_plus_residual']
        if config['model']['predict_mode'] not in valid_predict_modes:
            raise ValueError(f"Invalid predict_mode. Must be one of: {valid_predict_modes}")
    
    # Validate split setting if present
    if 'data' in config and 'split' in config['data'] and 'setting' in config['data']['split']:
        valid_split_settings = ['S1', 'S2', 'S3']
        split_setting = config['data']['split']['setting']
        if split_setting not in valid_split_settings:
            raise ValueError(f"Invalid split setting. Must be one of: {valid_split_settings}")
    
    # Validate ratios are in valid range
    if 'data' in config and 'split' in config['data']:
        split_config = config['data']['split']
        if 's1_train_ratio' in split_config:
            ratio = split_config['s1_train_ratio']
            if not (0.0 < ratio < 1.0):
                raise ValueError(f"s1_train_ratio must be between 0 and 1, got {ratio}")
        if 's1_val_ratio_from_train' in split_config:
            ratio = split_config['s1_val_ratio_from_train']
            if not (0.0 <= ratio < 1.0):
                raise ValueError(f"s1_val_ratio_from_train must be between 0 and 1, got {ratio}")
    
    return True


class ConfigManager:
    def __init__(self, config_path: Optional[str] = None):
        if config_path:
            self.config = load_config(config_path)
        else:
            self.config = get_default_config()
        validate_config(self.config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        self.config = merge_configs(self.config, updates)
        validate_config(self.config)
    
    def save(self, save_path: str) -> None:
        save_config(self.config, save_path)
    
    def to_dict(self) -> Dict[str, Any]:
        return self.config.copy()