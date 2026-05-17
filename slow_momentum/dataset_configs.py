"""
Dataset Configuration
Define dataset-specific settings
"""

class DatasetConfig:
    """Configuration for different datasets"""
    
    CONFIGS = {
        'boston': {
            'name': 'Boston Housing',
            'input_dim': 13,
            'output_dim': 1,
            'task': 'regression',
            'models': ['linear', 'mlp_shallow', 'mlp_deep'],
            'val_split': 0.2,
            'test_split': 0.2,
            'optimization': {'n_trials': 50, 'num_epochs': 100},
            'final_training': {'num_epochs': 700, 'num_runs': 10}
        },
        
        'california': {
            'name': 'California Housing',
            'input_dim': 8,
            'output_dim': 1,
            'task': 'regression',
            'models': ['linear', 'mlp_shallow', 'mlp_deep'],
            'val_split': 0.2,
            'test_split': 0.2,
            'optimization': {'n_trials': 50, 'num_epochs': 100},
            'final_training': {'num_epochs': 700, 'num_runs': 10}
        },
        
        'diabetes': {
            'name': 'Diabetes',
            'input_dim': 10,
            'output_dim': 1,
            'task': 'regression',
            'models': ['linear', 'mlp_shallow', 'mlp_deep'],
            'val_split': 0.2,
            'test_split': 0.2,
            'optimization': {'n_trials': 50, 'num_epochs': 100},
            'final_training': {'num_epochs': 700, 'num_runs': 10}
        },
        
        'concrete': {
            'name': 'Concrete Strength',
            'input_dim': 8,
            'output_dim': 1,
            'task': 'regression',
            'models': ['linear', 'mlp_shallow', 'mlp_deep'],
            'val_split': 0.2,
            'test_split': 0.2,
            'optimization': {'n_trials': 50, 'num_epochs': 100},
            'final_training': {'num_epochs': 700, 'num_runs': 10}
        },
        
        'energy': {
            'name': 'Energy Efficiency',
            'input_dim': 8,
            'output_dim': 2,  # Heating and Cooling
            'task': 'regression',
            'models': ['linear', 'mlp_shallow', 'mlp_deep'],
            'val_split': 0.2,
            'test_split': 0.2,
            'optimization': {'n_trials': 50, 'num_epochs': 100},
            'final_training': {'num_epochs': 700, 'num_runs': 10}
        },
        
        'kin8nm': {
            'name': 'Kin8nm',
            'input_dim': 8,
            'output_dim': 1,
            'task': 'regression',
            'models': ['linear', 'mlp_shallow', 'mlp_deep'],
            'val_split': 0.2,
            'test_split': 0.2,
            'optimization': {'n_trials': 50, 'num_epochs': 100},
            'final_training': {'num_epochs': 700, 'num_runs': 10}
        },
        
        'naval': {
            'name': 'Naval Propulsion',
            'input_dim': 16,
            'output_dim': 1,
            'task': 'regression',
            'models': ['linear', 'mlp_shallow', 'mlp_deep'],
            'val_split': 0.2,
            'test_split': 0.2,
            'optimization': {'n_trials': 50, 'num_epochs': 100},
            'final_training': {'num_epochs': 700, 'num_runs': 10}
        },
        
        'protein': {
            'name': 'Protein Structure',
            'input_dim': 9,
            'output_dim': 1,
            'task': 'regression',
            'models': ['linear', 'mlp_shallow', 'mlp_deep'],
            'val_split': 0.2,
            'test_split': 0.2,
            'optimization': {'n_trials': 50, 'num_epochs': 100},
            'final_training': {'num_epochs': 700, 'num_runs': 10}
        },

        'year_prediction_msd': {
            'name': 'Year Prediction MSD',
            'input_dim': 90,
            'output_dim': 1,
            'task': 'regression',
            'models': ['linear', 'mlp_shallow', 'mlp_deep'],
            'val_split': 0.1,  # Smaller split due to large dataset
            'test_split': 0.1,
            'optimization': {
                'n_trials': 50,
                'num_epochs': 100,  # Fewer epochs due to size
                'timeout': None
            },
            'final_training': {
                'num_epochs': 700,
                'num_runs': 10  # Fewer runs due to size
            }
        },
        
        'ct_slices': {
            'name': 'CT Slices',
            'input_dim': 379,
            'output_dim': 1,
            'task': 'regression',
            'models': ['linear', 'mlp_shallow', 'mlp_deep'],
            'val_split': 0.15,
            'test_split': 0.15,
            'optimization': {
                'n_trials': 50,
                'num_epochs': 100,
                'timeout': None
            },
            'final_training': {
                'num_epochs': 700,
                'num_runs': 10
            }
        },
        
        'bike_sharing': {
            'name': 'Bike Sharing',
            'input_dim': 16,
            'output_dim': 1,
            'task': 'regression',
            'models': ['linear', 'mlp_shallow', 'mlp_deep'],
            'val_split': 0.2,
            'test_split': 0.2,
            'optimization': {
                'n_trials': 50,
                'num_epochs': 100,
                'timeout': None
            },
            'final_training': {
                'num_epochs': 700,
                'num_runs': 10
            }
        },
        
        'wine_quality': {
            'name': 'Wine Quality',
            'input_dim': 11,
            'output_dim': 1,
            'task': 'regression',
            'models': ['linear', 'mlp_shallow', 'mlp_deep'],
            'val_split': 0.2,
            'test_split': 0.2,
            'optimization': {
                'n_trials': 50,
                'num_epochs': 100,
                'timeout': None
            },
            'final_training': {
                'num_epochs': 700,
                'num_runs': 10
            }
        },
        
        'appliances_energy': {
            'name': 'Appliances Energy',
            'input_dim': 28,
            'output_dim': 1,
            'task': 'regression',
            'models': ['linear', 'mlp_shallow', 'mlp_deep'],
            'val_split': 0.2,
            'test_split': 0.2,
            'optimization': {
                'n_trials': 50,
                'num_epochs': 100,
                'timeout': None
            },
            'final_training': {
                'num_epochs': 700,
                'num_runs': 10
            }
        },
        
        'power_plant': {
            'name': 'Power Plant',
            'input_dim': 4,
            'output_dim': 1,
            'task': 'regression',
            'models': ['linear', 'mlp_shallow', 'mlp_deep'],
            'val_split': 0.2,
            'test_split': 0.2,
            'optimization': {
                'n_trials': 50,
                'num_epochs': 100,
                'timeout': None
            },
            'final_training': {
                'num_epochs': 700,
                'num_runs': 10
            }
        }
    }
    
    @classmethod
    def get_config(cls, dataset_name):
        """Get configuration for specific dataset"""
        if dataset_name not in cls.CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return cls.CONFIGS[dataset_name]
    
    @classmethod
    def get_available_datasets(cls):
        """Get list of available datasets"""
        return list(cls.CONFIGS.keys())