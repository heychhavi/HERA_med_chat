# Project Structure

```
HERA_med_chat/
├── configs/                    # Configuration files
│   └── training_config.py     # Training hyperparameters and settings
│
├── docs/                      # Documentation
│   └── PROJECT_STRUCTURE.md   # This file
│
├── notebooks/                 # Jupyter notebooks
│   ├── Gemma2_ft_medqa_3k.ipynb           # Stage 1: MedQA fine-tuning
│   ├── Gemma_Healthcaremagic_ft.ipynb     # Stage 2: HealthCareMagic fine-tuning
│   ├── Model_merge.ipynb                  # Model merging utilities
│   └── ollama.ipynb                       # Ollama integration
│
├── src/                       # Source code
│   ├── benchmarks/           # Benchmarking code
│   │   ├── medical_benchmark.py    # Medical evaluation scripts
│   │   └── verified_benchmark.py   # Verification utilities
│   │
│   ├── models/               # Model-related code
│   │   └── (model checkpoints)
│   │
│   ├── data/                 # Data processing code
│   │   └── (data processing scripts)
│   │
│   └── utils/                # Utility functions
│       └── (helper functions)
│
├── tests/                    # Test files
│   └── (test files)
│
├── .gitignore               # Git ignore file
├── README.md                # Project documentation
└── requirements.txt         # Project dependencies
```

## Directory Descriptions

### `configs/`
Contains configuration files for training and model parameters.

### `docs/`
Project documentation and guides.

### `notebooks/`
Jupyter notebooks for experimentation, training, and analysis.

### `src/`
Main source code directory:
- `benchmarks/`: Code for evaluating model performance
- `models/`: Model-related code and checkpoints
- `data/`: Data processing and preparation scripts
- `utils/`: Helper functions and utilities

### `tests/`
Unit tests and integration tests.

## File Naming Conventions

- Python files: Use snake_case (e.g., `medical_benchmark.py`)
- Notebooks: Use descriptive names with underscores (e.g., `Gemma2_ft_medqa_3k.ipynb`)
- Configuration files: Use descriptive names with underscores (e.g., `training_config.py`)

## Best Practices

1. Keep notebooks focused on specific tasks
2. Use relative imports within the `src` directory
3. Document all functions and classes
4. Write tests for critical functionality
5. Keep configuration separate from code
6. Use version control for all files 