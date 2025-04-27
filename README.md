# Transformer-Based Text Summarization

## Project Overview
This project implements a PyTorch transformer model for abstractive text summarization, trained on the CNN/DailyMail dataset. The primary focus is investigating the impact of various regularization techniques on model performance.

## Key Features
- Built-in PyTorch transformer architecture
- Advanced optimization with AdamW and learning rate scheduling
- Comprehensive evaluation using ROUGE metrics
- Modular design for easy experimentation

## Model Configuration
### Basic Configuration
```python
config = {
    "vocab_size": 20000,          # Size of vocabulary (BPE tokens)
    "dropout": 0.1,               # Dropout rate for regularization
    "d_model": 512,               # Embedding dimension
    "nhead": 8,                   # Number of attention heads
    "num_encoder_layers": 4,      # Encoder stack depth
    "num_decoder_layers": 4,      # Decoder stack depth  
    "dim_feedforward": 2048       # FFN layer dimension
}
```

### Training Parameters
| Parameter              | Value                                  |
|------------------------|----------------------------------------|
| Epochs                 | 5                                      |
| Batch Size             | 64                                     |
| Workers                | 2                                      |
| Optimizer              | AdamW (lr=2e-4, weight_decay=0.001)   |
| Loss Function          | Cross-Entropy (ignoring padding)       |
| Gradient Clipping      | 2.0 norm                              |

### Learning Rate Scheduling
- **Warmup**: Linear increase from 5e-5 to 2e-4 over first epoch
- **Plateau**: Halves learning rate when validation improvement <1% for 2 epochs (min lr=1e-5)

### Teacher Forcing
- Initial ratio: 0.9
- Linear decay to 0.1 by epoch 8

## Evaluation Metrics
We use ROUGE F1 scores for evaluation:
- **ROUGE-1**: Unigram overlap (topic coverage)
- **ROUGE-2**: Bigram overlap (fluency)
- **ROUGE-L**: Longest common subsequence (coherence)

## Project Structure
```text
.
├── cnn_bpe_tokenizer_20k.json           # Generated 20K tokens with BPE tokenization
├── baseline_transformer_architecture.py  # Transformer model definition
├── data_utils.py                        # Data pipelines and loaders
├── optimizer_scheduler.py               # Training schedules and optimizers
├── modeling_functions.py                # Core training/inference logic
├── baseline_model_run.py                # Sample baseline code execution
├── visualization.py                    # Performance metrics plotting
├── inference_run.py                    # Sample inference ROUGE generation
└── baseline_outputs/                    # Example generated results
```

## Module Documentation
### `baseline_transformer_architecture.py`
- `create_small_transformer()`: Instantiates the baseline transformer model

### `data_utils.py`
- `set_seed(42)`: The function to set the global seed for reproducibility
- `get_train_loader()`: Training data loader
- `get_val_loader()`: Validation data loader
- `get_test_loader()`: Test data loader

### `optimizer_scheduler.py`
- AdamW optimizer configuration
- Learning rate schedulers (plateau)
- Teacher forcing ratio scheduler

### `modeling_functions.py`
**Training Functions:**
- `train_transformer_teacher_forcing()`: Main training loop that includes the warmup scheduler under 1 epoch.
- `validate_transformer()`: Validation with ROUGE scoring
- `greedy_decode()`: Training decoding strategy

**Inference Functions:**
- `generate_with_beam_search()`: Beam search decoding (beam=3)
- `calculate_and_save_rouge()`: Final evaluation

### `visualization.py`
- `visualize_training_dynamics()`: Generates training/validation plots. 
- You need to adjust the file names for result loading and visualization saving.
- Supports both script execution and notebook import.

### `inference_run.py`
- Shows how to run inference with `generate_with_beam_search` on `get_test_loader()`.
- You need to adjust the file names for model loading and result saving.

## Getting Started
1. Run the baseline model
```bash
python baseline_model_run.py
```
2. Run visualization
```bash
python visualization.py
```
## Package Dependencies
Create conda environment then install the following:
```bash
conda create -n torch_env python=3.9 -y
conda activate torch_env
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 cudatoolkit=12.1 -c pytorch -c nvidia
conda install -c conda-forge numpy matplotlib tokenizers datasets
pip install rouge-score transformers tqdm
```