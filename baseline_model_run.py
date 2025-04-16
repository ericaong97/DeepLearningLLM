# import libraries and modules
import torch
from data_utils import (
    get_train_loader,
    get_val_loader,
    set_seed
)
from baseline_transformer_architecture import create_small_transformer

from modeling_vis_functions import (
    train_transformer_teacher_forcing,
    TeacherForcingScheduler
)

from optimizer_scheduler import (
    get_optimizer, get_plateau_scheduler
)
from tokenizers import Tokenizer

# ============================================================================

# 1. Setting a global seed and device to use
set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Define model configuration 
config = {
    "vocab_size": 20000,
    "dropout": 0.1,
    "max_len": 512,
    "d_model": 512,
    "nhead": 8,
    "num_encoder_layers": 4,
    "num_decoder_layers": 4,
    "dim_feedforward": 2048
}


# 2. load the tokenizer
tokenizer_20 = Tokenizer.from_file("cnn_bpe_tokenizer_20k.json")


# 3. Initialize model
base_model = create_small_transformer(d_model=config['d_model'],
                                nhead=config['nhead'],
                                num_decoder_layers=config['num_decoder_layers'],
                                num_encoder_layers=config['num_encoder_layers'],
                                dim_feedforward=config['dim_feedforward'],
                                dropout=config['dropout'],
                                vocab_size=config['vocab_size']).to(device)

# 4. Data loading
train_loader = get_train_loader(tokenizer=tokenizer_20)
val_loader = get_val_loader(tokenizer=tokenizer_20)

# 5. Define criterion
# ignore the padding index
transformer_criterion = torch.nn.CrossEntropyLoss(ignore_index=1)

# 6. Training Loop
history = train_transformer_teacher_forcing(
    model=base_model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=get_optimizer(base_model),
    criterion=transformer_criterion,
    plateau_scheduler=get_plateau_scheduler(get_optimizer(base_model)),
    teacher_forcing_scheduler=TeacherForcingScheduler(),
    tokenizer=tokenizer_20,
    device=device,
    pad_idx=tokenizer_20.token_to_id("[PAD]"),
    clip_norm=2.0,
    num_epochs=10,
    max_length_generate=40,
    initial_teacher_forcing_ratio=0.9,
    filename='baseline_results'
)