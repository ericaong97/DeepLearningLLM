import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from rouge_score import rouge_scorer
import numpy as np
import json
import traceback
import os
import matplotlib.pyplot as plt

# 0. Masking Function
def create_mask(tgt, pad_idx, num_heads=1):
    """
    Creates the target mask (padding and look-ahead).
    Args:
        tgt: The target input sequence (tensor).
        pad_idx: The padding token ID.
        num_heads: The number of attention heads in the Transformer.

    Returns:
        tgt_key_padding_mask: Mask to ignore padding tokens in the target (batch_size, tgt_len).
        tgt_mask: Look-ahead mask to prevent attending to future tokens (batch_size * num_heads, tgt_len, tgt_len).
    """
    tgt_len = tgt.size(1)
    batch_size = tgt.size(0)

    # Padding mask for target
    tgt_key_padding_mask = (tgt == pad_idx).to(tgt.device)

    # Look-ahead mask
    look_ahead_mask = torch.triu(
        torch.ones((tgt_len, tgt_len), device=tgt.device, dtype=torch.bool),
        diagonal=1
    )
    look_ahead_mask = look_ahead_mask.unsqueeze(0).expand(batch_size * num_heads, -1, -1)

    return tgt_key_padding_mask, look_ahead_mask


# 1. Training Function
# def train_transformer_teacher_forcing(model, train_loader, val_loader, 
#                                     optimizer, criterion, warmup_scheduler, plateau_scheduler,
#                                     teacher_forcing_scheduler, tokenizer, device, 
#                                     pad_idx, clip_norm=2.0, num_epochs=10, max_length_generate=40,
#                                     initial_teacher_forcing_ratio=0.9,filename='result'):
#     history = {
#         'train_loss': [],
#         'val_loss': [],
#         'rouge1': [],
#         'rouge2': [],
#         'rougeL': [],
#         'learning_rate': [],
#         'teacher_forcing_ratio': []
#     }

#     warmup_epochs = 2
#     current_teacher_forcing_ratio = initial_teacher_forcing_ratio
    
#     # Initialize schedulers
#     optimizer.step()       
#     optimizer.zero_grad()
#     if warmup_scheduler is not None:
#         warmup_scheduler.step()

#     for epoch in range(num_epochs):
#         # --- Training Phase with Teacher Forcing ---
#         model.train()
#         train_loss = 0
        
#         # Initialize tqdm progress bar
#         train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', leave=True)
        
#         for batch in train_pbar:
#             input_ids = batch['input_ids'].to(device)
#             target_ids = batch['labels'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
            
#             # Create masks
#             src_key_padding_mask = (attention_mask == 0).to(device)
#             tgt_key_padding_mask, tgt_mask = create_mask(target_ids, pad_idx, num_heads=8)

#             # Forward and backward pass
#             optimizer.zero_grad()
            
#             # Forward pass with teacher forcing
#             output = model(
#                 input_ids, 
#                 target_ids,
#                 src_key_padding_mask=src_key_padding_mask,
#                 tgt_mask=tgt_mask,
#                 tgt_key_padding_mask=tgt_key_padding_mask,
#                 memory_key_padding_mask=src_key_padding_mask,
#                 teacher_forcing_ratio=current_teacher_forcing_ratio
#             )
            
#             # Loss calculation
#             loss = criterion(
#                 output.reshape(-1, len(tokenizer.get_vocab())), 
#                 target_ids[:, 1:].reshape(-1)
#             )
            
#             # Backward pass
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
#             optimizer.step()
            
#             # Update metrics and progress bar
#             train_loss += loss.item()
#             train_pbar.set_postfix({
#                 'loss': f'{loss.item():.4f}',
#                 'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
#                 'tf_ratio': f'{current_teacher_forcing_ratio:.2f}'
#             })

#         # --- Scheduler Updates ---
#         # Warmup scheduler (per batch during warmup)
#         if epoch < warmup_epochs and warmup_scheduler is not None:
#             warmup_scheduler.step()
            
#         # Teacher forcing ratio update
#         current_teacher_forcing_ratio = teacher_forcing_scheduler.step(
#             current_teacher_forcing_ratio
#         )

#         # --- Validation Phase (Deterministic) ---
#         avg_val_loss, avg_rouge = validate_transformer(
#             model, val_loader, criterion, tokenizer, device, pad_idx, max_length_generate
#         )
        
#         # Plateau scheduler (after warmup)
#         if epoch >= warmup_epochs and plateau_scheduler is not None:
#             plateau_scheduler.step(avg_val_loss)

#         # --- Metrics Tracking ---
#         history['train_loss'].append(train_loss / len(train_loader))
#         history['val_loss'].append(avg_val_loss)
#         history['learning_rate'].append(optimizer.param_groups[0]['lr'])
#         history['teacher_forcing_ratio'].append(current_teacher_forcing_ratio)
#         history.update({k: [v] for k, v in avg_rouge.items()})

#         # --- Epoch Summary ---
#         print(f'\nEpoch {epoch+1}/{num_epochs} Summary:')
#         print(f'Train Loss: {history["train_loss"][-1]:.4f} | Val Loss: {history["val_loss"][-1]:.4f}')
#         print(f'ROUGE-1: {avg_rouge["rouge1"]:.4f} | ROUGE-2: {avg_rouge["rouge2"]:.4f} | ROUGE-L: {avg_rouge["rougeL"]:.4f}')
#         print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.2e}')
#         print(f'Teacher Forcing Ratio: {current_teacher_forcing_ratio:.2f}')
#         print('-'*50)

#     # --- After the training loop finishes ---
#     # Save the training history to a JSON file
#     output_file = f"{filename}.txt"
#     with open(output_file, 'w') as f:
#         json.dump(history, f, indent=4)  # Use indent for pretty formatting

#     print(f"Training history saved to {output_file}")

#     return history

# 1. Training Function
def train_transformer_teacher_forcing(model, train_loader, val_loader, 
                                    optimizer, criterion, plateau_scheduler,
                                    teacher_forcing_scheduler, tokenizer, device, 
                                    pad_idx, clip_norm=2.0, num_epochs=10, max_length_generate=40,
                                    initial_teacher_forcing_ratio=0.9, filename='result',
                                    checkpoint_interval=2):
    # Initialize history tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'rouge1': [],
        'rouge2': [],
        'rougeL': [],
        'learning_rate': [],
        'teacher_forcing_ratio': []
    }

    current_teacher_forcing_ratio = initial_teacher_forcing_ratio
    checkpoint_file = f"{filename}_checkpoint.pt"
    start_epoch = 0

    # Load checkpoint if exists
    if os.path.exists(checkpoint_file):
        print(f"Loading checkpoint from {checkpoint_file}...")
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        history = checkpoint['history']
        start_epoch = checkpoint['epoch'] + 1
        current_teacher_forcing_ratio = checkpoint['teacher_forcing_ratio']
        
        # Load scheduler states if present
        if 'plateau_scheduler_state' in checkpoint and plateau_scheduler:
            plateau_scheduler.load_state_dict(checkpoint['plateau_scheduler_state'])
        if 'teacher_forcing_scheduler_state' in checkpoint:
            teacher_forcing_scheduler.load_state_dict(checkpoint['teacher_forcing_scheduler_state'])
            
        print(f"Resuming training from epoch {start_epoch}")

    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            train_loss = 0
            
            # Training phase
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
                # Prepare inputs
                input_ids = batch['input_ids'].to(device)
                target_ids = batch['labels'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Create masks
                src_key_padding_mask = (attention_mask == 0).to(torch.bool).to(device)
                tgt_key_padding_mask, tgt_mask = create_mask(target_ids, pad_idx, num_heads=8)
                tgt_mask = tgt_mask.to(torch.bool).to(device)
                tgt_key_padding_mask = tgt_key_padding_mask.to(torch.bool).to(device)

                # Forward pass
                optimizer.zero_grad()
                output = model(
                    input_ids, 
                    target_ids,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=src_key_padding_mask,
                    teacher_forcing_ratio=current_teacher_forcing_ratio
                )
                
                # Backward pass
                loss = criterion(
                    output.reshape(-1, len(tokenizer.get_vocab())), 
                    target_ids[:, 1:].reshape(-1)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
                
                # Update teacher forcing scheduler
                current_teacher_forcing_ratio = teacher_forcing_scheduler.step()
                train_loss += loss.item()

            # Validation phase
            avg_val_loss, avg_rouge = validate_transformer(
                model, val_loader, criterion, tokenizer, device, pad_idx, max_length_generate
            )
            
            # Update plateau scheduler (using validation loss)
            plateau_scheduler.step(avg_val_loss)

            # Update history
            history['train_loss'].append(train_loss / len(train_loader))
            history['val_loss'].append(avg_val_loss)
            history['rouge1'].append(avg_rouge['rouge1'])
            history['rouge2'].append(avg_rouge['rouge2'])
            history['rougeL'].append(avg_rouge['rougeL'])
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            history['teacher_forcing_ratio'].append(current_teacher_forcing_ratio)

            # --- Epoch Summary ---
            print(f'\nEpoch {epoch+1}/{num_epochs} Summary:')
            print(f'Train Loss: {history["train_loss"][-1]:.4f} | Val Loss: {history["val_loss"][-1]:.4f}')
            print(f'ROUGE-1: {avg_rouge["rouge1"]:.4f} | ROUGE-2: {avg_rouge["rouge2"]:.4f} | ROUGE-L: {avg_rouge["rougeL"]:.4f}')
            print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.2e}')
            print(f'Teacher Forcing Ratio: {current_teacher_forcing_ratio:.2f}')
            print('-'*50)

            # Save checkpoint
            if (epoch + 1) % checkpoint_interval == 0 or epoch == num_epochs - 1:
                torch.save({
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'plateau_scheduler_state': plateau_scheduler.state_dict(),
                    'teacher_forcing_scheduler_state': teacher_forcing_scheduler.state_dict(),
                    'history': history,
                    'teacher_forcing_ratio': current_teacher_forcing_ratio
                }, checkpoint_file)
                print(f"Checkpoint saved after epoch {epoch+1}")

    except (Exception, KeyboardInterrupt) as e:
        print(f"\nTraining interrupted: {str(e)}")
        print("Saving emergency checkpoint...")
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'history': history,
            'teacher_forcing_ratio': current_teacher_forcing_ratio
        }, f"{filename}_emergency.pt")
        raise

    # Final save
    torch.save(model.state_dict(), f"{filename}_final_model.pt")
    with open(f"{filename}_history.json", 'w') as f:
        json.dump(history, f, indent=4)

    return history

# 2. Validation Function
def validate_transformer(model, val_loader, criterion, tokenizer, device, pad_idx,
                        max_length_generate=40):
    """Fixed validation function with proper mask handling and error prevention"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    model.eval()
    
    # Disable nested tensors to prevent warnings
    model.transformer.use_nested_tensor = False
    
    val_loss = 0
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    with torch.no_grad():
        for batch in val_loader:
            # Convert all tensors to device first
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Create proper boolean masks (CRITICAL FIX)
            src_key_padding_mask = (attention_mask == 0).to(torch.bool)
            
            # Verify mask shapes
            if src_key_padding_mask.shape != input_ids.shape:
                raise ValueError(f"Mask shape mismatch: {src_key_padding_mask.shape} vs {input_ids.shape}")
            
            # Create target masks
            tgt_key_padding_mask, look_ahead_mask = create_mask(
                target_ids, 
                pad_idx, 
                num_heads=8
            )
            
            # Forward pass with explicit safety checks
            try:
                output = model(
                    input_ids, 
                    target_ids,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_mask=look_ahead_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=src_key_padding_mask
                )
                
                loss = criterion(
                    output[:, :-1].reshape(-1, len(tokenizer.get_vocab())),
                    target_ids[:, 1:].reshape(-1)
                )
                val_loss += loss.item()
                
            except RuntimeError as e:
                print(f"Forward pass failed: {str(e)}")
                continue

            # Generation with proper device handling
            try:
                # Get single example for generation
                single_input = input_ids[0:1]
                single_mask = src_key_padding_mask[0:1]
                
                # Encode source
                memory = model.transformer.encoder(
                    model.pos_encoder_src(model.embedding_src(single_input)),
                    src_key_padding_mask=single_mask
                )
                
                # Generate summary
                ys = torch.ones(1, 1, device=device).fill_(tokenizer.token_to_id("[SOS]")).long()
                
                for i in range(1, max_length_generate):
                    output = model.transformer.decoder(
                        model.pos_encoder_tgt(model.embedding_tgt(ys)),
                        memory,
                        tgt_mask=torch.triu(torch.ones(i, i, device=device), diagonal=1).bool(),
                        tgt_key_padding_mask=None,
                        memory_key_padding_mask=single_mask
                    )
                    logits = model.projection(output[:, -1:])
                    next_token = logits.argmax(-1)
                    ys = torch.cat([ys, next_token], dim=1)
                    
                    if next_token.item() == tokenizer.token_to_id("[EOS]"):
                        break
                
                pred_summary = tokenizer.decode(ys[0].tolist(), skip_special_tokens=True)
                ref_summary = tokenizer.decode(target_ids[0].tolist(), skip_special_tokens=True)
                
                scores = scorer.score(ref_summary, pred_summary)
                for key in rouge_scores:
                    rouge_scores[key].append(scores[key].fmeasure)
                    
            except Exception as e:
                print(f"Generation failed: {str(e)}")
                continue

    avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
    avg_rouge = {k: np.mean(v) if v else 0 for k, v in rouge_scores.items()}
    
    return avg_val_loss, avg_rouge


# 3. Teacher Forcing Ratio Function
class TeacherForcingScheduler:
    def __init__(self, decay_rate=0.85, min_ratio=0.3, initial_ratio=0.9):
        self.decay_rate = decay_rate
        self.min_ratio = min_ratio
        self.current_ratio = initial_ratio  # Track current state
        self.initial_ratio = initial_ratio  # For resetting
    
    def step(self):
        """Returns updated ratio and tracks it internally"""
        self.current_ratio = max(self.current_ratio * self.decay_rate, self.min_ratio)
        return self.current_ratio
    
    def state_dict(self):
        return {
            'decay_rate': self.decay_rate,
            'min_ratio': self.min_ratio,
            'current_ratio': self.current_ratio,
            'initial_ratio': self.initial_ratio
        }
    
    def load_state_dict(self, state_dict):
        self.decay_rate = state_dict['decay_rate']
        self.min_ratio = state_dict['min_ratio']
        self.current_ratio = state_dict['current_ratio']
        self.initial_ratio = state_dict.get('initial_ratio', 0.9)  # Backward compatible


# 4. Greedy decoding strategy
def greedy_decode(model, input_tensor, tokenizer, device, max_length=40):
    """Simplified generation without beam search for debugging"""
    model.eval()
    with torch.no_grad():
        # 1. Encode input
        src = input_tensor.unsqueeze(0) if input_tensor.dim() == 1 else input_tensor
        src_emb = model.pos_encoder_src(model.embedding_src(src))
        memory = model.transformer.encoder(src_emb)
        
        # 2. Initialize with SOS token
        output_ids = [tokenizer.token_to_id('[SOS]')]
        
        for _ in range(max_length):  # Now using integer max_length
            # 3. Decode step-by-step
            tgt = torch.tensor([output_ids], device=device)
            tgt_emb = model.pos_encoder_tgt(model.embedding_tgt(tgt))
            
            output = model.transformer.decoder(
                tgt_emb,
                memory,
                tgt_mask=torch.triu(
                    torch.ones(len(output_ids), len(output_ids), device=device),
                    diagonal=1
                ).bool()
            )
            
            # 4. Get next token
            next_token = model.projection(output[:, -1]).argmax(-1).item()
            output_ids.append(next_token)
            
            # 5. Stop if EOS is generated
            if next_token == tokenizer.token_to_id('[EOS]'):
                break
                
        # 6. Return decoded text (excluding SOS token)
        return tokenizer.decode(output_ids[1:], skip_special_tokens=True)
    

# 5. Beam search decoding strategy
def generate_with_beam_search(model, input_ids, tokenizer, device, 
                            beam_width=5, max_length=40,
                            pad_token_id=1, eos_token_id=3):
    """Beam search implementation with proper tensor handling"""
    model.eval()
    
    # 1. Encode input
    src = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
    src_mask = (src == pad_token_id).to(device)
    
    with torch.no_grad():
        # 2. Encoder forward pass
        src_emb = model.pos_encoder_src(model.embedding_src(src))
        memory = model.transformer.encoder(src_emb, src_key_padding_mask=src_mask)
        
        # 3. Initialize beams (sequence, score, completed)
        beams = [{
            'tokens': [tokenizer.token_to_id('[SOS]')],
            'score': 0.0,
            'completed': False
        }]

        # 4. Beam search loop
        for _ in range(max_length):
            candidates = []
            
            for beam in beams:
                if beam['completed']:
                    candidates.append(beam)
                    continue
                
                # Prepare decoder input
                tgt = torch.tensor([beam['tokens']], device=device)
                tgt_emb = model.pos_encoder_tgt(model.embedding_tgt(tgt))
                
                # Create attention mask
                tgt_mask = torch.triu(
                    torch.ones(len(beam['tokens']), len(beam['tokens']), device=device),
                    diagonal=1
                ).bool()
                
                # Decoder forward
                output = model.transformer.decoder(
                    tgt_emb,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=src_mask
                )
                
                # Get next token probabilities
                logits = model.projection(output[:, -1])
                log_probs = F.log_softmax(logits, dim=-1)
                topk_scores, topk_ids = log_probs.topk(beam_width, dim=-1)
                
                # Expand beams
                for i in range(beam_width):
                    new_tokens = beam['tokens'] + [topk_ids[0, i].item()]
                    new_score = beam['score'] + topk_scores[0, i].item()
                    completed = (topk_ids[0, i] == eos_token_id).item()
                    
                    candidates.append({
                        'tokens': new_tokens,
                        'score': new_score,
                        'completed': completed
                    })
            
            # Select top beams
            beams = sorted(candidates, key=lambda x: x['score'], reverse=True)[:beam_width]
            
            # Debug print (remove after verification)
            # print(f"Step {_+1} top beam:")
            # print(f"Tokens: {beams[0]['tokens']}")
            # print(f"Text: {tokenizer.decode(beams[0]['tokens'][1:])}")
            # print(f"Score: {beams[0]['score']:.2f}")
            # print("-" * 50)
            
            # Early stopping if all beams completed
            if all(beam['completed'] for beam in beams):
                break
        
        # 5. Return best sequence (excluding SOS)
        best_beam = beams[0]
        return tokenizer.decode(best_beam['tokens'][1:], skip_special_tokens=True)


########
## Model Result Visualization
########

def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    
    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot learning rate and teacher forcing ratio
    plt.subplot(1, 2, 2)
    plt.plot(history['learning_rate'], label='Learning Rate', color='green')
    plt.plot(history['teacher_forcing_ratio'], label='Teacher Forcing', color='purple')
    plt.title('Training Parameters')
    plt.xlabel('Epochs')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
    plt.show()
    
def plot_rouge_scores(history):
    plt.figure(figsize=(10, 5))
    
    metrics = ['rouge1', 'rouge2', 'rougeL']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for metric, color in zip(metrics, colors):
        plt.plot(history[metric], label=metric.upper(), color=color, linewidth=2)
    
    plt.title('ROUGE Scores During Training', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.savefig('rouge_scores.png', bbox_inches='tight', dpi=300)