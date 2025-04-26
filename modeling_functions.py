import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from rouge_score import rouge_scorer
import numpy as np
import json
from pathlib import Path
import os
from torch.optim.lr_scheduler import LambdaLR

################################
# Training Section
################################

# 1. Training Function
def train_transformer_teacher_forcing(model, train_loader, val_loader, 
                                    optimizer, criterion, plateau_scheduler,
                                    teacher_forcing_scheduler, tokenizer, device, 
                                    pad_idx, clip_norm=2.0, num_epochs=10, max_length_generate=40,
                                    filename='result', checkpoint_interval=2, 
                                    use_early_stopping=False, patience=3, min_delta=0.0001):
    # Initialize history tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'rouge1': [],
        'rouge2': [],
        'rougeL': [],
        'learning_rate': [],
        'teacher_forcing_ratio': [],
        'global_step': 0
    }

    # Early stopping variables
    best_val_loss = float('inf')
    early_stopping_counter = 0
    best_model_path = f"{filename}_best_model.pt"

    checkpoint_file = f"{filename}_checkpoint.pt"
    start_epoch = 0

    # Warmup scheduler configuration
    warmup_steps = 1 * len(train_loader)  # 1 epochs of batches
    initial_lr = 5e-5
    target_lr = 2e-4
    # Scaling from 5e-5 to 2e-4
    warmup_scheduler = LambdaLR(
    optimizer,
    lr_lambda=lambda step: (initial_lr + (target_lr - initial_lr) * min(1.0, step / warmup_steps))/ target_lr
    )
    
    # Load checkpoint if exists
    if os.path.exists(checkpoint_file):
        print(f"Loading checkpoint from {checkpoint_file}...")
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        history = checkpoint['history']
        start_epoch = checkpoint['epoch'] + 1
        if use_early_stopping and 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']
            early_stopping_counter = checkpoint['early_stopping_counter']
        print(f"Resuming training from epoch {start_epoch}")

    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            epoch_train_loss = 0
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
                # Get current teacher forcing ratio
                if epoch < 2:
                    tf_ratio = 0.9  # Warmup phase, keep the same
                else:
                    tf_ratio = teacher_forcing_scheduler.step()
                
                # Forward pass (masks handled internally by model)
                optimizer.zero_grad()
                output = model(
                    src=batch['input_ids'].to(device),
                    tgt=batch['labels'].to(device),
                    src_key_padding_mask=(batch['attention_mask'] == 0).to(device),
                    teacher_forcing_ratio=tf_ratio
                )
                
                # Loss calculation
                logits = output.view(-1, output.size(-1))
                targets = batch['labels'][:, 1:].contiguous().view(-1).to(device)
                loss = criterion(logits, targets)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
                
                # Update tracking
                epoch_train_loss += loss.item()
                history['global_step'] += 1
                
                # Learning rate warmup
                if history['global_step'] < warmup_steps:
                    warmup_scheduler.step()

            # Validation phase
            avg_val_loss, rouge_stats = validate_transformer(
                model, val_loader, criterion, tokenizer, device, pad_idx, max_length_generate
            )
            
            # Update history and schedulers
            if epoch >= 2:
                plateau_scheduler.step(avg_val_loss)

            history['train_loss'].append(epoch_train_loss / len(train_loader))
            history['val_loss'].append(avg_val_loss)
            history['rouge1'].append(rouge_stats['rouge1_mean'])
            history['rouge2'].append(rouge_stats['rouge2_mean'])
            history['rougeL'].append(rouge_stats['rougeL_mean'])
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            history['teacher_forcing_ratio'].append(tf_ratio)

            # Epoch summary
            print(f'\nEpoch {epoch+1}/{num_epochs} Summary:')
            print(f'Train Loss: {history["train_loss"][-1]:.4f} | Val Loss: {history["val_loss"][-1]:.4f}')
            print(f'ROUGE Scores: {rouge_stats["rouge1_mean"]:.4f}/{rouge_stats["rouge2_mean"]:.4f}/{rouge_stats["rougeL_mean"]:.4f}')
            print(f'Learning Rate: {history["learning_rate"][-1]:.2e} | TF Ratio: {tf_ratio:.2f}')

            # Early stopping check
            if use_early_stopping:
                if avg_val_loss < best_val_loss - min_delta:
                    best_val_loss = avg_val_loss
                    early_stopping_counter = 0
                    # Save best model
                    torch.save(model.state_dict(), best_model_path)
                    print(f"New best model saved with validation loss: {best_val_loss:.4f}")
                else:
                    early_stopping_counter += 1
                    print(f"Early stopping counter: {early_stopping_counter}/{patience}")
                    if early_stopping_counter >= patience:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                        break

            # Save checkpoint
            if (epoch + 1) % checkpoint_interval == 0:
                checkpoint_data = {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'history': history,
                }
                if use_early_stopping:
                    checkpoint_data.update({
                        'best_val_loss': best_val_loss,
                        'early_stopping_counter': early_stopping_counter
                    })
                torch.save(checkpoint_data, checkpoint_file)

    except Exception as e:
        print(f"Training interrupted: {str(e)}")
        checkpoint_data = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'history': history,
        }
        if use_early_stopping:
            checkpoint_data.update({
                'best_val_loss': best_val_loss,
                'early_stopping_counter': early_stopping_counter
            })
        torch.save(checkpoint_data, f"{filename}_emergency.pt")
        raise

    # Final save
    torch.save(model.state_dict(), f"{filename}_final_model.pt")
    
    # If using early stopping, load the best model
    if use_early_stopping and os.path.exists(best_model_path):
        print(f"Loading best model with validation loss: {best_val_loss:.4f}")
        model.load_state_dict(torch.load(best_model_path))
    
    with open(f"{filename}_history.json", 'w') as f:
        json.dump(history, f, indent=4)
    return history

##########################

# 2. Validation function
def validate_transformer(model, val_loader, criterion, tokenizer, device, pad_idx, max_length_generate=40):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    model.eval()
    val_loss = 0
    total_batches = 0
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            try:
                # Move data to device
                input_ids = batch['input_ids'].to(device)
                target_ids = batch['labels'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Create masks
                src_mask = (attention_mask == 0)
                
                # Forward pass - use full target length but let model handle slicing
                output = model(
                    input_ids,
                    target_ids,  # Full target sequence
                    src_key_padding_mask=src_mask,
                    memory_key_padding_mask=src_mask,
                    teacher_forcing_ratio=0.0  # Force generation path
                )
                
                # Calculate loss - compare output to targets shifted right by 1
                logits = output.reshape(-1, output.size(-1))
                targets = target_ids[:, 1:].reshape(-1)
                
                # Verify shapes
                assert logits.size(0) == targets.size(0), \
                    f"Shape mismatch: {logits.shape} vs {targets.shape}"
                
                # Filter padding tokens
                if pad_idx is not None:
                    non_pad = targets.ne(pad_idx)
                    logits = logits[non_pad]
                    targets = targets[non_pad]
                    if logits.numel() == 0:
                        continue
                
                loss = criterion(logits, targets)
                val_loss += loss.item()
                
                # ROUGE calculation (limited samples for efficiency)
                batch_rouge = {'rouge1': [], 'rouge2': [], 'rougeL': []}
                for i in range(min(2, input_ids.size(0))):  # Only first 2 samples
                    pred_summary = greedy_decode(
                        model,
                        input_ids[i:i+1],
                        tokenizer,
                        device,
                        max_length_generate
                    )
                    ref_summary = tokenizer.decode(
                        target_ids[i].tolist(),
                        skip_special_tokens=True
                    )
                    
                    # Compute and store ROUGE scores for this sample
                    scores = scorer.score(ref_summary, pred_summary)
                    for key in batch_rouge:
                        batch_rouge[key].append(scores[key].fmeasure)
                
                # Store batch-level ROUGE scores
                for key in rouge_scores:
                    if batch_rouge[key]:  # Only append if we have scores
                        rouge_scores[key].extend(batch_rouge[key])
                
                total_batches += 1
                
            except Exception as e:
                print(f"Validation error in batch {batch_idx}: {str(e)}")
                continue
    
    # Calculate average metrics
    avg_val_loss = val_loss / total_batches if total_batches > 0 else float('inf')
    
    # Compute ROUGE statistics
    rouge_stats = {}
    for key in rouge_scores:
        if rouge_scores[key]:  # Only compute if we have scores
            rouge_stats[f'{key}_mean'] = np.mean(rouge_scores[key])
            rouge_stats[f'{key}_std'] = np.std(rouge_scores[key])
            rouge_stats[f'{key}_min'] = np.min(rouge_scores[key])
            rouge_stats[f'{key}_max'] = np.max(rouge_scores[key])
        else:
            rouge_stats.update({f'{key}_mean': 0, f'{key}_std': 0, 
                                f'{key}_min': 0, f'{key}_max': 0})
    
    return avg_val_loss, rouge_stats


##########################

# 3. Greeding decoding for training
def greedy_decode(model, input_ids, tokenizer, device, max_length):
    model.eval()
    sos_token = tokenizer.token_to_id("[SOS]") if hasattr(tokenizer, 'token_to_id') else tokenizer.convert_tokens_to_ids("[SOS]")
    eos_token = tokenizer.token_to_id("[EOS]") if hasattr(tokenizer, 'token_to_id') else tokenizer.convert_tokens_to_ids("[EOS]")
    
    # Initialize with SOS token
    generated = torch.tensor([[sos_token]], device=device)
    
    # Encode source
    src_mask = (input_ids != 1).to(device)
    with torch.no_grad():
        memory = model.transformer.encoder(
            model.pos_encoder_src(model.embedding_src(input_ids)),
            src_key_padding_mask=~src_mask
        )
    
    # Autoregressive generation
    for _ in range(max_length - 1):
        # Create target mask
        tgt_mask = torch.triu(
            torch.ones((generated.size(1), generated.size(1)), device=device),
            diagonal=1
        ).bool()
        
        # Decode step
        with torch.no_grad():
            output = model.transformer.decoder(
                model.pos_encoder_tgt(model.embedding_tgt(generated)),
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=~src_mask
            )
            logits = model.projection(output[:, -1:])
            next_token = logits.argmax(-1)
        
        # Break if EOS generated
        if next_token.item() == eos_token:
            break
            
        generated = torch.cat([generated, next_token], dim=1)
    
    return tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)    
    
################################    
# Inference Section    
################################
    
# 1. Beam search decoding strategy
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
            
            # Early stopping if all beams completed
            if all(beam['completed'] for beam in beams):
                break
        
        # 5. Return best sequence (excluding SOS)
        best_beam = beams[0]
        return tokenizer.decode(best_beam['tokens'][1:], skip_special_tokens=True)


# 2. Rouge-score calculator
def calculate_and_save_rouge(generated_summaries, reference_summaries, output_path="rouge_scores.json"):
    """
    Calculate and save only average ROUGE scores:
    {
        "rouge1_avg": 0.43,
        "rouge2_avg": 0.28,
        "rougeL_avg": 0.38
    }
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Initialize accumulators
    rouge1_sum = rouge2_sum = rougeL_sum = 0.0
    count = 0
    
    # Calculate running totals
    for gen, ref in zip(generated_summaries, reference_summaries):
        scores = scorer.score(ref, gen)
        rouge1_sum += scores['rouge1'].fmeasure
        rouge2_sum += scores['rouge2'].fmeasure
        rougeL_sum += scores['rougeL'].fmeasure
        count += 1
    
    # Compute averages
    results = {
        "rouge1_avg": rouge1_sum / count if count > 0 else 0,
        "rouge2_avg": rouge2_sum / count if count > 0 else 0,
        "rougeL_avg": rougeL_sum / count if count > 0 else 0
    }
    
    # Save compact JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)  # Smaller indent for more compact file
    
    return results