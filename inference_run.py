# libraries for inference
import torch
import json
from pathlib import Path
from tqdm import tqdm
from baseline_transformer_architecture import create_small_transformer
from modeling_functions import generate_with_beam_search, calculate_and_save_rouge
from tokenizers import Tokenizer
from data_utils import get_test_loader, set_seed
import torch.nn.functional as F

def main():
    # 1. Configuration setup
    set_seed(42)
    config = {
        "vocab_size": 20000,
        "dropout": 0.1,
        "d_model": 512,
        "nhead": 8,
        "num_encoder_layers": 4,
        "num_decoder_layers": 4,
        "dim_feedforward": 2048
    }

    # 2. Initialize with automatic GPU selection
    tokenizer = Tokenizer.from_file("cnn_bpe_tokenizer_20k.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Auto-select GPU
    output_dir = Path("baseline_inf_results")
    output_dir.mkdir(exist_ok=True)

    # 3. Model setup with MAXIMUM optimization
    model = create_small_transformer(**config).to(device)
    model.load_state_dict(torch.load('final_baseline_final_model.pt'))
    model.eval()
    
    # 4. Ultimate performance optimizations
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.enable_math_sdp(True)  # Fallback option
    
    # 5. Data loader with optimized settings
    test_loader = get_test_loader(
        tokenizer,
        batch_size=64,  # Adjusted for your GPU
        num_workers=2   
    )

    # 6. Inference pipeline
    generated_summaries = []
    reference_summaries = []
    demo_examples = []
    demo_count = 5
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Processing batches")):
            # Memory management
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
                
            # Get batch data
            input_ids = batch['input_ids'].to(device, non_blocking=True)  # shape: [64, seq_len]
            labels = batch['labels'].to(device, non_blocking=True)
            
            # Process EACH sequence in the batch
            batch_summaries = []
            for seq_idx in range(input_ids.size(0)):  # Loop through batch dimension
                summary = generate_with_beam_search(
                    model=model,
                    input_ids=input_ids[seq_idx],  # Pass individual sequence
                    tokenizer=tokenizer,
                    device=device,
                    beam_width=3,
                    max_length=40
                )
                batch_summaries.append(summary)
            
            # Store results
            generated_summaries.extend(batch_summaries)
            batch_references = [tokenizer.decode(label.tolist(), skip_special_tokens=True) 
                            for label in labels]
            reference_summaries.extend(batch_references)
            
            # Save examples
            if len(demo_examples) < demo_count:
                for i in range(min(len(batch_summaries), demo_count - len(demo_examples))):
                    demo_examples.append({
                        "input": tokenizer.decode(input_ids[i].tolist(), skip_special_tokens=True)[:200] + "...",
                        "generated": batch_summaries[i],
                        "reference": batch_references[i]
                    })
                    
    # 7. Save outputs
    rouge_results = calculate_and_save_rouge(
        generated_summaries=generated_summaries,
        reference_summaries=reference_summaries,
        output_path=str(output_dir/"full_rouge_scores.json")
    )
    
    with open(output_dir/"baseline_demo_examples.json", 'w') as f:
        json.dump(demo_examples, f, indent=2)
    
    print("\nResults saved to:")
    print(f"- Full ROUGE scores: {output_dir/'full_rouge_scores.json'}")
    print(f"- Example summaries: {output_dir/'baseline_demo_examples.json'}")
    print(f"\nProcessed {len(generated_summaries)} samples (Batch size=64, Workers=2)")

if __name__ == "__main__":
    main()