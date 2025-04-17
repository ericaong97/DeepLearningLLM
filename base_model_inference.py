# libraries for inference
import torch
from baseline_transformer_architecture import create_small_transformer
from modeling_functions import generate_with_beam_search,calculate_and_save_rouge
from tokenizers import Tokenizer
from data_utils import get_test_loader,set_seed

# 1. Configuration setup
set_seed(42)
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

# 2. Load the tokenizer
tokenizer_20 = Tokenizer.from_file("cnn_bpe_tokenizer_20k.json")  # Make sure this is the correct path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pad_idx = tokenizer_20.token_to_id("[PAD]")

# 3. Recreate the model architecture
loaded_model = create_small_transformer(d_model=config['d_model'],
                                nhead=config['nhead'],
                                num_decoder_layers=config['num_decoder_layers'],
                                num_encoder_layers=config['num_encoder_layers'],
                                dim_feedforward=config['dim_feedforward'],
                                dropout=config['dropout'],
                                vocab_size=config['vocab_size']).to(device)

# 4. Load the saved state dictionary
# change the file based on your model name
loaded_model.load_state_dict(torch.load('updated_baseline_final_model.pt'))
loaded_model = loaded_model.to(device)

# 5. Set the model to evaluation mode
loaded_model.eval()

# 6. Load dataset
test_loader = get_test_loader(tokenizer_20)  #  Use the tokenizer_20 instance

# 7. Get a batch of data based on number of samples selected
num_examples_to_show = 5
generated_summaries = []
reference_summaries = []
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        # 1. Move data to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 2. Verify special tokens (do this once before the loop)
        if i == 0:  # Only print once
            print("[DEBUG] Special tokens:")
            print(f"[PAD]: {tokenizer_20.token_to_id('[PAD]')} -> '{tokenizer_20.decode([tokenizer_20.token_to_id('[PAD]')])}'")
            print(f"[SOS]: {tokenizer_20.token_to_id('[SOS]')}")
            print(f"[EOS]: {tokenizer_20.token_to_id('[EOS]')}")
            print(f"Dot token: {tokenizer_20.token_to_id('.')} -> '{tokenizer_20.decode([tokenizer_20.token_to_id('.')])}'")
        
        generated_summary = generate_with_beam_search(
                            model=loaded_model,
                            input_ids=input_ids[0],  # Single sequence
                            tokenizer=tokenizer_20,
                            device=device,
                            beam_width=3,
                            max_length=40
                )
        
        # 4. Decode references
        actual_summary = tokenizer_20.decode(
            labels[0].tolist(), 
            skip_special_tokens=True
        )
        input_article = tokenizer_20.decode(
            input_ids[0].tolist(),
            skip_special_tokens=True
        )
    
        # Store for ROUGE calculation
        generated_summaries.append(generated_summary)
        reference_summaries.append(actual_summary)
        
        # 5. Print results
        # print(f"\n--- Example {i+1} ---")
        # print(f"Input: {input_article[:200]}...")  # Truncate long inputs
        # print(f"Generated Summary: {generated_summary}")
        # print(f"Actual Summary: {actual_summary}")
        # print("-" * 50)
        
        if i + 1 >= num_examples_to_show:
            break

# 8. Generating final rouge scores
# change the output_path for your own experiments
rouge_results = calculate_and_save_rouge(
    generated_summaries=generated_summaries,
    reference_summaries=reference_summaries,
    output_path="inf_rouge_scores.json"
)