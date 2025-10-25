#! python3
"""
Inference script for Transformer model.
Evaluates BLEU score on held-out test set and generates sample translations.
"""

# %% Imports
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
import yaml
import sentencepiece as spm
import evaluate

from scripts.dataset import TranslationDataset
from scripts.transformer_model import TransformerTranslation
from scripts.utils import token_to_sentence


# %% Setup paths and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

root_path = Path(__file__).resolve().parents[0]
data_path = root_path / "data"
model_path = root_path / "saved_models"
checkpoint_path = model_path / "transformer_checkpoint.tar"

# Tokenizer paths
source_tokenizer_path = data_path / "tokenizer_en.model"
target_tokenizer_path = data_path / "tokenizer_fr.model"

# Test data path
test_path = data_path / "test" / "translations.csv"


# %% Load config
with open('config/config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

print(f"Loaded config: {config}")


# %% Load tokenizers
def get_tokenizer(tokenizer_path):
    token = spm.SentencePieceProcessor()
    token.Load(str(tokenizer_path))
    return token

source_tokenizer = get_tokenizer(source_tokenizer_path)
target_tokenizer = get_tokenizer(target_tokenizer_path)

EOS_token = target_tokenizer.piece_to_id('</s>')  # Should be 2


# %% Load test dataset
print(f"\nLoading test set from: {test_path}")
test_df = pd.read_csv(test_path)
test_dataset = TranslationDataset(test_df, source_tokenizer, target_tokenizer)
print(f"Test set size: {len(test_dataset)} examples")

# Create dataloader
collate_fn = lambda x: (pad_sequence(i, batch_first=True) for i in x)
test_loader = DataLoader(
    test_dataset,
    batch_size=config['batch_size'],
    shuffle=False,
    collate_fn=collate_fn
)


# %% Load model and checkpoint
print(f"\nLoading checkpoint from: {checkpoint_path}")

# Initialize Transformer model
model = TransformerTranslation(
    src_vocab_size=config['vocab_source'],
    tgt_vocab_size=config['vocab_target'],
    d_model=config.get('d_model', 512),
    num_heads=config.get('num_heads', 8),
    num_layers=config['num_layers'],
    d_ff=config.get('d_ff', 2048),
    dropout=0.0,  # No dropout during inference
    max_len=512
)

# Load checkpoint if exists
if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["nn_state"], strict=False)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Best validation loss: {checkpoint['loss']:.4f}")
else:
    print(f"⚠️  WARNING: No checkpoint found at {checkpoint_path}")
    print("Using randomly initialized model")

model.to(device)
model.eval()


# %% Evaluate - BLEU score on test set
print("\n" + "="*60)
print("EVALUATING ON TEST SET")
print("="*60)

bleu = evaluate.load("bleu")
preds_list = []
actuals_list = []

print("Generating translations...")
with torch.no_grad():
    for x_s, x_t, _ in test_loader:
        x_s, x_t = x_s.to(device), x_t.to(device)

        # Generate translations
        outs, _ = model.evaluate(x_s)

        # Convert token IDs to text
        preds = token_to_sentence(outs, target_tokenizer)
        actuals = token_to_sentence(x_t, target_tokenizer)

        preds_list.extend(preds)
        actuals_list.extend(actuals)

# Compute BLEU score
references = [[actual] for actual in actuals_list]
score = bleu.compute(predictions=preds_list, references=references)

print(f"\n✓ Test set BLEU score: {100*score['bleu']:.4f}")
print(f"  (Evaluated on {len(preds_list)} held-out test examples)")


# %% Sample translations on custom sentences
print("\n" + "="*60)
print("SAMPLE TRANSLATIONS")
print("="*60)

source_sentences = [
    "Can you walk?",
    "What is love?",
    "I am very happy",
    "Can you give me a cheese omelette?",
    "I dream in french.",
    "I saw you cooking.",
    "He says he met my mother.",
    "Would you lend me some money?",
    "I'm looking forward to seeing you next Sunday.",
    "The war between the two countries ended with a big loss for both sides.",
    "The only thing that really matters is whether or not you did your best.",
    "The police compared the fingerprints on the gun with those on the door.",
    "The English language is undoubtedly the easiest and at the same time the most efficient means of international communication.",
    "I only eat the vegetables that I grow myself.",
]

# Prepare input
x_test = test_dataset.from_sentence_list('source', source_sentences)

# Generate translations
print("\nGenerating translations for sample sentences...\n")
with torch.no_grad():
    outs, weights = model.evaluate(x_test)
    preds = token_to_sentence(outs, target_tokenizer)

# Display results
for i, (src, pred) in enumerate(zip(source_sentences, preds), 1):
    print(f"{i:2d}. EN: {src}")
    print(f"    FR: {pred}")
    print()