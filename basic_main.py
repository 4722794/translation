#! python3
# Basic training script - minimal output
import torch
import torch.nn as nn
from pathlib import Path
from setup import get_tokenizer, get_dataset, get_dataloader, get_model, get_optimizer, get_scheduler, get_bleu, init_checkpoint
from scripts.utils import train_loop, valid_loop, save_checkpoint
import yaml
from dataclasses import make_dataclass

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_path = Path(__file__).resolve().parents[0]
data_path = root_path / "data"
model_path = root_path / "saved_models"
model_path.mkdir(exist_ok=True)

checkpoint_path = model_path / "checkpoint.tar"

train_path = data_path / "train/translations.csv"
val_path = data_path / "val/translations.csv"
test_path = data_path / "test/translations.csv"
source_tokenizer_path = data_path / "tokenizer_en.model"
target_tokenizer_path = data_path / "tokenizer_fr.model"

# Load config
with open('config/config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

fields = [(k, type(v)) for k, v in config.items()]
DotDict = make_dataclass('DotDict', fields)
conf = DotDict(**config)

# Initialize tokenizers
source_tokenizer = get_tokenizer(source_tokenizer_path)
target_tokenizer = get_tokenizer(target_tokenizer_path)

# Initialize or load checkpoint
if not checkpoint_path.exists():
    checkpoint = init_checkpoint(conf, checkpoint_path, device)
else:
    checkpoint = torch.load(checkpoint_path, map_location=device)

# Main training
def main(config, checkpoint):
    # Load data
    train_set = get_dataset(train_path, source_tokenizer, target_tokenizer)
    val_set = get_dataset(val_path, source_tokenizer, target_tokenizer)
    test_set = get_dataset(test_path, source_tokenizer, target_tokenizer)

    train_loader = get_dataloader(train_set, config['batch_size'])
    val_loader = get_dataloader(val_set, config['batch_size'])
    test_loader = get_dataloader(test_set, config['batch_size'])

    # Initialize model
    model = get_model(
        config['vocab_source'], config['vocab_target'],
        config['embedding_size'], config['hidden_size'],
        config['dropout'], config['dropout'],
        config['num_layers'], config['dot_product']
    )
    model.to(device)

    optim = get_optimizer(model, config['optimizer'], config['learning_rate'])
    scheduler = get_scheduler(optim, config['scheduler'])
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    # Training loop
    for epoch in range(config['num_epochs']):
        train_loss = train_loop(model, train_loader, loss_fn, optim, scheduler, epoch, device)
        val_loss = valid_loop(model, val_loader, loss_fn, device)

        # Save if improved
        if val_loss < checkpoint["loss"]:
            checkpoint = save_checkpoint(checkpoint_path, model, epoch, val_loss, optim, scheduler)

        # Stats per epoch
        bleu_val = get_bleu(model, val_loader, device)
        bleu_test = get_bleu(model, test_loader, device)

        print(f"Epoch {epoch+1}/{config['num_epochs']} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"BLEU Val: {bleu_val:.4f} Test: {bleu_test:.4f}")

if __name__ == "__main__":
    main(config, checkpoint)
