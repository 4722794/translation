#! python3
# Training script for Transformer model - minimal output
import torch
import torch.nn as nn
from pathlib import Path
from setup import get_tokenizer, get_dataset, get_dataloader
from scripts.transformer_model import TransformerTranslation
from scripts.utils import save_checkpoint, token_to_sentence
import yaml
from dataclasses import make_dataclass
import evaluate

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_path = Path(__file__).resolve().parents[0]
data_path = root_path / "data"
model_path = root_path / "saved_models"
model_path.mkdir(exist_ok=True)

checkpoint_path = model_path / "transformer_checkpoint.tar"

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


def get_bleu_transformer(model, test_loader, device):
    """Compute BLEU score for transformer model."""
    preds_list, actuals_list = list(), list()
    token_t = test_loader.dataset.sp_t

    for x_s, x_t, _ in test_loader:
        with torch.no_grad():
            model.to(device)
            x_s, x_t = x_s.to(device), x_t.to(device)
            outs, _ = model.evaluate(x_s)
        preds = token_to_sentence(outs, token_t)
        actuals = token_to_sentence(x_t, token_t)
        preds_list.extend(preds)
        actuals_list.extend(actuals)

    predictions = preds_list
    references = [[i] for i in actuals_list]

    try:
        bleu = evaluate.load('bleu')
        score = bleu.compute(predictions=predictions, references=references)['bleu']
    except (FileNotFoundError, Exception):
        try:
            import sacrebleu
            score = sacrebleu.corpus_bleu(predictions, [actuals_list]).score / 100.0
        except:
            score = 0.0

    return score


def train_epoch(model, loader, loss_fn, optim, scheduler, epoch, device):
    """Train for one epoch."""
    model.to(device)
    model.train()
    total_loss = 0
    num_batches = len(loader)

    for batch_idx, (x_s, x_t, y) in enumerate(loader):
        x_s, x_t, y = x_s.to(device), x_t.to(device), y.to(device)

        # Forward pass
        output = model(x_s, x_t)  # (B, L, vocab_size)

        # Compute loss (ignore padding tokens)
        B, L, V = output.shape
        loss = loss_fn(output.reshape(-1, V), y.reshape(-1))
        mask = (y != 0).reshape(-1)
        loss = (loss * mask).sum() / mask.sum()

        # Backward pass
        optim.zero_grad()
        loss.backward()
        optim.step()

        if scheduler is not None:
            scheduler.step(epoch + batch_idx / num_batches)

        total_loss += loss.item()

    return total_loss / num_batches


def valid_epoch(model, loader, loss_fn, device):
    """Validate for one epoch."""
    model.to(device)
    model.eval()
    total_loss = 0
    num_batches = len(loader)

    with torch.no_grad():
        for x_s, x_t, y in loader:
            x_s, x_t, y = x_s.to(device), x_t.to(device), y.to(device)

            # Forward pass
            output = model(x_s, x_t)

            # Compute loss
            B, L, V = output.shape
            loss = loss_fn(output.reshape(-1, V), y.reshape(-1))
            mask = (y != 0).reshape(-1)
            loss = (loss * mask).sum() / mask.sum()

            total_loss += loss.item()

    return total_loss / num_batches


# Initialize or load checkpoint
if not checkpoint_path.exists():
    checkpoint = {
        "epoch": 0,
        "loss": torch.inf,
    }
else:
    checkpoint = torch.load(checkpoint_path, map_location=device)


def main(config, checkpoint):
    # Load data
    train_set = get_dataset(train_path, source_tokenizer, target_tokenizer)
    val_set = get_dataset(val_path, source_tokenizer, target_tokenizer)
    test_set = get_dataset(test_path, source_tokenizer, target_tokenizer)

    train_loader = get_dataloader(train_set, config['batch_size'])
    val_loader = get_dataloader(val_set, config['batch_size'])
    test_loader = get_dataloader(test_set, config['batch_size'])

    # Initialize Transformer model
    model = TransformerTranslation(
        src_vocab_size=config['vocab_source'],
        tgt_vocab_size=config['vocab_target'],
        d_model=config.get('d_model', 512),
        num_heads=config.get('num_heads', 8),
        num_layers=config['num_layers'],
        d_ff=config.get('d_ff', 2048),
        dropout=config['dropout'],
        max_len=512
    )
    model.to(device)

    # Optimizer and scheduler
    if config['optimizer'] == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    elif config['optimizer'] == 'adamw':
        optim = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    else:
        optim = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)

    # Scheduler
    if config['scheduler'] == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        scheduler = CosineAnnealingWarmRestarts(optim, T_0=10, T_mult=2, eta_min=1e-6)
    else:
        scheduler = None

    # Loss function
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    # Load checkpoint if resuming
    if 'nn_state' in checkpoint:
        model.load_state_dict(checkpoint['nn_state'], strict=False)
        if 'opt_state' in checkpoint:
            optim.load_state_dict(checkpoint['opt_state'])

    # Training loop
    for epoch in range(config['num_epochs']):
        train_loss = train_epoch(model, train_loader, loss_fn, optim, scheduler, epoch, device)
        val_loss = valid_epoch(model, val_loader, loss_fn, device)

        # Save if improved
        if val_loss < checkpoint["loss"]:
            checkpoint = save_checkpoint(checkpoint_path, model, epoch, val_loss, optim, scheduler)

        # Compute BLEU scores
        bleu_val = get_bleu_transformer(model, val_loader, device)
        bleu_test = get_bleu_transformer(model, test_loader, device)

        print(f"Epoch {epoch+1}/{config['num_epochs']} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"BLEU Val: {bleu_val:.4f} Test: {bleu_test:.4f}")


if __name__ == "__main__":
    main(config, checkpoint)