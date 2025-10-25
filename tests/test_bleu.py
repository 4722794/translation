#!/usr/bin/env python3
"""Test BLEU score computation to ensure it works before training."""

import torch
from setup import get_tokenizer, get_dataset, get_dataloader, get_model, get_bleu
from pathlib import Path

def test_bleu():
    """Test that BLEU score computation works correctly."""
    device = torch.device('cpu')
    data_path = Path('data')
    test_path = data_path / 'test/translations.csv'
    source_tokenizer_path = data_path / 'tokenizer_en.model'
    target_tokenizer_path = data_path / 'tokenizer_fr.model'

    print('Loading tokenizers...')
    source_tokenizer = get_tokenizer(source_tokenizer_path)
    target_tokenizer = get_tokenizer(target_tokenizer_path)

    print('Loading test dataset...')
    test_set = get_dataset(test_path, source_tokenizer, target_tokenizer)
    test_loader = get_dataloader(test_set, batch_size=32)

    print('Creating small test model...')
    model = get_model(5001, 5001, 64, 128, 0.1, 0.1, num_layers=1)
    model.to(device)

    print('Computing BLEU score...')
    score = get_bleu(model, test_loader, device)
    print(f'BLEU score: {score:.4f}')

    # Verify score is valid
    assert 0.0 <= score <= 1.0, f"BLEU score should be between 0 and 1, got {score}"

    print('\n✓ Test passed! BLEU computation is working.')
    return True

if __name__ == "__main__":
    test_bleu()
