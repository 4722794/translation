# Project: translation

## Folder structure

- scripts- temp- data- archive

> Remote says hello

> > Hercules says hello from the other side

# LRschedule Branch

- In this commit, I'm optimizing on the current model.

Current progress
- Change in bleu score computation
- Ran one iteration with a pretrained model


Pending
- introduce sweeps

└─ deepl/
   ├─ training/
   │  ├─ __init__.py
   │  ├─ model.py
   │  ├─ train.py
   │  └─ README.md
   ├─ testing/
   │  ├─ __init__.py
   │  └─ bleu_score.py
   └─ inference/
      ├─ __init__.py
      ├─ infer.py
      └─ README.md