# ProAlign-ADFC

This repository provides the implementation of ProAlign for sequential recommendation.

## Environment

The code was tested with the following environment:

```text
python             3.11.14
torch              2.8.0+cu128
numpy              2.3.5
pandas             2.3.3
scipy              1.16.3
scikit-learn       1.8.0
```

## Data

Place the processed Beauty dataset under:

```text
data/Beauty/
```

The folder should contain the processed `.df` files and the LLM intent embeddings used by the model.

## Run

```bash
python train.py --model_type ProAlign --data Beauty \
  --num_prototypes 256 \
  --alpha 0.4 \
  --beta_proto 0.06 \
  --num_heads_proto 4
```
