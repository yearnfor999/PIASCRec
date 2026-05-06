# PIASCRec
ProAlign-ADFC
This repository provides the implementation of PIASCRec, a LLM-enhanced sequential recommendation framework with prospective intent alignment and prototype-guided semantic continuity modeling.

Environment
The code was tested with the following environment:

python             3.11.14
torch              2.8.0+cu128
numpy              2.3.5
pandas             2.3.3
scipy              1.16.3
scikit-learn       1.8.0
Data
Download the processed Beauty dataset from Google Drive, and place it under:

data/Beauty/
The folder should contain the processed data files and LLM intent embeddings:

data/Beauty/
+-- train_data.df
+-- val_data.df
+-- test_data.df
+-- data_statis.df
+-- usr_intent_emb.pkl
+-- itm_intent_emb.pkl
+-- sim_user_100.pkl
`-- 3large_emb.pickle
Run
python train.py --model_type ProAlign --data Beauty \
  --num_prototypes 256 \
  --alpha 0.4 \
  --beta_proto 0.06 \
  --num_heads_proto 4
