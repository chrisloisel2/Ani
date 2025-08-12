# Ani

Ce projet propose un petit modèle Seq2Seq optimisé pour les puces Apple Silicon (M1/M2/M3). Le notebook `ani.ipynb` configure automatiquement TensorFlow pour tirer parti de la puce M3 via `tensorflow-metal` et la **mixed precision**.

## Précautions d'entraînement
- Installez les dépendances principales :
  ```bash
  pip install -U pip wheel setuptools
  pip install -U tensorflow tensorflow-metal
  pip install -q faiss-cpu datasets pandas sentence-transformers sacrebleu tf-keras
  ```
- L'utilitaire `setup_apple_silicon()` choisit un `batch_size` conseillé :
  - GPU Metal disponible → batch ≈ **128**
  - CPU seul → batch ≈ **32**
- Pré-entraînement conseillé sur [SQuAD](https://huggingface.co/datasets/squad)
  - ~**50** époques
  - Scheduler LR warmup+cosine
- Affinez ensuite sur votre jeu de conversations (ex. `Shirayuki`)
  - ~**10** époques
  - Conservez un petit set de validation pour surveiller `ROUGE-L`

Un entraînement plus long et des jeux de données massifs sont nécessaires pour approcher les performances de modèles comme DeepSeek.
