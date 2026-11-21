# LLM Guard prompt injection classifier

This checkpoint is a LoRA adapter and trained pointer head for a binary prompt injection classifier. It was fine-tuned from the Kev-4B model on `Qwen/Qwen3.5-4B-Base`. Use the backbone at revision `1001bb4d826a52d1f399e183466143f4da7b741b` with the bundled `kev` source and `backend` serving code. The backbone weights are not included here.

The adapter is `adapter_model.safetensors`; the pointer head is `head.pt`. The tokenizer and training metadata are included. The classifier's question text and criteria are defined in `backend/src/backend/main.py`.

The selected decision threshold is 0.6483194828033447, chosen on a validation split. On the untouched 942-message test split: accuracy 96.28%, F1 96.83%, ROC-AUC 0.992, injection recall 96.92%, and false positive rate 4.62%. Confusion counts were 535 true positives, 372 true negatives, 18 false positives, and 17 false negatives. The guard can miss attacks and block benign messages; do not rely on it as a sole security control.

Training configuration and metrics are in this directory. The dataset is not bundled. See the root README for setup and serving instructions.
