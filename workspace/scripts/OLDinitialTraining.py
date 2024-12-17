from nemo.collections.nlp.models import IntentSlotClassificationModel
from nemo.utils.exp_manager import exp_manager
import torch

# File paths based on your structure
TRAIN_DATA_PATH = "../data/trainingData.json"
VALIDATE_DATA_PATH = "../data/testData.json"
MODEL_OUTPUT_DIR = "../models"

# Training configuration
MODEL_CONFIG = {
    "train_ds.file_path": TRAIN_DATA_PATH,
    "train_ds.batch_size": 26,
    "validation_ds.file_path": TRAIN_DATA_PATH,  # Reusing as no separate validation file
    "validation_ds.batch_size": 14,
    "optim.lr": 5e-5,
    "model.num_epochs": 10,
}

# Check for GPU or CPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training on: {DEVICE}")

# Initialize the IntentSlotClassificationModel
print("Initializing the model...")
model = IntentSlotClassificationModel.from_pretrained("bert-base-uncased", strict=False)

# Configure experiment manager (to save checkpoints and logs)
print("Setting up experiment manager...")
exp_manager_cfg = {"exp_dir": MODEL_OUTPUT_DIR, "name": "intent_slot_model"}
exp_manager(trainer=model.trainer, cfg=exp_manager_cfg)

# Update model configuration
model.cfg.update(MODEL_CONFIG)

# Start training
print("Starting training...")
model.train()
print(f"Training complete! Model saved to: {MODEL_OUTPUT_DIR}")
