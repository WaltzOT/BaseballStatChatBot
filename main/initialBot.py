from nemo.collections.nlp.models.intent_slot_classification import IntentSlotModel
from nemo.collections.nlp.data.intent_slot import IntentSlotDataset

# Load your JSON dataset
dataset_path = "baseball_queries.json"
dataset = IntentSlotDataset.from_file(dataset_path)

# Split into training and validation sets
train_data, val_data = dataset.split(0.8)
