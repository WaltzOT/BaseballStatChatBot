import os
import json
from nemo.collections.nlp.models import IntentSlotClassificationModel
from nemo.utils import logging

def load_and_test_model(model_path, test_data_path):
    # Load the trained model
    logging.info(f"Loading model from: {model_path}")
    model = IntentSlotClassificationModel.restore_from(restore_path=model_path)
    
    # Load test data
    with open(test_data_path, "r") as f:
        data = json.load(f)
    
    queries = [item["query"] for item in data]
    logging.info("Running inference on training data...")

    # Perform inference
    pred_intents, pred_slots = model.predict_from_examples(queries, model.model_cfg.test_ds)

    # Display results
    for query, intent, slots in zip(queries, pred_intents, pred_slots):
        print(f"Query: {query}")
        print(f"Predicted Intent: {intent}")
        print(f"Predicted Slots: {slots}")
        print("-" * 50)

if __name__ == "__main__":
    # Paths
    model_path = os.getenv("MODEL_DIR", "./models") + "/baseball_stat_model.nemo"
    training_data_path = "./data/trainingData.json"

    # Test the model
    load_and_test_model(model_path, training_data_path)