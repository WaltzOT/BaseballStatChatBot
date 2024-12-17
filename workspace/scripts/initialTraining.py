import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from nemo.collections.nlp.models import IntentSlotClassificationModel
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
import os

def main(cfg: DictConfig) -> None:
    logging.info(f'Config Params:\n {OmegaConf.to_yaml(cfg)}')

    # Setup Trainer
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    # Initialize Model
    model = IntentSlotClassificationModel(cfg.model, trainer=trainer)

    # Training
    logging.info("Starting training...")
    trainer.fit(model)

    # Save model to models directory
    model_path = os.getenv('MODEL_DIR', './models') + "/baseball_stat_model.nemo"
    model.save_to(model_path)
    logging.info(f"Model saved to: {model_path}")

if __name__ == "__main__":
    import hydra
    from omegaconf import OmegaConf
    hydra.main(config_path="../data", config_name="baseball_chatbot_config")(main)
