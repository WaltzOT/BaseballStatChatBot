import torch
print(torch.__version__)
print(torch.cuda.is_available())

import nemo
import nemo.collections.nlp as nemo_nlp
print("NeMo NLP toolkit installed successfully!")

from nemo.collections.nlp.models import IntentSlotModel
print("IntentSlotModel imported successfully!")
