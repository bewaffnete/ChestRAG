from transformers import AutoProcessor, SiglipVisionModel
import torch
from embedding.hf_log import hf_login

def model_and_processor_instance():
    hf_login()
    processor = AutoProcessor.from_pretrained("google/medsiglip-448", use_fast=True)
    model = SiglipVisionModel.from_pretrained("google/medsiglip-448")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    model.to(device).eval()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    return model, processor, device

if __name__ == "__main__":
    model, processor, device = model_and_processor_instance()
