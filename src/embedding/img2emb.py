from typing import List
from PIL import Image
import torch
import io

def get_image_embedding(image_bytes, processor, model, device) -> List[float] | None:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            emb = outputs.pooler_output
            emb = emb / emb.norm(dim=-1, keepdim=True)

        return emb[0].cpu().float().numpy().tolist()
    except Exception as e:
        print(f"Image embedding error: {e}")
        return None
