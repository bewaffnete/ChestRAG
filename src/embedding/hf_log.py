from huggingface_hub import login
from env_config import EMBEDDING_MODEL_TOKEN

def hf_login():
    login(EMBEDDING_MODEL_TOKEN)

if __name__ == '__main__':
    hf_login()