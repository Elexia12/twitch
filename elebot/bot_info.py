import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------- CONFIG ----------------
model_id = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"
# ----------------------------------------

print("⏳ Checking GPU availability...")

use_gpu = False
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ GPU detected: {gpu_name}")

    # Check if your GPU is supported by this PyTorch
    torch_version = torch.version.cuda
    print(f"🔧 Torch CUDA version: {torch_version}")

    try:
        # Try a quick GPU operation
        torch.zeros(1).cuda()
        use_gpu = True
    except Exception as e:
        print(f"⚠️ GPU not usable with this PyTorch build: {e}")
else:
    print("⚠️ No GPU available, falling back to CPU.")

# Load tokenizer
print("⏳ Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load model
print("⏳ Loading model (this may take a bit)...")
if use_gpu:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",   # will use GPU if available
    )
    print("✅ Model loaded on GPU.")
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu",   # force CPU
    )
    print("✅ Model loaded on CPU (expect slower responses).")
