import torch

print("Checking for AI power...")
if torch.cuda.is_available():
    print(f"Success! I found a GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU found, but that's okay! We can still use the CPU.")