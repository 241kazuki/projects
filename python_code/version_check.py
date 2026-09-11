import torch
import transformers
import accelerate
import pandas as pd
import sklearn
import platform

print(f"OS: {platform.system()} {platform.release()}")
print(f"Python Version: {platform.python_version()}")
print("-" * 30)
print(f"torch: {torch.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"accelerate: {accelerate.__version__}")
print(f"pandas: {pd.__version__}")
print(f"scikit-learn: {sklearn.__version__}")

# GPUが認識されているか、およびその名称
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU: Not Available")