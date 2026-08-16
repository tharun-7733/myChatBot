from beta9 import Image
from beta9 import Image
from beam.integrations import VLLM, VLLMArgs

# Define the integration and the cloud hardware you need
qwen_lora_api = VLLM(
    name="my-qwen-api",
    cpu=4,
    memory="16Gi",
    gpu="A10G",  # The A10G is cost-effective and easily handles a 1.5B model
    secrets=["HF_TOKEN"], # This tells Beam to inject your Hugging Face token securely
    vllm_version="0.6.2",
    custom_image=Image(
        python_version="python3.11",
        python_packages=["cachetools==5.3.3"]
    ),
    vllm_args=VLLMArgs(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        served_model_name=["my_custom_model"],
        enable_lora=True,
        # The repository where you uploaded your fine-tuned model
        lora_modules=["my_custom_model=tharuntej7373/qwen-finetuned-model"],
        max_model_len=8192,
        gpu_memory_utilization=0.90
    )
)
