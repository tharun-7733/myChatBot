from beta9 import Image
from beam.integrations import VLLM, VLLMArgs

# Define a custom image to fix the vLLM + cachetools bug
custom_image = Image(python_version="python3.11").add_python_packages(["cachetools==5.5.2"])

# Define the integration and the cloud hardware you need
qwen_lora_api = VLLM(
    name="my-qwen-api",
    image=custom_image,
    cpu=4,
    memory="16Gi",
    gpu="A10G",  # The A10G is cost-effective and easily handles a 1.5B model
    secrets=["HF_TOKEN"], # This tells Beam to inject your Hugging Face token securely
    vllm_version="0.6.2", # Fixed version to bypass the LoRALRUCache bug
    vllm_args=VLLMArgs(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        served_model_name=["my_custom_model"],
        enable_lora=True,
        # The repository where you uploaded your fine-tuned model
        lora_modules=["my_custom_model=tharuntej7373/qwen-finetuned-model"]
    )
)
