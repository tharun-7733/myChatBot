from beam.integrations import VLLM, VLLMArgs

# Define the integration and the cloud hardware you need
qwen_lora_api = VLLM(
    name="my-qwen-api",
    cpu=4,
    memory="16Gi",
    gpu="A10G",  # The A10G is cost-effective and easily handles a 1.5B model
    secrets=["HF_TOKEN"], # This tells Beam to inject your Hugging Face token securely
    vllm_args=VLLMArgs(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        served_model_name=["my_custom_model"],
        enable_lora=True,
        # TODO: Replace 'your_hf_username/my-qwen-lora' with your actual Hugging Face model repository!
        lora_modules=["my_custom_model=your_hf_username/my-qwen-lora"]
    )
)
