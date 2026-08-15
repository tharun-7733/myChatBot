"""
Direct deploy script - bypasses the broken CLI version mismatch.
Calls deploy() on the VLLM object directly without going through the CLI.
"""
from beam_deploy import qwen_lora_api

print("Starting deployment of qwen_lora_api ...")
result, ok = qwen_lora_api.deploy(name="my-qwen-api")
if ok:
    print(f"✅ Deployed successfully! Deployment ID: {result.get('deployment_id')}")
else:
    print(f"❌ Deployment failed: {result}")
