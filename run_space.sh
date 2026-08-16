#!/bin/bash
set -e

echo "Starting vLLM server in the background..."
python3 -m vllm.entrypoints.openai.api_server \
  --port 8001 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --served-model-name my_custom_model \
  --enable-lora \
  --lora-modules my_custom_model=tharuntej7373/qwen-finetuned-model \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 &

echo "Waiting for vLLM to start on port 8001..."
# Wait until the /v1/models endpoint returns a successful response
while ! curl -s -f http://127.0.0.1:8001/v1/models > /dev/null; do
    echo "Waiting..."
    sleep 5
done
echo "vLLM is up and running!"

# Start the FastAPI proxy on port 7860
export BEAM_URL="http://127.0.0.1:8001/v1"
export BEAM_API_KEY="local-no-auth"
export PORT=7860

echo "Starting FastAPI proxy on port 7860..."
cd backend
exec uvicorn server:app --host 0.0.0.0 --port 7860
