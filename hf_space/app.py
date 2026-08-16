import gradio as gr
import spaces
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
import torch
from threading import Thread
import os
# Model settings
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_MODEL = "tharuntej7373/qwen-finetuned-model"

tokenizer = None
model = None
@spaces.GPU
def chat(messages, max_new_tokens=512, temperature=0.7, top_p=0.9):
    global tokenizer, model
    
    if model is None:
        hf_token = os.environ.get("HF_TOKEN")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=hf_token)
        
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
            token=hf_token
        )
        
        model = PeftModel.from_pretrained(base_model, LORA_MODEL, token=hf_token)

    # Convert messages into prompt format using chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
    
    generate_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True if temperature > 0 else False,
    )
    
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()
    
    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        yield partial_text

# Define the Gradio interface to expose an API
demo = gr.Interface(
    fn=chat,
    inputs=[
        gr.JSON(label="messages"),
        gr.Slider(1, 2048, value=512, label="max_new_tokens"),
        gr.Slider(0.0, 1.0, value=0.7, label="temperature"),
        gr.Slider(0.0, 1.0, value=0.9, label="top_p")
    ],
    outputs=gr.Textbox(label="response")
)

demo.launch()
