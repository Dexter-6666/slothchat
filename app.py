import gradio as gr
from transformers import pipeline
import torch
import re
import os

print("Loading Qwen 1.5B...")
pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

def summarize_file(file):
    if file is None:
        return "Please upload a chat export."
    
    # Read the uploaded file
    with open(file.name, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Regex to parse WhatsApp format
    messages = []
    for line in raw_text.splitlines():
        match = re.match(r"\[?(\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}(?::\d{2})?\s?[AP]M)\]?\s?-\s?([^:]+):\s?(.*)", line)
        if match:
            sender = match.group(2).strip()
            text = match.group(3).strip()
            messages.append(f"{sender}: {text}")
        elif messages and line.strip():
            messages[-1] += f" {line.strip()}"

    if not messages:
        return "Could not parse the chat format. Ensure it's a standard WhatsApp export."

    # Grab the last 100 messages (The Overnight Context)
    chat_log = "\n".join(messages[-100:])
    
    system_prompt = "You are an AI assistant designed to save time. Read the following group chat and provide a maximum 3-bullet summary of the important details. Then, list the Action Items with the person responsible."
    user_prompt = f"Chat log:\n{chat_log}\n\nSummary and Action Items:"
    
    messages_prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    outputs = pipe(messages_prompt, max_new_tokens=250, do_sample=False, temperature=0.1)
    return outputs[0]["generated_text"][-1]["content"]

demo = gr.Interface(
    fn=summarize_file,
    inputs=gr.File(label="Upload WhatsApp Chat Export (.txt)"),
    outputs=gr.Textbox(label="SlothChat Summary"),
    title="🦥 SlothChat",
    description="Upload your exported group chat (.txt). Get a 3-bullet summary and action items. Runs entirely locally on an open-weight model."
)

demo.launch(server_name="0.0.0.0", server_port=7860)
