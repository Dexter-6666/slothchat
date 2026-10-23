import gradio as gr
from transformers import pipeline
import torch

# Load the model once when the app starts
print("Loading Qwen 1.5B...")
pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

def summarize(chat_log):
    if not chat_log.strip():
        return "Please paste a chat log."
    
    system_prompt = "You are an AI assistant designed to save time. Read the following group chat and provide a maximum 3-bullet summary of the important details. Then, list the Action Items with the person responsible."
    user_prompt = f"Chat log:\n{chat_log}\n\nSummary and Action Items:"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    outputs = pipe(messages, max_new_tokens=200, do_sample=False, temperature=0.1)
    return outputs[0]["generated_text"][-1]["content"]

# Create the web UI
demo = gr.Interface(
    fn=summarize,
    inputs=gr.Textbox(lines=10, label="Paste your group chat here..."),
    outputs=gr.Textbox(label="SlothChat Summary"),
    title="🦥 SlothChat",
    description="Paste your chaotic group chat. Get a 3-bullet summary and action items. Runs entirely locally on an open-weight model."
)

demo.launch()
