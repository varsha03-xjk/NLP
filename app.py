from transformers import pipeline

# Load the FLAN Alpaca Large model
paraphraser = pipeline("text2text-generation", model="declare-lab/flan-alpaca-large")

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr

# Load model and tokenizer
model_name = "ramsrigouthamg/t5_paraphraser"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Define style prompts
def generate_prompt(text, style):
    style_prompts = {
        "Formal": "Please rewrite the following text in a formal and professional tone:\n\n",
        "Friendly": "Please rewrite the following text in a casual and friendly tone:\n\n",
        "Poetic": "Rewrite the text in a poetic and metaphorical way, like a short verse:\n\n",
        "Gen Z": "Rewrite this text using Gen Z slang, internet expressions, abbreviations, and emojis:\n\n"
    }

    # Default if style not found
    base_prompt = style_prompts.get(style, "Rewrite the text:\n\n")
    return base_prompt + text

# Function to rewrite text in selected style
def rewrite_text(text, style):
    if not text.strip():
        return "Please enter some text."

    # Build the prompt based on the selected style
    if style == "Gen Z":
        prompt = f"Rewrite the following text in a funny Gen Z tone with slang, emojis, and internet expressions:\n\n{text}"
    elif style == "Poetic":
        prompt = f"Rewrite the following text in a poetic and artistic style:\n\n{text}"
    elif style == "Formal":
        prompt = f"Rewrite the following text in a formal, professional tone:\n\n{text}"
    elif style == "Friendly":
        prompt = f"Rewrite the following text in a friendly and conversational style:\n\n{text}"
    else:
        prompt = f"Rewrite the following text:\n\n{text}"

    # Call the model (paraphraser)
    response = paraphraser(prompt, max_length=100)[0]['generated_text']

    # Optional: clean output (remove repeated prompt from response if needed)
    return response.replace(prompt, "").strip()


# Gradio UI
import gradio as gr

# Define the interface
with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown(
        """
        <h1 style="text-align: center;">📝 Rewrite My Text</h1>
        <p style="text-align: center;">Transform your text into <b>fun, formal, poetic, or Gen Z</b> styles using AI! 🚀</p>
        """,
        elem_id="header",
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Input Text",
                placeholder="Enter your sentence here...",
                lines=4
            )
            style = gr.Dropdown(
                label="Choose Style",
                choices=["Gen Z", "Formal", "Poetic", "Friendly"],
                value="Gen Z"
            )
            submit_button = gr.Button("✨ Submit", variant="primary")
            clear_button = gr.Button("🧹 Clear")

        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="Rewritten Text",
                placeholder="Your rewritten sentence will appear here...",
                lines=4
            )

    # Button functionality
    submit_button.click(fn=rewrite_text, inputs=[input_text, style], outputs=output_text)
    clear_button.click(fn=lambda: ("", ""), inputs=[], outputs=[input_text, output_text])

# Launch the app
demo.launch(share=True)
