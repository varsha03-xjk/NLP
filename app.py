import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load model and tokenizer with caching to avoid reloading
@st.cache_resource
def load_model():
    model_name = "ramsrigouthamg/t5_paraphraser"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Rewrite logic
def rewrite_text(text, style):
    if not text.strip():
        return "Please enter some text."

    # Style-specific prompts
    style_prompts = {
        "Gen Z": "Rewrite the following text in a funny Gen Z tone with slang, emojis, and internet expressions:\n\n",
        "Poetic": "Rewrite the following text in a poetic and artistic style:\n\n",
        "Formal": "Rewrite the following text in a formal, professional tone:\n\n",
        "Friendly": "Rewrite the following text in a friendly and conversational style:\n\n",
    }

    prompt = style_prompts.get(style, "Rewrite the following text:\n\n") + text

    # Encode and generate
    input_ids = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)

    outputs = model.generate(
        input_ids=input_ids,
        max_length=150,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if prompt in result:
        result = result.replace(prompt, "").strip()

    return result

# Streamlit UI
st.set_page_config(page_title="Rewrite My Text", page_icon="📝", layout="centered")
st.title("📝 Rewrite My Text")
st.markdown("Transform your text into **Formal**, **Friendly**, **Poetic**, or **Gen Z** styles using AI!")

text_input = st.text_area("Enter your text:", height=150)
style = st.selectbox("Choose a style:", ["Gen Z", "Formal", "Poetic", "Friendly"])

if st.button("✨ Rewrite"):
    rewritten = rewrite_text(text_input, style)
    st.subheader("🔁 Rewritten Text:")
    st.success(rewritten)
