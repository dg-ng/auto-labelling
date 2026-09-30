"""Gradio label demo — text paragraph or image → unsupervised + semi-supervised label."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import gradio as gr
from PIL import UnidentifiedImageError

from app.inference.text import predict_text_unsupervised, predict_text_semisupervised
from app.inference.image import predict_image_unsupervised, predict_image_semisupervised


def label_text(text: str):
    if not text or not text.strip():
        return "—", "—"
    try:
        unsup = predict_text_unsupervised(text.strip())
        semisup = predict_text_semisupervised(text.strip())
    except FileNotFoundError as e:
        return str(e), str(e)
    except Exception as e:
        return f"Error: {e}", f"Error: {e}"
    return unsup, semisup


def label_image(image):
    if image is None:
        return "—", "—"
    try:
        from PIL import Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        unsup = predict_image_unsupervised(image)
        semisup = predict_image_semisupervised(image)
    except (UnidentifiedImageError, OSError) as e:
        return f"Unsupported image: {e}", f"Unsupported image: {e}"
    except FileNotFoundError as e:
        return str(e), str(e)
    except Exception as e:
        return f"Error: {e}", f"Error: {e}"
    return unsup, semisup


with gr.Blocks(title="Auto-Label Demo") as demo:
    gr.Markdown("# Auto-Label Demo\nPaste a news paragraph **or** upload an animal image.")

    with gr.Tab("Text"):
        gr.Markdown(
            "**Text pipeline** — 16 news categories  \n"
            "Unsupervised: MiniLM-L6-v2 + KMeans &nbsp;|&nbsp; Semi-supervised: DistilBERT pseudo-labeling"
        )
        text_input = gr.Textbox(
            label="News paragraph",
            placeholder="Paste a news article excerpt here…",
            lines=5,
        )
        text_btn = gr.Button("Predict")
        with gr.Row():
            text_unsup_out = gr.Textbox(label="Unsupervised label (MiniLM+KMeans)", interactive=False)
            text_semisup_out = gr.Textbox(label="Semi-supervised label (DistilBERT)", interactive=False)
        text_btn.click(fn=label_text, inputs=text_input, outputs=[text_unsup_out, text_semisup_out])

    with gr.Tab("Image"):
        gr.Markdown(
            "**Image pipeline** — 10 animal categories  \n"
            "Unsupervised: CLIP + DEC &nbsp;|&nbsp; Semi-supervised: CLIP + MLP (pseudo-labeling A)"
        )
        image_input = gr.Image(label="Upload animal image", type="pil")
        image_btn = gr.Button("Predict")
        with gr.Row():
            image_unsup_out = gr.Textbox(label="Unsupervised label (CLIP+DEC)", interactive=False)
            image_semisup_out = gr.Textbox(label="Semi-supervised label (CLIP+MLP)", interactive=False)
        image_btn.click(fn=label_image, inputs=image_input, outputs=[image_unsup_out, image_semisup_out])

if __name__ == "__main__":
    demo.launch()
