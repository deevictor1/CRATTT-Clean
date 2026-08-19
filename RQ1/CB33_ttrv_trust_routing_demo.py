# ============================================================
# BLOCK 33: TTRV Trust-Routing Demo — Gradio interface
# Wraps the actual validated pipeline (b24_run_dino,
# b24_compute_snr, b24_apply_gate) in an interactive UI.
# Requires: dino_model with LoRA injected, B24 calibrated
# (same dependencies as Phase 6 / Block 24e).
# ============================================================
!pip install gradio --quiet

import gradio as gr
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont
from imagecorruptions import corrupt as ic_corrupt

CORRUPTION_OPTIONS = ["none", "gaussian_noise", "motion_blur", "snow", "contrast", "fog"]

def draw_detections(image_np, dets, verified_boxes):
    img = PILImage.fromarray(image_np.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)
    verified_set = {tuple(b.tolist()) for b in verified_boxes}

    for d in dets:
        box = d["box"].tolist()
        is_verified = tuple(d["box"].tolist()) in verified_set
        color = (29, 158, 117) if is_verified else (186, 117, 23)  # teal / amber
        draw.rectangle(box, outline=color, width=3)
        label = f'{d["label"]} ({"verified" if is_verified else "flagged"})'
        draw.rectangle([box[0], box[1] - 16, box[0] + len(label) * 7, box[1]], fill=color)
        draw.text((box[0] + 2, box[1] - 15), label, fill=(255, 255, 255))

    return np.array(img)

def run_demo(image, corruption, severity):
    if image is None:
        return None, "Upload an image first."

    raw_img = np.array(PILImage.fromarray(image).convert("RGB"))

    if corruption == "none":
        c_img = raw_img
    else:
        c_img = ic_corrupt(raw_img, corruption_name=corruption, severity=int(severity))

    dets = b24_compute_snr(c_img)
    if not dets:
        return c_img, "No detections found on this image."

    v_boxes, v_labels, v_rjoints = b24_apply_gate(dets)
    annotated = draw_detections(c_img, dets, v_boxes)

    summary = (
        f"{len(dets)} total detections | "
        f"{len(v_boxes)} verified (auto-accept) | "
        f"{len(dets) - len(v_boxes)} flagged for review\n"
        f"Gate threshold (tau): {B24['tau_internal']:.3f}"
    )
    return annotated, summary

with gr.Blocks(title="TTRV Trust-Routing Demo") as demo:
    gr.Markdown("## TTRV gate demo — verify before you trust")
    gr.Markdown(
        "Upload an image, optionally apply a corruption, and see which "
        "detections the gate would auto-accept (teal) versus flag for "
        "human review (amber)."
    )
    with gr.Row():
        with gr.Column():
            img_in = gr.Image(label="Input image", type="numpy")
            corruption_in = gr.Dropdown(CORRUPTION_OPTIONS, value="gaussian_noise", label="Corruption")
            severity_in = gr.Slider(1, 5, value=5, step=1, label="Severity")
            run_btn = gr.Button("Run gate")
        with gr.Column():
            img_out = gr.Image(label="Verified (teal) vs flagged (amber)")
            summary_out = gr.Textbox(label="Summary", lines=3)

    run_btn.click(run_demo, inputs=[img_in, corruption_in, severity_in], outputs=[img_out, summary_out])

demo.launch(share=True, debug=False)
