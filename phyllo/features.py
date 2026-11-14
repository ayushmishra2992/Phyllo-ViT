"""Model loading, batched embedding/logit extraction and attention-rollout explanations."""
import base64
import io
import os

import numpy as np
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor, ViTModel

from .paths import PRETRAINED_ID

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ViTEncoder:
    """Wraps either a fine-tuned classifier (has logits) or the frozen pretrained backbone."""

    def __init__(self, source, classifier=True, eager_attention=False):
        kw = {"attn_implementation": "eager"} if eager_attention else {}
        self.processor = ViTImageProcessor.from_pretrained(str(source))
        if classifier:
            self.model = ViTForImageClassification.from_pretrained(str(source), **kw)
            self.backbone = self.model.vit
            self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        else:
            self.model = ViTModel.from_pretrained(str(source), add_pooling_layer=False, **kw)
            self.backbone = self.model
            self.id2label = None
        self.model.to(DEVICE).eval()
        self.classifier = classifier

    @torch.no_grad()
    def encode(self, images, batch_size=16, attentions=False):
        """Returns dict(embeddings (n,768), logits (n,C) or None, attentions list or None)."""
        embs, logits, atts = [], [], []
        for i in range(0, len(images), batch_size):
            batch = [im.convert("RGB") for im in images[i:i + batch_size]]
            px = self.processor(images=batch, return_tensors="pt")["pixel_values"].to(DEVICE)
            out = self.backbone(pixel_values=px, output_attentions=attentions)
            cls = out.last_hidden_state[:, 0, :]
            embs.append(cls.float().cpu().numpy())
            if self.classifier:
                logits.append(self.model.classifier(cls).float().cpu().numpy())
            if attentions:
                atts.append([a.float().cpu() for a in out.attentions])
        return {
            "embeddings": np.concatenate(embs),
            "logits": np.concatenate(logits) if logits else None,
            "attentions": atts if attentions else None,
        }


def load_pretrained():
    return ViTEncoder(PRETRAINED_ID, classifier=False)


def attention_rollout(attentions, discard_ratio=0.0):
    """Abnar & Zuidema (2020). attentions: list over layers of (1, heads, T, T) tensors."""
    with torch.no_grad():
        n = attentions[0].shape[-1]
        result = torch.eye(n)
        for a in attentions:
            a = a[0].mean(0)
            if discard_ratio > 0:
                flat = a.flatten()
                k = int(flat.numel() * discard_ratio)
                idx = flat.argsort()[:k]
                flat[idx] = 0
                a = flat.view(n, n)
            a = a + torch.eye(n)
            a = a / a.sum(-1, keepdim=True)
            result = a @ result
        mask = result[0, 1:]
        side = int(mask.numel() ** 0.5)
        mask = mask.reshape(side, side).numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-12)
        return mask


def overlay_png_b64(image, mask, size=320):
    """Blend a [0,1] heatmap onto the image and return a base64 PNG string."""
    import matplotlib
    matplotlib.use("Agg")

    img = image.convert("RGB").resize((size, size))
    heat = Image.fromarray((mask * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)
    colored = (matplotlib.colormaps["inferno"](np.asarray(heat) / 255.0)[..., :3] * 255).astype(np.uint8)
    blended = Image.blend(img, Image.fromarray(colored), alpha=0.45)
    buf = io.BytesIO()
    blended.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
