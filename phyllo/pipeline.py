"""PhylloEngine — the deployed, reliability-aware inference pipeline.

  image ─► quality gate ─► 4-class ViT (+ temperature) ─► OOD check (Mahalanobis + kNN on embeddings)
        ─► abstention policy ─► attention-rollout explanation ─► evidence-gated AQI

Every threshold is loaded from artifacts/ (produced by experiments/e04 and e05, fit on train/val only).
"""
import io
import json
import time

import joblib
import numpy as np
import torch
from PIL import Image, ImageOps
from scipy.special import softmax

from . import quality
from .features import DEVICE, ViTEncoder, attention_rollout, overlay_png_b64
from .paths import ARTIFACTS, MODEL_3CLASS, MODEL_4CLASS, ROOT
from .reliability import KNNScore, Mahalanobis

LEGACY_RIDGE = ROOT / "legacy" / "results" / "aqi_regression" / "aqi_ridge_model.joblib"

HEADLINES = {
    "ACCEPT": "Prediction accepted — the image matches validated training evidence.",
    "ABSTAIN_UNCERTAIN": "This image does not contain sufficient validated evidence for a reliable prediction (low confidence).",
    "ABSTAIN_UNFAMILIAR": "This image does not contain sufficient validated evidence for a reliable prediction (it is unlike the validated leaf data).",
    "NOT_A_LEAF": "The image does not appear to contain a leaf.",
    "REJECT_QUALITY": "The image quality is too low to analyse.",
}


class PhylloEngine:
    def __init__(self, load_aqi_model=True):
        t0 = time.time()
        cfg = json.loads((ARTIFACTS / "reliability_ft4.json").read_text(encoding="utf-8"))
        arr = np.load(ARTIFACTS / "reliability_ft4.npz", allow_pickle=True)
        self.cfg = cfg
        self.T = float(cfg["temperature"])
        self.thr = cfg["thresholds"]
        self.classes = cfg["classes"]
        self.maha = Mahalanobis.from_state(arr["maha_means"], arr["maha_precision"], arr["maha_classes"])
        self.knn = KNNScore(cfg.get("knn_k", 5))
        self.knn.bank_ = arr["knn_bank"]

        self.clf = ViTEncoder(MODEL_4CLASS, eager_attention=True)
        assert [self.clf.id2label[i] for i in range(len(self.classes))] == self.classes, "class order mismatch"

        self.aqi_evidence = json.loads((ARTIFACTS / "aqi_evidence.json").read_text(encoding="utf-8"))
        self.aqi_enabled = load_aqi_model
        if load_aqi_model:
            self.aqi_encoder = ViTEncoder(MODEL_3CLASS)
            self.aqi_ridge = joblib.load(LEGACY_RIDGE)
            self.aqi_domain = KNNScore(self.aqi_evidence["applicability_knn_k"])
            self.aqi_domain.bank_ = np.load(ARTIFACTS / "aqi_domain_ft3.npz")["bank"]
        self.load_seconds = round(time.time() - t0, 1)

    # ------------------------------------------------------------------
    @staticmethod
    def read_image(data: bytes) -> Image.Image:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")

    def predict(self, image: Image.Image, explain=True, show_exploratory_aqi=False):
        t0 = time.time()
        q = quality.assess(image, noise_warn=self.cfg.get("noise_sigma_warn"))
        enc = self.clf.encode([image], attentions=explain)
        logits, emb = enc["logits"], enc["embeddings"]
        p_cal = softmax(logits / self.T, axis=1)[0]
        p_raw = softmax(logits, axis=1)[0]
        top = int(p_cal.argmax())
        top_class = self.classes[top]
        conf = float(p_cal[top])
        maha_s = float(self.maha.score(emb)[0])
        knn_s = float(self.knn.score(emb)[0])
        in_dist = maha_s >= self.thr["mahalanobis"] and knn_s >= self.thr["knn"]

        reasons = []
        if not q.passed:
            status = "REJECT_QUALITY"
            reasons += q.issues
        elif top_class == "Not_a_leaf":
            status = "NOT_A_LEAF"
            reasons.append(f"The classifier assigns {conf:.0%} calibrated probability to 'Not_a_leaf'.")
        elif not in_dist:
            status = "ABSTAIN_UNFAMILIAR"
            if maha_s < self.thr["mahalanobis"]:
                reasons.append("Embedding is far from every validated leaf class (Mahalanobis distance above the 95%-validation threshold).")
            if knn_s < self.thr["knn"]:
                reasons.append("No sufficiently similar image exists in the validated training set (kNN similarity below threshold).")
        elif conf < self.thr["confidence"]:
            status = "ABSTAIN_UNCERTAIN"
            second = int(np.argsort(p_cal)[-2])
            reasons.append(f"Calibrated confidence {conf:.0%} is below the {self.thr['confidence']:.0%} acceptance threshold; "
                           f"runner-up is {self.classes[second]} ({p_cal[second]:.0%}).")
        else:
            status = "ACCEPT"
            reasons.append(f"Calibrated confidence {conf:.0%} ≥ {self.thr['confidence']:.0%} and the embedding lies inside the validated leaf distribution.")
        reasons += q.warnings

        explanation = None
        if explain:
            mask = attention_rollout(enc["attentions"][0], discard_ratio=0.0)
            explanation = {"method": "Attention rollout (Abnar & Zuidema, 2020) over all 12 layers",
                           "note": "Shows which image patches the transformer's [CLS] token aggregated. It is an "
                                   "attribution aid, not proof of the botanical reason for the prediction.",
                           "heatmap_png_base64": overlay_png_b64(image, mask)}

        aqi = self._aqi(image, status, top_class, show_exploratory_aqi)

        accepted = status == "ACCEPT"
        return {
            "decision": {"status": status, "accepted": accepted, "headline": HEADLINES[status], "reasons": reasons},
            "phyllotaxy": {
                "class": top_class,
                "accepted": accepted,
                "confidence": round(conf, 4),
                "raw_confidence": round(float(p_raw[top]), 4),
                "probabilities": {c: round(float(p), 4) for c, p in zip(self.classes, p_cal)},
            },
            "reliability": {
                "temperature": round(self.T, 3),
                "confidence_threshold": self.thr["confidence"],
                "mahalanobis_score": round(maha_s, 2), "mahalanobis_threshold": round(self.thr["mahalanobis"], 2),
                "knn_similarity": round(knn_s, 4), "knn_threshold": round(self.thr["knn"], 4),
                "in_distribution": bool(in_dist),
            },
            "quality": q.to_dict(),
            "explanation": explanation,
            "aqi": aqi,
            "model": {"device": str(DEVICE), "classifier": "ViT-B/16 4-class (fine-tuned)",
                      "embedding_dimensions": int(emb.shape[1]), "latency_ms": round((time.time() - t0) * 1000)},
        }

    # ------------------------------------------------------------------
    def _aqi(self, image, status, top_class, show_exploratory):
        ev = self.aqi_evidence
        evidence = {"pairs": ev["n_pairs"], "pooled_cv_r2": round(ev["pooled_cv_R2_ft3_ridge"], 3),
                    "permutation_p": round(ev["permutation_p_value"], 3), "supported": ev["supported"]}
        base = {"value": None, "unit": "AQI (index, dimensionless)", "evidence": evidence}
        if status != "ACCEPT":
            return {**base, "status": "NOT_APPLICABLE", "message": "No AQI: the image was not accepted as a validated leaf."}
        if top_class != "Whorled":
            return {**base, "status": "NOT_APPLICABLE",
                    "message": f"No AQI: verified image–AQI pairs exist only for Whorled leaves, not {top_class}."}
        if not self.aqi_enabled:
            return {**base, "status": "WITHHELD", "message": "AQI model not loaded."}
        e = self.aqi_encoder.encode([image])["embeddings"]
        sim = float(self.aqi_domain.score(e)[0])
        inside = sim >= ev["applicability_threshold"]
        raw = float(self.aqi_ridge.predict(e)[0])
        applicability = {"similarity": round(sim, 4), "threshold": round(ev["applicability_threshold"], 4), "inside": bool(inside)}
        if ev["supported"] and inside:
            return {**base, "status": "EXPLORATORY", "value": round(raw, 1), "applicability": applicability,
                    "message": "Exploratory AQI estimate (supported by cross-validated evidence; not regulatory-grade)."}
        out = {**base, "status": "WITHHELD", "applicability": applicability, "message": ev["decision"]}
        if not inside:
            out["message"] += " The image is also outside the applicability domain of the 46 training pairs."
        if show_exploratory:
            out["exploratory_value"] = round(raw, 1)
            out["exploratory_warning"] = "Shown on request only. Not better than predicting the mean AQI under cross-validation."
        return out

    def info(self):
        return {"temperature": self.T, "thresholds": self.thr, "classes": self.classes,
                "device": str(DEVICE), "cuda": torch.cuda.is_available(), "load_seconds": self.load_seconds,
                "aqi_supported": self.aqi_evidence["supported"]}
