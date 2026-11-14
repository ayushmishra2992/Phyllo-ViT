# Model card — Phyllo-ViT v3 (deployed pipeline)

**Model:** ViT-B/16 (`google/vit-base-patch16-224-in21k`) fine-tuned for 4 classes (Alternate, Opposite, Whorled, Not_a_leaf); 8 epochs, AdamW 2e-5, batch 4, fp16. Plus post-hoc reliability layer (quality gate, temperature T = 0.5, Mahalanobis + kNN OOD, abstention).

**Intended use:** research demonstration and teaching; recognising the leaf arrangement of shoots similar to the training photos, and *declining* to answer otherwise.

**Not intended for:** species identification, pollution or AQI measurement, regulatory or health decisions.

**Training data:** 265 unique leaf/shoot photos (88/83/94) from a single collection with unknown species/session metadata + 150 COCO-2017 images as Not_a_leaf. Random 70/15/15 split.

**Evaluation (held-out 65 images):** 64/65 = 98.5% (Wilson 95% CI 91.8–99.7%). With the abstention policy: 41/42 leaves accepted, 100% correct on accepted; 23/23 non-leaves rejected; 151/168 corrupted images not accepted.

**Known limitations**
* Classes are confounded with colour and photo orientation (colour-only probe 95.6%, orientation/size probe 76.7%). The model may be recognising species/sessions rather than arrangement.
* Unseen-arrangement detection was only testable with a frozen backbone (AUROC 0.98), and in this dataset largely measures novel-species detection.
* Heavy sensor noise: 17 of 42 noisy test images were still accepted.
* Temperature could not be fit freely because validation accuracy is 100%; constrained to [0.5, 5].
* Uploads are EXIF-rotated before inference; 15 training images had EXIF rotation that training ignored.
* AQI: withheld. 46 Whorled image–AQI pairs show no cross-validated skill (R² ≈ −0.17 vs −0.13 for the mean; permutation p = 0.45).

**Thresholds:** fit on validation leaves at 95% TPR — `artifacts/reliability_ft4.json`.

**Reproduce:** `RUN_EXPERIMENTS.bat` → `results/RESULTS.md`.
