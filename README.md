# DeepFake Noise + Wavelet + ML Forensics

This clean project uses DeepFakeFusion-304K only.

Method:
image -> canonical preprocessing -> normal/spatial noise features and/or wavelet features -> ML -> REAL(0) / FAKE(1).

The supplied train/val/test split is respected. The 50-image holdout is kept outside those splits.

Do not mix this project with previous CIFAKE, AI-art, or other datasets.
