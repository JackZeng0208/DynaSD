# Improving Multi-candidate Speculative Decoding
[Paper link](https://arxiv.org/abs/2409.10644)

# News
- October 28, 2024: Camera-ready version of paper is available on Arxiv
- October 7th, 2024: The paper is accepted by NeurIPS 2024 ENLSP Workshop!

## Installation
```bash
pip install -r requirements.txt
```

## Code Explanation
- `dynamic_mcsd_inference.py` : Dynamic MCSD Configuration Inference (ignoring quality)
- `static_ti_mcsd_inference.py`: Static MCSD Configuration Inference (considering quality)

All results and figure generation code are stored in `Dyna/results/` folder