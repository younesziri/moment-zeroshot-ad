Produced by `scripts/diagnose_moment.py`.

| Model | Loads strictly, emb. on | Loads strictly, emb. off | corr(sine), emb. on | corr(sine), emb. off | max abs(head bias) | max abs(W·Wᵀ − 1.41²·I) |
|---|---|---|---|---|---|---|
| MOMENT-1-small | yes | no: Unexpected key(s) in state_dict: "patch_embedding.position_embedding.pe" | -0.066 | 0.111 | 0.0000 | 0.0000 |
| MOMENT-1-base | yes | no: Unexpected key(s) in state_dict: "patch_embedding.position_embedding.pe" | 0.001 | -0.162 | 0.0000 | 0.0000 |
| MOMENT-1-large | yes | no: Unexpected key(s) in state_dict: "patch_embedding.position_embedding.pe" | 0.987 | -0.562 | 0.0121 | 1.3097 |
| small, untrained (no weights) | – | – | 0.028 | – | 0.0000 | 0.0000 |
