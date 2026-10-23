Produced by `scripts/diagnose_moment.py`.

| Model | HF revision | Loads strictly, emb. on | Loads strictly, emb. off | corr(sine), emb. on | corr(sine), emb. off | corr(sine), masked | max abs(head bias), file | max abs(W·Wᵀ − 1.41²·I), file |
|---|---|---|---|---|---|---|---|---|
| MOMENT-1-small | 411e288267 | yes | no: Unexpected key(s) in state_dict: "patch_embedding.position_embedding.pe" | -0.066 | 0.111 | -0.053 | 0.0000 | 0.0000 |
| MOMENT-1-base | 5e44b0ea26 | yes | no: Unexpected key(s) in state_dict: "patch_embedding.position_embedding.pe" | 0.001 | -0.162 | 0.027 | 0.0000 | 0.0000 |
| MOMENT-1-large | ca58581bc7 | yes | no: Unexpected key(s) in state_dict: "patch_embedding.position_embedding.pe" | 0.987 | -0.562 | 0.986 | 0.0121 | 1.3097 |
| small, untrained (no weights) | – | – | – | 0.028 | – | 0.000 | 0.0000 | 0.0000 |
