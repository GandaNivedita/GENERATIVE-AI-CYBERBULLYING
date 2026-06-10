# Cyberbullying Detection — Experimental Results
Training-free Gen-AI prompting + rule-based hybrid scoring on Twitter data.
Model: Groq llama-3.1-8b-instant | Eval sample: balanced, 6 classes.

## 1. Master comparison (6-class)
| name | accuracy | macro_f1 | weighted_f1 | binary_recall |
| --- | --- | --- | --- | --- |
| Exp7 Hybrid (Reasoning+Rules) | 0.4433 | 0.4036 | 0.4036 | 0.878 |
| Exp3 Few-shot | 0.365 | 0.3782 | 0.3782 | 0.838 |
| Exp1 Rule-based (no LLM) | 0.4167 | 0.3712 | 0.3712 | 0.396 |
| Exp5 Persona | 0.4 | 0.3643 | 0.3643 | 0.878 |
| Exp2 Zero-shot | 0.3833 | 0.3626 | 0.3626 | 0.838 |
| Exp4 Chain-of-Thought | 0.3533 | 0.3334 | 0.3334 | 0.868 |
| Exp6 Few-shot + CoT | 0.3217 | 0.3265 | 0.3265 | 0.872 |

## 2. Prompt-style ablation (macro-F1 vs zero-shot)
| name | macro_f1 | delta_vs_zero_shot |
| --- | --- | --- |
| Exp2 Zero-shot | 0.3626 | 0.0 |
| Exp3 Few-shot | 0.3782 | 0.0156 |
| Exp4 Chain-of-Thought | 0.3334 | -0.0292 |
| Exp5 Persona | 0.3643 | 0.0017 |
| Exp6 Few-shot + CoT | 0.3265 | -0.0361 |

## 3. Hybrid (Reasoning+Rules) vs best pure LLM
- Best pure LLM: **Exp3 Few-shot** — macro-F1 0.3782, binary recall 0.838
- Hybrid: macro-F1 0.4036, binary recall 0.878
- Macro-F1 delta: 0.0254; recall delta: 0.04

## 4. Robustness on hard cases (Exp8: sarcasm/coded/quoted)
| style | n_hard | acc_hard | acc_easy | acc_overall |
| --- | --- | --- | --- | --- |
| zero_shot | 123 | 0.4309 | 0.3711 | 0.3833 |
| few_shot | 123 | 0.3089 | 0.3795 | 0.365 |
| cot | 123 | 0.3577 | 0.3522 | 0.3533 |
| persona | 123 | 0.4472 | 0.3878 | 0.4 |
| few_shot_cot | 123 | 0.252 | 0.3396 | 0.3217 |

## 5. Explanation quality (Exp9, LLM-as-judge)
- Mean composite explanation quality: **0.569**
  - names_target: 0.308
  - cites_reason: 0.550
  - intent_coherent: 0.950
  - consistent_with_gold: 0.467
