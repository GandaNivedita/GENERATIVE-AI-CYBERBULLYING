# GENERATIVE-AI-CYBERBULLYING
# Generative AI for Cyberbullying Detection

## 📖 Overview
This repository contains the research framework, datasets, and experimental configurations for **Generative AI-driven cyberbullying detection**.  
The study benchmarks multiple approaches — including zero-shot, few-shot, persona prompting, chain-of-thought reasoning, and hybrid rule+reasoning systems — to evaluate their effectiveness in detecting harmful online behavior.

---

## 🧩 Research Objectives
- Develop a **hybrid architecture** combining rule-based systems with LLM reasoning.
- Benchmark **accuracy, F1 scores, recall, and parse validity** across experimental setups.
- Provide **transparent, reproducible methodology** for academic and stakeholder review.
- Contribute to **explainable AI** in online harm detection.

---

## ⚙️ Experimental Configurations
Seven experiments were conducted:

1. **Rule-based (no LLM)**  
2. **Zero-shot prompting**  
3. **Few-shot prompting**  
4. **Chain-of-Thought reasoning**  
5. **Persona prompting**  
6. **Few-shot + Chain-of-Thought**  
7. **Hybrid (Reasoning + Rules)**  

Each configuration is documented in **Appendix C** with prompt templates and screenshots.

---

## 📊 Results Summary
| Experiment | Accuracy | Macro-F1 | Weighted-F1 | Binary Recall | Parse-OK |
|------------|----------|----------|-------------|---------------|----------|
| Hybrid (Reasoning+Rules) | 0.443 | 0.404 | 0.404 | 0.878 | ✅ |
| Few-shot | 0.365 | 0.378 | 0.378 | 0.838 | ✅ |
| Rule-based | 0.417 | 0.371 | 0.371 | 0.396 | ❌ |
| Persona | 0.400 | 0.364 | 0.364 | 0.878 | ✅ |
| Zero-shot | 0.383 | 0.363 | 0.363 | 0.838 | ✅ |
| Chain-of-Thought | 0.353 | 0.333 | 0.333 | 0.868 | ✅ |
| Few-shot + CoT | 0.322 | 0.327 | 0.327 | 0.872 | ✅ |

---
├── data/                  # Datasets used in experiments
├── prompts/               # Prompt templates (Exp 1–7)
├── configs/               # Configuration files and screenshots
├── results/               # Evaluation metrics and tables
├── appendix/              # Appendix C materials
└── README.md              # Project overview



---

## 🚀 Getting Started
### Prerequisites
- Python 3.9+
- Required libraries: `transformers`, `scikit-learn`, `pandas`, `matplotlib`
- Access to LLM API (e.g., OpenAI, Azure OpenAI)

### Installation
```bash
git clone https://github.com/YourUsername/cyberbullying-genai.git
cd cyberbullying-genai
pip install -r requirements.txt


python run_experiment.py --config configs/exp7_hybrid.json


📌 Key Contributions
First comparative study of prompt engineering strategies for cyberbullying detection.

Demonstrates that Hybrid (Reasoning+Rules) achieves the best balance of accuracy and recall.

Provides reproducible templates and configurations for academic validation.


@article{ganda2026cyberbullying,
  title={Generative AI for Cyberbullying Detection: Hybrid Reasoning and Rule-based Approaches},
  author={Nivedita Ganda.},
  year={2026},
  journal={Under Review}
}



---

This README is publication-ready** and aligns with academic + GitHub standards.  

Would you like me to also prepare a **shorter “executive summary” version** of the README (1–2 pages) that you can share with stakeholders or interview panels, so they don’t have to read the full technical details?




## 📂 Repository Structure
