# LoopLM

Project page for **What Makes Recurrence Effective in Looped Language Models?**



[Paper](https://arxiv.org/abs/2609.36636) · [Project Page](https://spherelab.ai/looplm/) · [Model Page](https://huggingface.co/collections/Mihara-bot/looplm) · [Code](looplm_pipeline/)

We study when recurrence helps, where it should be applied, and how it should be conditioned in looped language models. Through controlled experiments across inference budgets, we show that additional loops can improve reasoning beyond the training horizon while degrading knowledge performance. Combining history-state injection with timestep conditioning provides a lightweight way to improve robustness across inference budgets.

## Code

[`looplm_pipeline/`](looplm_pipeline/) contains the pretraining pipeline for the BaseLoop and LoopFormer experiments. See its [README](looplm_pipeline/README.md) for setup and usage.
