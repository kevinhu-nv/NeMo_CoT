# CoT Prompting for Speech Translation

This repo contains the code for the following paper:

Ke Hu, Zhehuai Chen, Chao-Han Huck Yang, Piotr Żelasko, Oleksii Hrinchuk, Vitaly Lavrukhin, Jagadeesh Balam, Boris Ginsburg, "Chain-of-Thought Prompting for Speech Translation", in Proc. ICASSP 2025. ([arxiv](https://arxiv.org/abs/2409.11538))

## Overview

Large language models (LLMs) have demonstrated remarkable advancements in language understanding and generation. Building on the success of text-based LLMs, recent research has adapted these models to use speech embeddings for prompting, resulting in Speech-LLM models that exhibit strong performance in automatic speech recognition (ASR) and automatic speech translation (AST). In this work, we propose a novel approach to leverage ASR transcripts as prompts for AST in a Speech-LLM built on an encoder-decoder text LLM. The Speech-LLM model consists of a speech encoder and an encoder-decoder structure Megatron-T5. By first decoding speech to generate ASR transcripts and subsequently using these transcripts along with encoded speech for prompting, we guide the speech translation in a two-step process like chain-of-thought (CoT) prompting. Low-rank adaptation (LoRA) is used for the T5 LLM for model adaptation and shows superior performance to full model fine-tuning. Experimental results show that the proposed CoT prompting significantly improves AST performance, achieving an average increase of 2.4 BLEU points across 6 En->X or X->En AST tasks compared to speech prompting alone. Additionally, compared to a related CoT prediction method that predicts a concatenated sequence of ASR and AST transcripts, our method performs better by an average of 2 BLEU points.

## Installation

The code was developed based on a previous NVIDIA NeMo repo [NeMo](https://github.com/NVIDIA/NeMo). Docker container was used to train and decode the model.

## Citation
```BibTex
@inproceedings{hu2025chain,
  title={Chain-of-thought prompting for speech translation},
  author={Hu, Ke and Chen, Zhehuai and Yang, Chao-Han Huck and {\.Z}elasko, Piotr and Hrinchuk, Oleksii and Lavrukhin, Vitaly and Balam, Jagadeesh and Ginsburg, Boris},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
