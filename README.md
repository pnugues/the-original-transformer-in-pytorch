# Programming the Original Transformer with a Modern API

Vaswani et al. published their paper on the original Transformer, [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), in June 2017. Since then, it has become the standard building block of many systems in natural language and image processing.

## A Kitchen-Science Experiment

In 2017, both hardware and machine-learning APIs were very different. Training the original Transformer required 8 NVIDIA P100 GPUs and between 12 hours and 3.5 days for the large models. Not so long ago, PyTorch programs would still crash when compiled or run on Apple's MPS backend. In 2026, the situation has changed completely. Replicating Vaswani et al.'s system is now within reach of anyone with a good laptop. Mine is an Apple M3 Pro — a bit expensive, but prices are expected to drop, and it should soon be possible to run the same experiment on a low-end machine.

This repository documents that claim. I reimplemented the original Transformer, including the training recipe, in order to reproduce the reported BLEU scores on German. The program relies on high-level PyTorch classes that do most of the work: [`Transformer`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html) and [`CrossEntropyLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html). There are no optimization tricks. This keeps the code short and, I hope, easy to understand.

One difference is that I use the [SentencePiece](https://github.com/google/sentencepiece) implementation of BPE for tokenization.

## Running the Programs

Here are the steps to reproduce the experiment:

1. `01_download_wmt14.py` — Downloads the English-German dataset from Hugging Face.
2. `02_tokenize_wmt14.py` — Creates subword tokens with a shared vocabulary using SentencePiece.
3. `03_subtokens_EDA.ipynb` — Visualizes the tokenization results and corpus properties (exploratory data analysis).
4. `04_train_model.ipynb` — Creates and trains the model. The five last checkpoints (saved every 1,400 steps) are kept so that they can later be averaged. A notebook is used to make it easy to change parameters or functions. 
5. `05_average_checkpoints.py` — Averages the last five checkpoints, as done by Vaswani et al.
6. `06_evaluate_translation.py` — Runs inference on the test set with greedy or beam search and reports a rough BLEU score.
7. `07_eval_bleu_moses.sh` — Runs the official BLEU evaluation used in the original paper (slightly different from the score produced by the previous script).

### Temporary / Experimental Files

- `04_train_model_bug.ipynb` — A version that previously worked and reached 27.5 BLEU. It has become prohibitively slow, most likely after the upgrade to macOS 27.

## Evaluation and Results

The training procedure follows the original paper and includes optional checkpoint averaging and beam search. The English-German evaluation used in the paper contained a small post-processing step that improves the score (special handling of dash-separated compounds). I applied the same processing described by Google:

https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/get_ende_bleu.sh#L14

In addition, the official WMT14 test set does not use all 3,003 sentences; a subset is used instead. The correct subset can be obtained with:

```bash
sacrebleu -t wmt14 -l en-de --echo src
```

I included the indices of the 2,737 sentences that belong to this official test set.

I obtained a BLEU score of **27.2**, extremely close to Vaswani et al.'s 27.3. In another run with a slightly different training loop I reached 27.5. Unfortunately I could not re-run that version after the macOS 27 upgrade because it became far too slow (probably related to a change in memory management).

|Model|Test set| Pair|Averaging|Beam|Vaswani|This model|
|-------| -------- | ------- |-------|---|---|---|
|base|newtest2014|EN-DE|yes|yes|27.3|27.2 (27.5)
|base|newtest2013|EN-DE|no|no|25.8|
|base|newtest2014|EN-FR|yes|yes|38.1|
|large|newtest2014|EN-DE|yes|yes|28.4|
|large|newtest2013|EN-DE|no|no|26.4|
|large|newtest2014|EN-FR|yes|yes|41.8|

## Future Work

I plan to publish additional versions of the architecture with increasing levels of detail:

1. The same pedagogical Transformer, but replacing the `Transformer` class with layer implementations that use PyTorch's `scaled_dot_product_attention`.
2. A deeper pedagogical version in which multi-head attention is implemented from scratch.
3. An even deeper version that also implements label smoothing explicitly.
4. Adding the French-English language pair.

## Other Implementations

Several excellent earlier reimplementations exist, although none is fully complete:

- [The Annotated Transformer](https://github.com/harvardnlp/annotated-transformer/) (highly recommended pedagogical version)
- https://github.com/jadore801120/attention-is-all-you-need-pytorch
- https://github.com/gordicaleksa/pytorch-original-transformer
- https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Transformers

Facebook's fairseq is a complete, production-grade implementation (higher complexity). Useful links:

1. https://github.com/facebookresearch/fairseq
2. https://github.com/facebookresearch/fairseq/issues/346 (how to reproduce Vaswani's results)
3. https://github.com/One-sixth/fairseq/ (port to newer Python versions)

## Acknowledgements

I started from a simplified character-based implementation published in my book [*Python for Natural Language Processing*](https://link.springer.com/book/10.1007/978-3-031-57549-5). Alice Totie, Maja Rygård, and Pernilla Åström improved it with a subword implementation and applied it to a [Swedish-English corpus](https://github.com/pernillaastrom/edan70). I later rewrote the training recipe so that it follows Vaswani et al.'s paper as closely as possible.

I used large language models extensively to debug and improve the code (especially the training loop, memory management, and beam search). Grok, DeepSeek, Z.ai, Kimi, and especially Claude were particularly helpful.
