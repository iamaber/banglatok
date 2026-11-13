# Bundled model files

The files in `src/bn_tokenizers_embedding/pretrained/` (`bn-bpe-58k.json` and `bn-bpe-28k.json`) are licensed under the Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0): https://creativecommons.org/licenses/by-sa/4.0/legalcode

They contain subword vocabularies, merge rules and frequent-word lists learned from:

- Bengali Wikipedia (dump `20231101.bn`, via the `wikimedia/wikipedia` dataset). © Wikipedia contributors, CC BY-SA 3.0 / GFDL.
- Tatoeba sentences, via the ManyThings.org Anki exports. © Tatoeba contributors, CC BY 2.0.
- BanglaTLit, https://github.com/farhanishmam/BanglaTLit. MIT License.

Changes: the source text was NFC- and whitespace-normalized, filtered and split as described in `data/CORPUS.md`. The models were then learned from the training split with BPE training, space fusion and word-frequency counting. See `docs/MODELS.md`.

Everything else in this repository and in the distributions, including the tokenizer software and the hand-authored demo vocabulary, is covered by `LICENSE` (MIT) and `THIRD_PARTY_NOTICES.md`.
