# Bundled model files

The JSON and `.bntok` files in `src/banglatok/pretrained/` are licensed under
the Creative Commons Attribution-ShareAlike 4.0 International License
(CC BY-SA 4.0): https://creativecommons.org/licenses/by-sa/4.0/legalcode

All bundled models contain subword vocabularies, merge rules and frequent-word
lists learned from:

- Bengali Wikipedia (dump `20231101.bn`, via the `wikimedia/wikipedia`
  dataset). © Wikipedia contributors, CC BY-SA 3.0 / GFDL.
- Tatoeba sentences, via the ManyThings.org Anki exports. © Tatoeba
  contributors, CC BY 2.0.
- BanglaTLit, https://github.com/farhanishmam/BanglaTLit. MIT License.

`bn-bpe-28k-v2` additionally learned from:

- Global Voices Bengali articles by their credited authors and translators,
  https://bn.globalvoices.org/. Global Voices original prose is CC BY 3.0 under
  its attribution policy:
  https://globalvoices.org/about/global-voices-attribution-policy/. Partner
  republications, quotations, captions and embeds were excluded.
- Storybooks Bangladesh stories whose individual colophons declare CC BY 3.0
  or CC BY 4.0, including the credited writers and translators,
  https://global-asp.github.io/storybooks-bangladesh/stories/bn/. Source
  snapshot: `global-asp/storybooks-bangladesh@66907f9ec1af273526d70f1e54ffb0f1f9a41d73`.

Changes: source text was NFC- and whitespace-normalized, filtered, deduplicated
and split as described in `data/CORPUS.md` and `docs/MODELS.md`. The models were
then learned with BPE training, space fusion and word-frequency counting. No
YouTube text, viewer comments or synthetic transliterations were used in
`bn-bpe-28k-v2`.

Everything else in this repository and in the distributions, including the tokenizer software and the hand-authored demo vocabulary, is covered by `LICENSE` (MIT) and `THIRD_PARTY_NOTICES.md`.
