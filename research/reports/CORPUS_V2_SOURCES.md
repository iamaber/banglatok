# Corpus v2 acquisition sources

## Scope and decisions

The expansion uses real, contributor-authored paragraphs, not transliterations, repeated minority records, sentence fragments cut from long paragraphs, or task/benchmark datasets. Its supply is **candidates**, not accepted, canonically unique training records. Preparation must exclude frozen held-out text/groups, reserve source groups, apply cross-source canonical/near deduplication, and report the final record and character gates separately.

Bengali growth is prioritized by reading **all** articles in the existing Bengali Wikipedia snapshot, rather than the former SHA-256-first-byte modulo-five article sample. Bengali Wikibooks and Wikivoyage provide smaller educational and travel supplements. English Wikipedia supplies additional licensed volume; it does not fix underserved informal registers. **No additional naturally authored Banglish/code-mixed source examined passed both the text-rights and non-benchmark gates. Those buckets do not grow materially in this acquisition.** Existing bucket-weighted training and downstream non-regression gates remain essential; no sentence duplication is used to manufacture balance.

## Acquired-source identities and primary rights evidence

### Wikipedia: Bengali and English

- Publisher dataset: [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia), exact repository revision **`b04c8d1ceb2f5cd4588862100d08de323dccfbaa`**, dump configuration **`20231101.bn` / `20231101.en`**. The [pinned dataset card](https://huggingface.co/datasets/wikimedia/wikipedia/blob/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/README.md) declares CC BY-SA 3.0 / GFDL, documents article `id,url,title,text` fields and a single upstream `train` split, and says the dump was parsed with mwparserfromhell. That upstream split is not our leakage-safe training assignment.
- The pinned card reports **143,069 Bengali articles** and **6,407,814 English articles**. We use both Bengali Parquet shards; English acquisition is bounded to the first eight pinned shards at most, stopping at a per-source record quota. These shard-prefix selections are deterministic, **not a uniform random sample of all English Wikipedia**.
- Primary checksum inventories: [Bengali tree](https://huggingface.co/api/datasets/wikimedia/wikipedia/tree/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/20231101.bn) and [English tree](https://huggingface.co/api/datasets/wikimedia/wikipedia/tree/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/20231101.en?limit=20). Git LFS object IDs are SHA-256 pins; exact bytes and shard paths are in `data/corpus-v2-sources.lock.json`. Bengali pins match the pre-existing raw files; they are read in place without modification.
- Pinned license card: **130,835 bytes**, SHA-256 **`a0f2055ac612f924721e9981fa2a49a18adec1c67cc8026f6c97d12f17211b21`**. This is downloaded and verified before acquisition.
- Current [Wikimedia dump legal guidance](https://dumps.wikimedia.org/legal.html) instead describes CC BY-SA **4.0** / GFDL and warns that exceptions, possible infringements and controlling terms remain. We preserve the specific **3.0 notice in the pinned historical dataset card**, rather than silently relabeling historical data from a moving legal page. [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) permits commercial sharing/adaptation with attribution, license linkage, change indication and share-alike obligations. GFDL is qualified as “where applicable,” because the card explicitly says not all text is GFDL-available.
- Each candidate preserves the upstream canonical article URL, “Wikipedia contributors,” license URLs, snapshot identifier and NFC/whitespace modification notice. Article hyperlinks support author attribution through page history under [Wikimedia terms §7](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content). These notices do not constitute a complete page-by-page legal audit.

### Bengali Wikibooks and Wikivoyage

- Official snapshots: **`20261101`**, [Wikibooks dump manifest](https://dumps.wikimedia.org/bnwikibooks/20261101/dumpstatus.json) and [Wikivoyage dump manifest](https://dumps.wikimedia.org/bnwikivoyage/20261101/dumpstatus.json). Only the finite `pages-articles.xml.bz2` artifacts are selected, not revision histories, user pages or discussions.
- Contributor text is CC BY-SA **4.0**, verified using the projects' primary [Wikibooks rights API](https://bn.wikibooks.org/w/api.php?action=query&meta=siteinfo&siprop=rightsinfo&format=json) and [Wikivoyage rights API](https://bn.wikivoyage.org/w/api.php?action=query&meta=siteinfo&siprop=rightsinfo&format=json), together with [Wikimedia contributor/reuser terms](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content). Each rights API response is **162 bytes**, SHA-256 **`7ae07721a04512fe73ffce0159a7746d239739205ec4d31437fa19f1fa61e57c`**. The acquired API responses are locally retained and checksum-pinned; a later changed response is rejected, not silently repinned.
- Extraction uses streaming BZip2 XML parsing and mwparserfromhell. Only nonredirect namespace-zero content pages are eligible. Templates are not expanded; references, galleries, tables, math/code blocks, headings/list remnants and URL-bearing lines are removed. Actual line-separated paragraphs, never newly sentence-split chunks, pass the same substantial-paragraph filters as Wikipedia. These small project supplements additionally require **at least 80 whitespace-normalized characters and terminal sentence punctuation**, avoiding short navigation/link-label fragments; their metadata records extraction filter version 2.
- Each row includes its canonical page URL, project contributor notice, CC BY-SA 4.0 license URL, exact revision `oldid` URL, contributor-history URL, snapshot and markup/normalization modification notice. Imported passages may carry additional page-specific obligations; canonical/revision links are retained, and no blanket absence-of-infringement guarantee is made.
- Wikibooks group is **`wikibooks-bn:<root title before first slash>`**, so chapters/subpages of the same book cannot be split across training and held-out sets. Wikivoyage group is **`wikivoyage-bn:<page-id>`**. Wikipedia-derived and cross-project passages require preparation-stage deduplication.

## Deterministic extraction and quality controls

`tools/acquire_corpus_v2.py` reads Parquet in **128-article batches** with prebuffering disabled; BZip2 XML parsing releases completed page trees. Downloads stream in at most 1 MiB blocks, have finite five-attempt retries and HTTP timeouts, preserve `.part` files, resume with validated HTTP Range responses, enforce pinned byte limits, and fail on SHA-256 mismatch. Fully present artifacts are verified before reuse. No moving HF `main` URL is used for text acquisition.

For each selected article/page, keep its **first 20 qualifying original paragraphs** by default (`--max-paragraphs` allows 10–20). Normalize NFC and Unicode whitespace; require **12–2,000 Unicode characters and at least eight whitespace words**. Reject replacement characters, control corruption, recognizable email addresses, Bangladeshi phone patterns and simple credential assignments. A majority-of-alphabetic-characters script check rejects grossly mislabelled paragraphs. Source-declared Bengali/English labels are retained; arbitrary Latin text is never labeled Banglish, and incidental Latin names in Bengali Wikipedia do not establish a code-mixed source.

The privacy filters are conservative heuristics, **not complete PII removal**. Public biographies, sensitive historical/political content, toxic quotations and remaining identifiers may exist. The pinned upstream Wikipedia card itself leaves personal/sensitive-information documentation unspecified. Do not interpret public availability or an open copyright license as blanket privacy consent.

Output is atomic UTF-8 JSONL with exactly `text,bucket,source,group,attribution,split`; new rows have empty `split`. A corresponding provenance sidecar records extraction settings, actual records/characters/articles, rejection counts, artifact URL/hash/bytes, candidate SHA-256/bytes and whether a quota truncated a shard. Partial `.jsonl.part` files are not candidates. Existing completed outputs are reused only when settings and hashes match; changing quotas/extraction settings requires a separate `--raw`/`--output` pair so provenance sidecars cannot collide.

**Leakage:** Bengali Wikipedia groups remain exactly **`wiki-bn:<original article-id>`**, including articles already represented by frozen-v1. English uses **`wiki-en:<id>`**. The larger Bengali read deliberately overlaps old training and held-out source groups; acquisition does not claim that overlap is new growth. Preparation must remove the entire reserved group, not merely the formerly selected ten paragraphs. Cross-language translations and source copying are not guaranteed absent. No BanglishRev, existing benchmark source, official BanglaTLit held-out file, or synthetic BanglaDual record is acquired here.

## Other sources investigated, not counted as acquired growth

| Source | Primary evidence and decision |
|---|---|
| Bengali Wikisource, `20261101` | [Manifest](https://dumps.wikimedia.org/bnwikisource/20261101/dumpstatus.json): 362,329,274-byte articles archive; upstream SHA-1 `dd5b6c3fcf5b845e2fb3f41fdfdf54e2be0f3d6f`. [Copyright policy](https://bn.wikisource.org/wiki/Wikisource:Copyright_policy) requires free/public-domain work-level rights tags, not one universal worldwide-public-domain assertion. Not ingested: editions/translations require work-level attribution and jurisdiction review; assembled chapters and ProofreadPage transcriptions must not be double-counted. |
| Global Voices Bengali / OPUS `v2018q4` | [Publisher republication policy](https://globalvoices.org/about/global-voices-attribution-policy/) grants CC BY 3.0 for Global Voices-created text, including commercial adaptation. [Official OPUS inventory](https://opus.nlpl.eu/legacy/GlobalVoices.php) lists 147,385 English–Bengali alignment units and explicitly warns counts include duplicates. Not bulk-ingested: partner republications, embedded social posts and quotations are not blanket-cleared; sentence-aligned exports do not preserve enough per-article author/translator and passage-rights evidence for an automatic clean cutover. |
| Storybooks Bangladesh, commit `66907f9ec1af273526d70f1e54ffb0f1f9a41d73` | [Publisher repository](https://github.com/global-asp/storybooks-bangladesh/tree/66907f9ec1af273526d70f1e54ffb0f1f9a41d73) and [example Bengali story](https://global-asp.github.io/storybooks-bangladesh/stories/bn/0262/) provide story-level CC BY 4.0 and author/translator credits. Forty stories: sound small supplement but not scale supply. Repeated translations across country sites and duplicate HTML language fields must be grouped once. Not counted. |
| StoryWeaver | [Publisher policy](https://storyweaver.org.in/en/open-content) grants CC BY 4.0 with detailed creator/publisher credits. No verified immutable bulk Bengali inventory/revision was established; not acquired or counted. |
| Biblica Open Bengali Contemporary Version (`benobcv`) | [Edition/distribution record](https://ebible.org/find/details.php?id=benobcv) grants CC BY-SA 4.0 for translation text, with copyright and trademark conditions. No immutable inspected artifact revision; religious-domain skew and likely parallel-corpus overlap require separate review. Not counted. |
| IndicCorp | [First-party corpus page](https://indicnlp.ai4bharat.org/corpora/) reports 39.9 million Bengali sentences but licenses CC BY-NC 4.0. **Excluded: NC.** |
| B-CORE, HF revision `fbd637a917efb72d3621f0c9a85727b031ad76fc` | [Depositor card](https://huggingface.co/datasets/nahid-hub/B-CORE-bengali-corpus/blob/fbd637a917efb72d3621f0c9a85727b031ad76fc/README.md) advertises 16.5 million documents / CC BY 4.0 but no inspected source-level rights inventory establishing original text permissions. **Excluded: insufficient provenance.** General web crawls and dataset-wrapper CC0 claims do not confer rights to underlying sites. |
| OpenSLR SLR53 | [Release](https://www.openslr.org/53/) declares CC BY-SA 4.0 for recordings; [author paper](https://storage.googleapis.com/gweb-research2023-media/pubtools/4707.pdf) says prompts came from unspecified “open and available” online sources. **Excluded: underlying prompt licenses and Wikipedia overlap unresolved; recording counts are not unique text counts.** |
| Common Voice | [Mozilla hosting/consent guidance](https://community.mozilladatacollective.com/faq-can-i-get-the-common-voice-or-other-mdc-datasets-from-other-platforms-like-github-or-hugging-face/) describes exclusive hosting/new terms and consent withdrawals. Not inferred cleared from a historical CC0 card; no acquisition. |
| Direct Tatoeba | [Official downloads](https://tatoeba.org/en/downloads) and [terms §6](https://tatoeba.org/en/terms_of_use) license sentence text CC BY 2.0 FR / sentence-specific compatible licenses. Existing corpus already includes Tatoeba; further exports are overlapping provenance, not an independent large Bengali/Banglish source. Not reacquired. |

### Naturally authored Banglish/code-mixed exclusions

| Source / pin | Why excluded |
|---|---|
| BanglishRev | Entire source reserved for evaluation by project contract. No training use, regardless of a license claim. |
| BanglaMix, Zenodo `22086156`; code `117e34871f8a1690fa3d4cdce4fa35b303edca9a` | [Author repository](https://github.com/smahmuddz/BanglaMix) explicitly does not redistribute original/cleaned comments, only IDs/annotations requiring rehydration. Natural YouTube comments are not an author-granted permanently redistributable text corpus. |
| BanTH, HF `f7cec5b84505793ddc7bb31289ae33cad7cedb79` | [Author card](https://huggingface.co/datasets/aplycaebous/BanTH) describes a YouTube-comment task dataset with official splits. MIT metadata does not establish commenter text permissions. Excluded as benchmark/task material and unresolved original rights. |
| BnSentMix, HF `0e6d8a275ce4a54c428dff018577342e7d741a3d` | [Author repository](https://github.com/Nishita2000/BnSentMix) describes Facebook/YouTube/e-commerce sentiment samples. Repository Apache-2.0 versus dataset MIT declarations do not clear original commenter rights. Benchmark/task corpus excluded. |
| Code-mixed Chaos, DOI `10.17632/23dp3t88vk.1` | [Depositor record](https://data.mendeley.com/datasets/23dp3t88vk/1): toxicity task comments, social-platform collection and explicit emotion-dataset reuse. Aggregate CC BY 4.0 does not resolve commenter rights or benchmark overlap. |
| Bengali/Banglish emotion data, DOI `10.17632/4dnrwbxt8n.2` | [Depositor record](https://data.mendeley.com/datasets/4dnrwbxt8n/2) says Bengali was “translated into Banglish” and aggregates existing emotion corpora. Not independent naturally authored Banglish; task overlap. |
| Banglish Hate Speech, DOI `10.17632/58sff5bdxd.1` | [Depositor record](https://data.mendeley.com/datasets/58sff5bdxd/1): labeled task data and insufficient original-text permission evidence. |
| BanglaDual, DOI `10.17632/pxw2bb4wyt.1` | [Depositor record](https://data.mendeley.com/datasets/pxw2bb4wyt/1): 85.34% Wikipedia origin, transliteration/translation pairs and BanglaTLit/BanTH/emotion aggregation. Not independent natural Banglish; upstream and held-out overlap unresolved. |
| Dakshina `v1.0` | [Google repository](https://github.com/google-research-datasets/dakshina) supplies elicited Wikipedia romanizations, including development/test sentences. Not naturally authored informal Banglish; held-out evaluation material excluded. |
| banglakit/transliteration-data, `1cfa9b9888a45386eecab03f5affe5c94cd6117b` | [Repository](https://github.com/banglakit/transliteration-data/tree/1cfa9b9888a45386eecab03f5affe5c94cd6117b): word/name spelling variants rather than authored sentence discourse. Not a corpus-scale natural-text addition. |

These are exclusion decisions, not a claim that every dataset is unlawful. The project requires text-level redistribution evidence, natural authorship and clean evaluation boundaries simultaneously. Existing BanglaTLit retains its original third-party-rights caveat; this acquisition does not relabel that caveat away.

## Reproduction

Research-only dependencies, not tokenizer runtime dependencies:

```sh
uv venv --python 3.12 .venv/corpus-v2
uv pip install --python .venv/corpus-v2/bin/python pyarrow==22.0.0 mwparserfromhell==0.7.2
.venv/corpus-v2/bin/python tools/acquire_corpus_v2.py --source all
.venv/corpus-v2/bin/python tools/acquire_corpus_v2.py --verify
```

Selectors: `--source bn-wikipedia|en-wikipedia|bn-wikibooks|bn-wikivoyage|all`; `--bn-quota`, `--en-quota`, `--books-quota`, `--voyage-quota` (zero means all pinned material), `--en-shards 1..8`, `--max-paragraphs 10..20`, `--batch-size`, `--raw`, `--output`, `--lock`. Defaults: all Bengali material, 3,200,000 English records, first eight English shards maximum, twenty paragraphs/page.

Continuation selectors (default `0`/unset leaves the commands above byte-for-byte unchanged): `--en-offset N` re-reads English shards in the identical pinned order and extraction, **skips the first N qualifying records** that the default run would emit, and only then writes; `--en-quota` counts emitted (new) records only. Skipped rows are never written or counted as records, but each shard's sidecar records the requested `record_offset` setting, `skipped_records`, `skipped_normalized_characters`, `skipped_bytes` and the SHA-256 of the skipped rows exactly as they would have been serialized, so the skipped prefix can be matched to the prior candidate files. A shard lying wholly inside the offset produces provenance only (`"output": null`), no candidate file. The per-article 20-paragraph cap counts skipped paragraphs, so an article split at the offset continues with its next qualifying paragraph under the same group. `--en-offset` requires `--source en-wikipedia` and `--raw`/`--output` outside `data/raw/expanded-v2`. `--reuse-artifacts DIR` hard-links checksum-verified files from a prior artifact directory instead of re-downloading; neither link is written.

Candidates: `data/raw/expanded-v2/candidates/*.jsonl`. Downloaded artifacts: `data/raw/expanded-v2/artifacts/`; metadata: `data/raw/expanded-v2/metadata/*.json`; acquisition summaries: `data/raw/expanded-v2/acquisition-<selector>.json`. Frozen raw/processed data and published model files are not rewritten. Candidate hashes/counts and final preparation acceptance are distinct measurements.

## Measured candidate supply

The finalized seven-file supply contains **4,063,101 candidate records**, **1,231,604,505 normalized characters**, and **3,390,944,492 JSONL bytes**. Bengali accounts for **863,101 records / 275,814,306 characters**; English accounts for **3,200,000 / 955,790,199**. These are pre-preparation counts, including overlap with frozen Wikipedia groups and within/across-source duplicates. They do **not** certify the final accepted training gates.

| Candidate basename under `data/raw/expanded-v2/candidates/` | Records | Normalized characters | Bytes | SHA-256 |
|---|---:|---:|---:|---|
| `bn-wikipedia-train-00000-of-00002.jsonl` | 414,228 | 130,572,350 | 554,142,004 | `ffb94dabebd7490789fb1be98d98cf755a1d0299f033be1cf63fdecba04c588b` |
| `bn-wikipedia-train-00001-of-00002.jsonl` | 375,307 | 116,652,713 | 503,975,917 | `9438ec9fb6838aa0677367fead99357c40e9cabf53ce52f74ea76075f7fbf58a` |
| `en-wikipedia-train-00000-of-00041.jsonl` | 1,481,256 | 454,098,426 | 1,024,141,258 | `168d3970ed0a2ec74b422c172d7a0ebc869452890cf7dce662108ff18eddda46` |
| `en-wikipedia-train-00001-of-00041.jsonl` | 1,367,067 | 400,905,812 | 926,686,138 | `5f3aa243d1b563c50a0c754a6eae6381c67a0c4ece74b63fd0759bb41924f2e6` |
| `en-wikipedia-train-00002-of-00041.jsonl` | 351,677 | 100,785,961 | 235,995,543 | `5b557ed3c96add9e54d4696198e3fc5314fc4853fc14c8fe21db8254bec1facd` |
| `bn-wikibooks-20261101.jsonl` | 38,289 | 16,382,166 | 89,101,658 | `678a752a9ca2f800af5aa9e0bc396e120eefa85fe762da748f13024e97a301c1` |
| `bn-wikivoyage-20261101.jsonl` | 35,277 | 12,207,077 | 56,901,974 | `93f0f5a3ad57ce98f1712ff2a37b7fcedc5d89a19cee77eb5362a1b56476fbeb` |

Only **three** of eight pinned English shards were needed/downloaded. The first two are fully read; the third stops deterministically at the remaining 351,677-record quota after scanning 43,237 articles. Bengali Wikipedia scans all 143,069 articles; Wikibooks scans 4,825 eligible nonredirect main-namespace pages, and Wikivoyage 3,031. Inventory “article” statistics and dump main-namespace page counts need not be equal.

Final source-lock SHA-256: **`37b99fb6577c8a92240977cfa8c454db6405a6b9a6d571be128b83a81e73d531`**. `data/raw/expanded-v2/acquisition-all.json` records this digest and every finalized output. The two small official XML snapshots were bootstrapped against their first-party manifest byte counts/SHA-1 values before their SHA-256 values were frozen for future downloads:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `bnwikibooks-20261101-pages-articles.xml.bz2` | 19,137,711 | `8665bf7f5c2df76d9e90b47309611ca9de6f227a2410e4ff813031a56667605d` |
| `bnwikivoyage-20261101-pages-articles.xml.bz2` | 21,363,536 | `803f150c0a096daed5baf406d4f933cc98f90baa6f9fea0ee9e072bf23ed22bd` |

## Attempt 1 shortfall and English continuation

Preparing the seven-file supply above accepted **2,372,517 unique training records** (558,268,779 normalized characters). The character gate and the leakage audit passed; the record gate (2,817,380) **failed**, about 4.2× the original 563,476 records. That output is kept unchanged at `data/processed/expanded-v2-attempt1/` and is not used for model candidates. No gate was lowered.

To close the gap with genuinely new text, the same pinned English snapshot was read again with `--en-offset 3200000 --en-quota 1800000`, skipping exactly the 3,200,000 passages already acquired:

```sh
.venv/corpus-v2/bin/python tools/acquire_corpus_v2.py --source en-wikipedia \
  --en-offset 3200000 --en-quota 1800000 \
  --raw data/raw/expanded-v2-extension \
  --output data/raw/expanded-v2-extension/candidates \
  --reuse-artifacts data/raw/expanded-v2/artifacts
```

The SHA-256 of each shard's skipped rows equals the original candidate file for that shard (`168d3970…`, `5f3aa243…`, `5b557ed3…`), so the continuation re-derived the identical prefix and emitted none of it. The first three shard downloads were hard-linked from the verified prior artifacts; shard `00003` was downloaded and checked against the lock.

| Candidate basename under `data/raw/expanded-v2-extension/candidates/` | Records | Normalized characters | Bytes | SHA-256 |
|---|---:|---:|---:|---|
| `en-wikipedia-train-00002-of-00041.jsonl` | 959,924 | 275,621,066 | 645,755,666 | `331f3c9ab3d7210d545390d9e70e1e24ca6c510e56e5e4a2d40a1db45024273e` |
| `en-wikipedia-train-00003-of-00041.jsonl` | 840,076 | 244,088,280 | 568,318,689 | `d8c05afe9e3aef436aa0ea6adf39b4a6354f8023f9b327365697efa07ee936d2` |

Total supply is now **5,863,101 candidates** (863,101 Bengali, 5,000,000 English). The extra volume is English only, which makes the corpus more English-heavy. Bucket-weighted training (equal sentence mass within each bucket, then a fixed mixture) keeps English from dominating the learned vocabulary; the per-bucket validation gates check that Bengali and Banglish do not regress.

The final corpus is prepared fresh from all nine candidate files (not appended to attempt 1), so deduplication, near-duplicate grouping and holdout protection apply across every source together.

## Focused proof

The final `.venv/corpus-v2/bin/python tools/acquire_corpus_v2.py --verify` run **passed across all seven complete files**. It streamed every row to check the exact six-string-field schema, permitted source buckets, NFC/whitespace normalization, length/word/script/privacy filters, empty new-material split, source/group prefixes and URL/contributor/license attribution; then independently recomputed record counts, normalized characters, SHA-256 and byte counts against provenance sidecars. Results are retained at **`data/raw/expanded-v2/verification.json`** and match the table above. A separate SHA-256 command confirmed the final source-lock digest.

The final acquisition pass also revalidated every acquired artifact—including the pinned historical Wikipedia card and both primary rights-API notices—against its locked SHA-256/byte count before reuse. Sampled finalized rows preserve Bengali educational/travel prose, whole-book grouping, page revision/history links and English article attribution. No build, lint, formatter, model-training or publication command was run. Final training acceptance, deduplication/leakage audit and bucket non-regression remain the integration owner's measurements.

