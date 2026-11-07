# Public tokenizer-library preview

The user selected an MIT-licensed tokenizer-library preview and delegated remaining implementation choices. The public product is the existing Python Unigram/BPE tokenizer with its Go runtime. Encoder adaptation stays research; stars and downstream quality improvements are not promised.

Approaches considered: source-only packaging would leave installation dependent on a compiler; publishing a trained embedding model would contradict the measured results. Prepare the existing tokenizer library with prebuilt Linux/macOS wheels and a source distribution, retaining the current API and package name.

Deliverables: MIT software license with native dependency notices; honest alpha metadata; a tiny hand-authored, clearly labeled bundled demo model; installed-wheel tests; Go race/vet and Python CI; source-archive and multi-platform wheel builds; an opt-in Trusted Publishing workflow; concise README, contribution/security/release guidance, issue templates and changelog.

Constraints: Python >=3.10 and Go >=1.24 for source builds; zero third-party Python runtime dependencies; no automatic model downloads; no corpus or research checkpoint in distributions; no fabricated CI badges/results, PyPI availability, platform guarantees, or embedding gains. Windows remains unverified in this preview. Local macOS tests cannot stand in for Linux/Intel CI.

Release preparation is authorized. Repository publication, pushing, and PyPI uploads require concrete verified artifacts and the applicable destination/account access; no publication is performed by a default CI run.
