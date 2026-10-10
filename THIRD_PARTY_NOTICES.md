# Third-party notices and scope

Original software, documentation and the hand-authored demo vocabulary are
MIT-licensed; see LICENSE. Native wheels also contain Go runtime/standard-library
code and golang.org/x/text v0.28.0, each under BSD-3-Clause. Their notices are
included in third_party/licenses and in the distribution's license directory.
The distribution's SPDX expression is MIT AND BSD-3-Clause to reflect both.

- Go: https://go.dev/ — third_party/licenses/GO-LICENSE
- golang.org/x/text: https://pkg.go.dev/golang.org/x/text —
  third_party/licenses/X-TEXT-LICENSE and X-TEXT-PATENTS

The Go corpus-preparation CLI additionally depends on modules listed in go.mod
and go.sum. Those modules are fetched separately when building the CLI; the
Python wheel's native bridge imports only the tokenizer and its Unicode dependency.
Dependencies retain their own licenses.

The software license does not grant rights to third-party corpus text, source
snapshots or external pretrained models. data/CORPUS.md and the research reports
record source-specific terms and limitations. Raw corpora and research checkpoints
are excluded from Python distributions. Do not treat their presence in a local
research checkout as permission to redistribute them.
