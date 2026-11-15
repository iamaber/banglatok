package native

import (
	"fmt"
	"testing"

	"github.com/iamaber/banglatok/tokenizer"
)

// BenchmarkEncodeBatch sweeps batch sizes across minChunkBytes,
// minWorkerBytes and minParallelBytes so the thresholds can be rechecked:
// below minParallelBytes, threads=all should match threads=1.
func BenchmarkEncodeBatch(b *testing.B) {
	model, err := tokenizer.Load("../../src/banglatok/pretrained/bn-bpe-28k.json")
	if err != nil {
		b.Skip(err)
	}
	const text = "আমি ভালো আছি। ami valo asi meetingটা 🙂"
	for _, total := range []int{1 << 10, 2 << 10, 4 << 10, 8 << 10, 16 << 10, 64 << 10, 1 << 20} {
		texts := make([]string, max(1, total/len(text)))
		for i := range texts {
			texts[i] = text
		}
		for _, threads := range []int{1, Threads(0)} {
			b.Run(fmt.Sprintf("bytes=%d/threads=%d", total, threads), func(b *testing.B) {
				b.SetBytes(int64(len(texts) * len(text)))
				for b.Loop() {
					batch, err := encodeBatch(model, texts, threads, Layout{})
					if err != nil {
						b.Fatal(err)
					}
					batch.Release()
				}
			})
		}
	}
}
