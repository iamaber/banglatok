package native

import (
	"fmt"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/iamaber/banglatok/tokenizer"
)

// mixedBatch spans several parallel chunks, with empty, tiny and long texts.
func mixedBatch() []string {
	parts := []string{"", "a", "ab", "আমি ভালো আছি। ", "ami valo asi 🙂 ", "a\x00b", "meetingটা ", "\xff"}
	var texts []string
	for i := range 3000 {
		texts = append(texts, strings.Repeat(parts[i%len(parts)], i%37))
	}
	texts = append(texts, strings.Repeat("আমি ab 🙂 ", 20000))
	return texts
}

func bundledModel(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	model, err := tokenizer.Load("../../src/banglatok/pretrained/bn-bpe-28k.json")
	if err != nil {
		t.Skip(err)
	}
	return model
}

func TestParallelMatchesSerial(t *testing.T) {
	_, small := loadTestModel(t, "bpe")
	for name, model := range map[string]*tokenizer.Tokenizer{"test": small, "bundled": bundledModel(t)} {
		t.Run(name, func(t *testing.T) {
			r := Registry{models: map[uint64]*tokenizer.Tokenizer{1: model}}
			texts := mixedBatch()
			want, err := encodeTexts(&r, 1, texts, 1)
			if err != nil {
				t.Fatal(err)
			}
			for i, text := range texts[:50] {
				if !slices.Equal(want[i], model.Encode(text)) {
					t.Fatalf("serial text %d differs from Encode", i)
				}
			}
			for _, threads := range []int{0, 2, 3, 8, 64} {
				got, err := encodeTexts(&r, 1, texts, threads)
				if err != nil {
					t.Fatal(err)
				}
				for i := range want {
					if !slices.Equal(got[i], want[i]) {
						t.Fatalf("threads=%d text %d: parallel IDs differ", threads, i)
					}
				}
			}
		})
	}
}

func TestParallelConcurrentCalls(t *testing.T) {
	_, model := loadTestModel(t, "bpe")
	r := Registry{models: map[uint64]*tokenizer.Tokenizer{1: model}}
	texts := mixedBatch()
	want, err := encodeTexts(&r, 1, texts, 1)
	if err != nil {
		t.Fatal(err)
	}
	var workers sync.WaitGroup
	for worker := range 8 {
		workers.Add(1)
		go func() {
			defer workers.Done()
			got, err := encodeTexts(&r, 1, texts, 1+worker)
			if err != nil || !slices.EqualFunc(got, want, slices.Equal) {
				t.Errorf("worker %d: concurrent parallel result differs", worker)
			}
		}()
	}
	workers.Wait()
}

func TestSplit(t *testing.T) {
	texts := mixedBatch()
	for _, threads := range []int{1, 2, 7, 10, 1000} {
		t.Run(fmt.Sprint(threads), func(t *testing.T) {
			chunks, workers := split(texts, threads)
			next := 0
			for _, c := range chunks {
				if c.first != next || c.last <= c.first {
					t.Fatalf("chunk %d..%d does not follow %d", c.first, c.last, next)
				}
				next = c.last
			}
			if next != len(texts) || (threads == 1) != (len(chunks) == 1) || workers < 1 || workers > min(threads, len(chunks)) {
				t.Fatalf("threads=%d: %d workers, %d chunks end at %d", threads, workers, len(chunks), next)
			}
		})
	}
	if chunks, workers := split([]string{strings.Repeat("a", minParallelBytes-2), "b"}, 8); len(chunks) != 1 || workers != 1 {
		t.Fatal("small batch was split")
	}
	if chunks, _ := split(nil, 8); len(chunks) != 1 || chunks[0].last != 0 {
		t.Fatal("empty batch must be one empty chunk")
	}
}
