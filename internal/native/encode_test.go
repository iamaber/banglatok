package native

import (
	"errors"
	"slices"
	"testing"
	"unicode/utf8"

	"github.com/iamaber/banglatok/tokenizer"
)

// encodeTexts encodes texts through EncodeBatch without a layout.
func encodeTexts(r *Registry, handle uint64, texts []string, threads int) ([][]int, error) {
	text, runes := runeCounts(texts)
	batch, err := r.EncodeBatch(handle, text, runes, threads, Layout{})
	if err != nil {
		return nil, err
	}
	out := make([]uint32, batch.Size())
	batch.Write(out, runes)
	rows := make([][]int, len(texts))
	for i, n := range runes {
		rows[i] = make([]int, n)
		for j := range rows[i] {
			rows[i][j] = int(out[j])
		}
		out = out[n:]
	}
	return rows, nil
}

func TestEncodeIntoUndersizedBuffer(t *testing.T) {
	_, model := loadTestModel(t, "unigram")
	r := Registry{models: map[uint64]*tokenizer.Tokenizer{1: model}}
	text := "আমি 🙂 আমি"
	small := make([]uint32, 2)
	if n, err := r.EncodeInto(1, text, small); err != nil || n != len(model.Encode(text)) || n <= len(small) {
		t.Fatalf("undersized buffer: n=%d err=%v", n, err)
	}
}

// Arbitrary text and code-point counts must yield either ErrBatchLengths or
// exactly one ID list per text, matching direct encoding.
func FuzzEncodeBatch(f *testing.F) {
	f.Add("ab", []byte{1, 1})
	f.Add("আমি 🙂", []byte{3, 0, 2})
	f.Add("a\xffb\x00", []byte{4})
	f.Add("", []byte{})
	_, model := loadTestModel(f, "unigram")
	r := &Registry{models: map[uint64]*tokenizer.Tokenizer{1: model}}
	f.Fuzz(func(t *testing.T, text string, counts []byte) {
		runes := make([]uint32, len(counts))
		for i, n := range counts {
			runes[i] = uint32(n)
		}
		if _, err := r.EncodeBatch(2, text, slices.Clone(runes), 1, Layout{}); err == nil {
			t.Fatal("invalid handle must return an error")
		}
		batch, err := r.EncodeBatch(1, text, runes, 1, Layout{})
		if err != nil {
			if !errors.Is(err, ErrBatchLengths) {
				t.Fatal(err)
			}
			return
		}
		texts, _ := splitRunes(text, runes)
		out := make([]uint32, batch.Size())
		batch.Write(out, runes)
		total := 0
		for i, piece := range texts {
			if utf8.ValidString(text) && utf8.RuneCountInString(piece) != int(counts[i]) {
				t.Fatalf("text %d has %d code points, want %d", i, utf8.RuneCountInString(piece), counts[i])
			}
			want := model.Encode(piece)
			if int(runes[i]) != len(want) {
				t.Fatalf("text %d: %d IDs, want %d", i, runes[i], len(want))
			}
			for j, id := range want {
				if out[total+j] != uint32(id) {
					t.Fatalf("text %d ID %d: %d != %d", i, j, out[total+j], id)
				}
			}
			total += len(want)
		}
		if total != len(out) {
			t.Fatal("trailing IDs")
		}
	})
}
