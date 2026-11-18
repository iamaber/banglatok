package native

import (
	"errors"
	"slices"
	"testing"

	"github.com/iamaber/banglatok/tokenizer"
)

// encodeTexts encodes texts through EncodeBatch without a layout.
func encodeTexts(r *Registry, handle uint64, texts []string, threads int) ([][]int, error) {
	text, lengths := byteLengths(texts)
	batch, err := r.EncodeBatch(handle, text, lengths, threads, Layout{})
	if err != nil {
		return nil, err
	}
	out := make([]uint32, batch.Size())
	batch.Write(out, lengths)
	rows := make([][]int, len(texts))
	for i, n := range lengths {
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

// Arbitrary text and byte lengths must yield either ErrBatchLengths or
// exactly one ID list per text, matching direct encoding.
func FuzzEncodeBatch(f *testing.F) {
	f.Add("ab", []byte{1, 1})
	f.Add("আমি 🙂", []byte{6, 0, 8})
	f.Add("a\xffb\x00", []byte{4})
	f.Add("", []byte{})
	_, model := loadTestModel(f, "unigram")
	r := &Registry{models: map[uint64]*tokenizer.Tokenizer{1: model}}
	f.Fuzz(func(t *testing.T, text string, counts []byte) {
		lengths := make([]uint32, len(counts))
		for i, n := range counts {
			lengths[i] = uint32(n)
		}
		if _, err := r.EncodeBatch(2, text, slices.Clone(lengths), 1, Layout{}); err == nil {
			t.Fatal("invalid handle must return an error")
		}
		batch, err := r.EncodeBatch(1, text, lengths, 1, Layout{})
		if err != nil {
			if !errors.Is(err, ErrBatchLengths) {
				t.Fatal(err)
			}
			return
		}
		texts, _ := splitBytes(text, lengths)
		out := make([]uint32, batch.Size())
		batch.Write(out, lengths)
		total := 0
		for i, piece := range texts {
			if len(piece) != int(counts[i]) {
				t.Fatalf("text %d has %d bytes, want %d", i, len(piece), counts[i])
			}
			want := model.Encode(piece)
			if int(lengths[i]) != len(want) {
				t.Fatalf("text %d: %d IDs, want %d", i, lengths[i], len(want))
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
