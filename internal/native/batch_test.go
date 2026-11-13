package native

import (
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"testing"
	"unicode/utf8"

	"github.com/iamaber/banglatok/tokenizer"
)

func runeCounts(texts []string) (string, []uint32) {
	runes := make([]uint32, len(texts))
	for i, text := range texts {
		runes[i] = uint32(utf8.RuneCountInString(text))
	}
	return strings.Join(texts, ""), runes
}

// shape mirrors the Python list implementation of Tokenizer.__call__.
func shape(batch [][]int, layout Layout) (rows [][]int, width int) {
	reserve := 0
	if layout.Special {
		reserve = 2
	}
	for _, ids := range batch {
		if layout.MaxLength > 0 && len(ids) > layout.MaxLength-reserve {
			ids = ids[:layout.MaxLength-reserve]
		}
		if layout.Special {
			ids = append(append([]int{bosID}, ids...), eosID)
		}
		rows = append(rows, ids)
		width = max(width, len(ids))
	}
	if layout.Width > 0 {
		width = layout.Width
	}
	return rows, width
}

func encodeLayout(t *testing.T, r *Registry, texts []string, threads int, layout Layout) ([]uint32, []uint32) {
	t.Helper()
	text, runes := runeCounts(texts)
	batch, err := r.EncodeBatch(1, text, runes, threads, layout)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]uint32, batch.Size())
	batch.Write(out, runes)
	return out, runes
}

func TestEncodeBatchLayouts(t *testing.T) {
	_, small := loadTestModel(t, "bpe")
	layouts := []Layout{
		{},
		{Special: true},
		{MaxLength: 1},
		{MaxLength: 2, Special: true},
		{MaxLength: 7, Special: true, Width: -1},
		{MaxLength: 7, Width: 9},
		{MaxLength: 300, Special: true, Width: 300},
		{Width: -1},
		{Special: true, Width: -1},
	}
	for name, model := range map[string]*tokenizer.Tokenizer{"test": small, "bundled": bundledModel(t)} {
		r := &Registry{models: map[uint64]*tokenizer.Tokenizer{1: model}}
		for _, texts := range [][]string{nil, {""}, {"", "", ""}, {"ab"}, mixedBatch()[:200], mixedBatch()[:1000]} {
			var encoded [][]int
			for _, text := range texts {
				encoded = append(encoded, model.Encode(text))
			}
			for _, layout := range layouts {
				want, width := shape(encoded, layout)
				for _, threads := range []int{1, 4} {
					t.Run(fmt.Sprintf("%s/%d/%+v/%d", name, len(texts), layout, threads), func(t *testing.T) {
						out, rows := encodeLayout(t, r, texts, threads, layout)
						offset := 0
						for i, ids := range want {
							if int(rows[i]) != len(ids) {
								t.Fatalf("row %d length %d, want %d", i, rows[i], len(ids))
							}
							n := len(ids)
							if layout.Width != 0 {
								n = width
							}
							got := out[offset : offset+n]
							for j := range got {
								id := 0
								if j < len(ids) {
									id = ids[j]
								}
								if got[j] != uint32(id) {
									t.Fatalf("row %d ID %d: %d != %d", i, j, got[j], id)
								}
							}
							offset += n
						}
						if offset != len(out) {
							t.Fatalf("wrote %d IDs, rows use %d", len(out), offset)
						}
					})
				}
			}
		}
	}
}

func TestEncodeBatchErrors(t *testing.T) {
	_, model := loadTestModel(t, "bpe")
	r := &Registry{models: map[uint64]*tokenizer.Tokenizer{1: model}}
	for _, c := range []struct {
		text  string
		runes []uint32
	}{
		{"ab", []uint32{1}},       // text left over
		{"ab", []uint32{3}},       // too many code points
		{"ab", []uint32{1, 0, 2}}, // later text past the end
		{"", []uint32{1}},
		{"আ", []uint32{2}},
		{"a", nil},
	} {
		if _, err := r.EncodeBatch(1, c.text, c.runes, 1, Layout{}); !errors.Is(err, ErrBatchLengths) {
			t.Fatalf("%q %v: %v", c.text, c.runes, err)
		}
	}
	for _, layout := range []Layout{
		{MaxLength: -1}, {MaxLength: 1, Special: true}, {Width: -2},
		{Width: 3}, {Width: 3, MaxLength: 4},
	} {
		if _, err := r.EncodeBatch(1, "ab", []uint32{2}, 1, layout); !errors.Is(err, ErrLayout) {
			t.Fatalf("%+v: %v", layout, err)
		}
	}
	if _, err := r.EncodeBatch(1, "ab", []uint32{1, 1}, 1, Layout{MaxLength: 1 << 30, Width: 1 << 30}); !errors.Is(err, ErrBatchSize) {
		t.Fatalf("oversized padding: %v", err)
	}
	if _, err := r.EncodeBatch(2, "", nil, 1, Layout{}); err == nil || err.Error() != "unknown or closed tokenizer handle" {
		t.Fatalf("closed handle: %v", err)
	}
	// An empty batch has nothing to write.
	batch, err := r.EncodeBatch(1, "", nil, 1, Layout{})
	if err != nil || batch.Size() != 0 {
		t.Fatal(batch, err)
	}
	batch.Write(nil, nil)
}

func TestSplitRunes(t *testing.T) {
	texts := []string{"", "a", "আমি", "🙂x", "", "a\x00b"}
	text, runes := runeCounts(texts)
	got, ok := splitRunes(text, runes)
	if !ok || !slices.Equal(got, texts) {
		t.Fatalf("%q %v", got, ok)
	}
}

func TestEncodeBatchConcurrent(t *testing.T) {
	model := bundledModel(t)
	r := &Registry{models: map[uint64]*tokenizer.Tokenizer{1: model}}
	texts := mixedBatch()
	layout := Layout{MaxLength: 64, Special: true, Width: -1}
	want, _ := encodeLayout(t, r, texts, 1, layout)
	var workers sync.WaitGroup
	for worker := range 8 {
		workers.Add(1)
		go func() {
			defer workers.Done()
			text, runes := runeCounts(texts)
			batch, err := r.EncodeBatch(1, text, runes, worker+1, layout)
			if err != nil {
				t.Error(err)
				return
			}
			out := make([]uint32, batch.Size())
			batch.Write(out, runes)
			if !slices.Equal(out, want) {
				t.Errorf("worker %d: result differs", worker)
			}
		}()
	}
	workers.Wait()
}
