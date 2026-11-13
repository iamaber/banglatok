package native

import (
	"encoding/binary"
	"slices"
	"testing"

	"github.com/iamaber/banglatok/tokenizer"
)

func frame(texts []string) []byte {
	var input []byte
	for _, text := range texts {
		input = binary.LittleEndian.AppendUint32(input, uint32(len(text)))
		input = append(input, text...)
	}
	return input
}

// parseBinary decodes an EncodeBinary response; ok is false for error responses.
func parseBinary(t *testing.T, output []byte, texts int) (batch [][]int, message string, ok bool) {
	t.Helper()
	if len(output) < 4 {
		t.Fatalf("short response %v", output)
	}
	if status := binary.NativeEndian.Uint32(output); status != 0 {
		return nil, string(output[4:]), false
	}
	if len(output)%4 != 0 {
		t.Fatalf("unaligned response length %d", len(output))
	}
	words := make([]int, len(output)/4)
	for i := range words {
		words[i] = int(binary.NativeEndian.Uint32(output[4*i:]))
	}
	counts, ids := words[1:1+texts], words[1+texts:]
	for _, n := range counts {
		batch = append(batch, append([]int{}, ids[:n]...))
		ids = ids[n:]
	}
	if len(ids) != 0 {
		t.Fatal("trailing IDs")
	}
	return batch, "", true
}

func TestBinaryEncode(t *testing.T) {
	model, err := tokenizer.New(tokenizer.Model{Version: 1, Algorithm: "unigram", Normalization: tokenizer.Normalization, Pieces: []tokenizer.Piece{{Text: "আমি", Score: -1}}})
	if err != nil {
		t.Fatal(err)
	}
	r := Registry{models: map[uint64]*tokenizer.Tokenizer{1: model}}
	texts := []string{"আমি", "", "a\x00b 🙂", "\t বাংলা\n"}
	batch, message, ok := parseBinary(t, r.EncodeBinary(1, frame(texts)), len(texts))
	if !ok {
		t.Fatalf("encode: %s", message)
	}
	for i, text := range texts {
		if !slices.Equal(batch[i], model.Encode(text)) {
			t.Fatalf("different IDs for %q", text)
		}
	}
	if batch, _, ok := parseBinary(t, r.EncodeBinary(1, nil), 0); !ok || len(batch) != 0 {
		t.Fatal("empty batch")
	}
	for _, input := range [][]byte{{0}, {255, 255, 255, 255}, {2, 0, 0, 0, 'a'}} {
		if _, _, ok := parseBinary(t, r.EncodeBinary(1, input), 0); ok {
			t.Fatal("accepted malformed input")
		}
	}
	if _, _, ok := parseBinary(t, r.EncodeBinary(99, nil), 0); ok {
		t.Fatal("accepted closed handle")
	}
}

func TestEncodeInto(t *testing.T) {
	model, err := tokenizer.New(tokenizer.Model{Version: 1, Algorithm: "unigram", Normalization: tokenizer.Normalization, Pieces: []tokenizer.Piece{{Text: "আমি", Score: -1}}})
	if err != nil {
		t.Fatal(err)
	}
	r := Registry{models: map[uint64]*tokenizer.Tokenizer{1: model}}
	text := "আমি 🙂 আমি"
	want := model.Encode(text)
	small := make([]uint32, 2)
	if n, err := r.EncodeInto(1, text, small); err != nil || n != len(want) || n <= len(small) {
		t.Fatalf("undersized buffer: n=%d err=%v", n, err)
	}
	out := make([]uint32, len(want))
	n, err := r.EncodeInto(1, text, out)
	if err != nil || n != len(want) {
		t.Fatalf("n=%d err=%v", n, err)
	}
	for i, id := range want {
		if out[i] != uint32(id) {
			t.Fatalf("ID %d: %d != %d", i, out[i], id)
		}
	}
	if _, err := r.EncodeInto(99, text, out); err == nil {
		t.Fatal("accepted closed handle")
	}
}

// Arbitrary framing must yield either an error or exactly one ID list per
// well-formed text, matching direct encoding.
func FuzzBinaryEncode(f *testing.F) {
	f.Add([]byte{0, 0, 0, 0})
	f.Add([]byte{1, 0, 0, 0, 'a'})
	f.Add(frame([]string{"আমি 🙂", "", "a\xffb"}))
	model, err := tokenizer.New(tokenizer.Model{Version: 1, Algorithm: "unigram", Normalization: tokenizer.Normalization, Pieces: []tokenizer.Piece{{Text: "আমি", Score: -1}}})
	if err != nil {
		f.Fatal(err)
	}
	r := Registry{models: map[uint64]*tokenizer.Tokenizer{1: model}}
	f.Fuzz(func(t *testing.T, input []byte) {
		if _, _, ok := parseBinary(t, r.EncodeBinary(2, input), 0); ok {
			t.Fatal("invalid handle must return an error")
		}
		var texts []string
		for rest := input; len(rest) >= 4; {
			n := int(binary.LittleEndian.Uint32(rest))
			if n > len(rest)-4 {
				break
			}
			texts = append(texts, string(rest[4:4+n]))
			rest = rest[4+n:]
		}
		batch, _, ok := parseBinary(t, r.EncodeBinary(1, input), len(texts))
		if !ok {
			return
		}
		for i, text := range texts {
			if !slices.Equal(batch[i], model.Encode(text)) {
				t.Fatalf("text %d: %v != %v", i, batch[i], model.Encode(text))
			}
		}
	})
}
