package native

import (
	"encoding/json"
	"slices"
	"testing"

	"github.com/iamaber/banglatok/tokenizer"
)

// Batch encoding operates on handles created and released through the JSON control path.
func TestWireContracts(t *testing.T) {
	for _, algorithm := range []string{"unigram", "bpe"} {
		t.Run(algorithm, func(t *testing.T) {
			path, model := loadTestModel(t, algorithm)
			var registry Registry
			control := func(req request) response {
				t.Helper()
				data, err := json.Marshal(req)
				if err != nil {
					t.Fatal(err)
				}
				var res response
				if err := json.Unmarshal(registry.Call(data), &res); err != nil {
					t.Fatal(err)
				}
				return res
			}
			loaded := control(request{Operation: "load", Path: path})
			if loaded.Error != "" || loaded.Handle == 0 || loaded.Vocabulary != model.VocabSize() || loaded.Algorithm != model.Algorithm() {
				t.Fatal(loaded)
			}
			for _, texts := range [][]string{nil, {"", "ab", "আমি\tভালো 🙂", "a\x00b"}} {
				got, err := encodeTexts(&registry, loaded.Handle, texts, 1)
				if err != nil || len(got) != len(texts) {
					t.Fatalf("batch encode failed: %v", err)
				}
				for i, text := range texts {
					want := model.Encode(text)
					if !slices.Equal(got[i], want) {
						t.Fatalf("batch/direct mismatch for %q: %v != %v", text, got[i], want)
					}
					out := make([]uint32, len(want))
					if n, err := registry.EncodeInto(loaded.Handle, text, out); err != nil || n != len(want) {
						t.Fatalf("encode into failed: n=%d err=%v", n, err)
					}
					for j, id := range want {
						if out[j] != uint32(id) {
							t.Fatalf("encode into/direct mismatch for %q at ID %d: %d != %d", text, j, out[j], id)
						}
					}
					decoded := control(request{Operation: "decode", Handle: loaded.Handle, IDs: got[i]})
					if decoded.Error != "" || decoded.Text != tokenizer.Normalize(text) {
						t.Fatal(decoded)
					}
				}
			}
			pieces := control(request{Operation: "pieces", Handle: loaded.Handle})
			wantPieces, wantFusion := model.Vocabulary()
			if pieces.Error != "" || !slices.Equal(pieces.Pieces, wantPieces) || !slices.Equal(pieces.SpaceFusion, wantFusion) {
				t.Fatal(pieces)
			}
			for range 2 {
				if closed := control(request{Operation: "close", Handle: loaded.Handle}); closed.Error != "" {
					t.Fatal(closed.Error)
				}
			}
			closed := control(request{Operation: "decode", Handle: loaded.Handle})
			if closed.Error != "unknown or closed tokenizer handle" {
				t.Fatal(closed)
			}
			if _, err := encodeTexts(&registry, loaded.Handle, nil, 1); err == nil || err.Error() != closed.Error {
				t.Fatalf("batch close error: %v", err)
			}
			if _, err := registry.EncodeInto(loaded.Handle, "", nil); err == nil || err.Error() != closed.Error {
				t.Fatalf("encode into close error: %v", err)
			}
		})
	}
}
