package native

import (
	"encoding/json"
	"path/filepath"
	"slices"
	"testing"

	"github.com/iamaber/banglatok/tokenizer"
)

// Both wire formats operate on handles created and released through the C control path.
func TestWireContracts(t *testing.T) {
	models := []tokenizer.Model{
		{Version: 1, Algorithm: "unigram", Normalization: tokenizer.Normalization, Pieces: []tokenizer.Piece{{Text: "আমি", Score: -1}}},
		{Version: 1, Algorithm: "bpe", Normalization: tokenizer.Normalization, Pieces: []tokenizer.Piece{{Text: "a"}, {Text: "b"}, {Text: "ab"}}, Merges: []tokenizer.Merge{{Left: 0, Right: 1, Result: 2}}},
	}
	for _, model := range models {
		t.Run(model.Algorithm, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "model.json")
			if err := model.Save(path); err != nil {
				t.Fatal(err)
			}
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
			if loaded.Error != "" || loaded.Handle == 0 || loaded.Vocabulary != tokenizer.PieceOffset+len(model.Pieces) {
				t.Fatal(loaded)
			}
			for _, texts := range [][]string{nil, {"", "ab", "আমি\tভালো 🙂", "a\x00b"}} {
				input := frame(texts)
				batch := control(request{Operation: "encode_batch", Handle: loaded.Handle, Texts: texts})
				if batch.Error != "" {
					t.Fatal(batch.Error)
				}
				var want [][]int
				for i, ids := range batch.Batch {
					single := control(request{Operation: "encode", Handle: loaded.Handle, Text: texts[i]})
					if single.Error != "" || !slices.Equal(single.IDs, ids) {
						t.Fatalf("single/batch mismatch: %+v, %v", single, ids)
					}
					decoded := control(request{Operation: "decode", Handle: loaded.Handle, IDs: ids})
					if decoded.Error != "" || decoded.Text != tokenizer.Normalize(texts[i]) {
						t.Fatal(decoded)
					}
					want = append(want, append([]int{}, ids...))
				}
				got, message, ok := parseBinary(t, registry.EncodeBinary(loaded.Handle, input), len(texts))
				if !ok || len(got) != len(want) {
					t.Fatalf("binary encode failed: %s", message)
				}
				for i := range want {
					if !slices.Equal(got[i], want[i]) {
						t.Fatalf("binary/JSON mismatch: %v != %v", got[i], want[i])
					}
				}
				pieces := control(request{Operation: "pieces", Handle: loaded.Handle})
				if pieces.Error != "" || len(pieces.Pieces) != len(model.Pieces) || pieces.Pieces[0] != model.Pieces[0].Text {
					t.Fatal(pieces)
				}
			}
			for range 2 {
				if closed := control(request{Operation: "close", Handle: loaded.Handle}); closed.Error != "" {
					t.Fatal(closed.Error)
				}
			}
			closed := control(request{Operation: "encode_batch", Handle: loaded.Handle})
			if closed.Error != "unknown or closed tokenizer handle" {
				t.Fatal(closed)
			}
			if _, message, ok := parseBinary(t, registry.EncodeBinary(loaded.Handle, nil), 0); ok || message != closed.Error {
				t.Fatalf("binary close error: %q", message)
			}
		})
	}
}
