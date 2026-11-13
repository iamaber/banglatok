package native

import (
	"encoding/json"
	"path/filepath"
	"sync"
	"testing"

	"github.com/iamaber/banglatok/tokenizer"
)

func loadTestModel(t testing.TB, algorithm string) (string, *tokenizer.Tokenizer) {
	t.Helper()
	m := tokenizer.Model{Version: 1, Algorithm: algorithm, Normalization: tokenizer.Normalization}
	switch algorithm {
	case "unigram":
		m.Pieces = []tokenizer.Piece{{Text: "আমি", Score: -1}}
	case "bpe":
		m.Pieces = []tokenizer.Piece{{Text: "a"}, {Text: "b"}, {Text: "ab"}}
		m.Merges = []tokenizer.Merge{{Left: 0, Right: 1, Result: 2}}
	default:
		t.Fatalf("unknown test algorithm %q", algorithm)
	}
	path := filepath.Join(t.TempDir(), "model.json")
	if err := m.Save(path); err != nil {
		t.Fatal(err)
	}
	model, err := tokenizer.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	return path, model
}

func TestRegistryLifecycle(t *testing.T) {
	path, _ := loadTestModel(t, "unigram")
	var registry Registry
	loaded, err := registry.call(request{Operation: "load", Path: path})
	if err != nil {
		t.Fatal(err)
	}
	var workers sync.WaitGroup
	for range 16 {
		workers.Add(1)
		go func() {
			defer workers.Done()
			for range 20 {
				batch, err := encodeTexts(&registry, loaded.Handle, []string{"আমি hello 🙂"}, 0)
				if err != nil {
					t.Error(err)
					return
				}
				decoded, err := registry.call(request{Operation: "decode", Handle: loaded.Handle, IDs: batch[0]})
				if err != nil || decoded.Text != "আমি hello 🙂" {
					t.Error(decoded, err)
					return
				}
			}
		}()
	}
	workers.Wait()
	if _, err := registry.call(request{Operation: "close", Handle: loaded.Handle}); err != nil {
		t.Fatal(err)
	}
	if len(registry.models) != 0 {
		t.Fatal("model handle retained after close")
	}
	if _, err := registry.call(request{Operation: "decode", Handle: loaded.Handle}); err == nil {
		t.Fatal("accepted closed handle")
	}
}

func TestMalformedRequest(t *testing.T) {
	var r Registry
	for _, s := range []string{"{", `{"op":"decode","ids":[true]}`, `{"op":"decode","handle":99}`} {
		var result response
		if err := json.Unmarshal(r.Call([]byte(s)), &result); err != nil {
			t.Fatal(err)
		}
		if result.Error == "" {
			t.Fatal("missing error")
		}
	}
}
