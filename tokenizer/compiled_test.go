package tokenizer

import (
	"bytes"
	"crypto/sha256"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

var compiledTexts = []string{"", "abab aaaa আমি বাংলা", " aaaa\tabab ", "🙂🙂🙂 banglish \xff", "é কা"}

func writeCompiledModel(t *testing.T, m Model) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "model.json")
	if err := m.Save(path); err != nil {
		t.Fatal(err)
	}
	if err := WriteCompiled(path); err != nil {
		t.Fatal(err)
	}
	return path
}

func encodeAll(tok *Tokenizer) [][]int {
	var out [][]int
	for _, text := range compiledTexts {
		out = append(out, tok.Encode(text))
	}
	return out
}

func TestCompiledMatchesJSON(t *testing.T) {
	fused := smallBPE(t)
	fused.Version, fused.SpaceFusion = 2, []int{PieceOffset, ByteOffset + 0xf0}
	fused.CacheWords = []string{"abab", "aaaa", "আমি", "🙂🙂🙂"}
	for _, m := range []Model{smallBPE(t), fused, readModel(t, "../src/banglatok/data/demo-unigram.json")} {
		path := writeCompiledModel(t, m)
		source, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		compiled, err := loadCompiled(CompiledPath(path), source)
		if err != nil {
			t.Fatal(err)
		}
		plain, _, _, err := loadJSON(source)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(compiled.model, plain.model) {
			t.Fatalf("%s: compiled model differs", m.Algorithm)
		}
		if got, want := encodeAll(compiled), encodeAll(plain); !reflect.DeepEqual(got, want) {
			t.Fatalf("%s: %v != %v", m.Algorithm, got, want)
		}
	}
}

// Every damaged, truncated or stale sidecar must be rejected, and Load must
// then fall back to the JSON with identical IDs.
func TestCompiledRejectsDamage(t *testing.T) {
	m := smallBPE(t)
	m.CacheWords = []string{"abab", "aaaa"}
	path := writeCompiledModel(t, m)
	source, _ := os.ReadFile(path)
	good, _ := os.ReadFile(CompiledPath(path))
	plain, _, _, err := loadJSON(source)
	if err != nil {
		t.Fatal(err)
	}
	want := encodeAll(plain)
	check := func(name string, sidecar, json []byte) {
		t.Helper()
		if err := os.WriteFile(CompiledPath(path), sidecar, 0644); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, json, 0644); err != nil {
			t.Fatal(err)
		}
		if _, err := loadCompiled(CompiledPath(path), json); err == nil {
			t.Fatalf("%s: accepted", name)
		}
		tok, err := Load(path)
		if err != nil {
			t.Fatal(err)
		}
		if bytes.Equal(json, source) && !reflect.DeepEqual(encodeAll(tok), want) {
			t.Fatalf("%s: fallback changed IDs", name)
		}
	}
	for i := range good {
		damaged := bytes.Clone(good)
		damaged[i] ^= 1
		check("flip", damaged, source)
	}
	for _, n := range []int{0, 8, compiledHeader - 1, compiledHeader, len(good) - 1} {
		check("truncate", good[:n], source)
	}
	check("trailing", append(bytes.Clone(good), 0), source)
	// A JSON edit, even one that keeps the model, makes the sidecar stale.
	check("stale", good, append(bytes.Clone(source), ' '))
	// A rehashed payload that disagrees with the encoder is also rejected.
	forged := bytes.Clone(good)
	forged[len(forged)-4*len(cacheWords(m))-8] ^= 1
	sum := sha256.Sum256(forged[compiledHeader:])
	copy(forged[compiledHeader-sha256.Size:], sum[:])
	check("forged", forged, source)
}

// The shipped sidecars must be exactly what this code compiles.
func TestBundledCompiledCurrent(t *testing.T) {
	for _, path := range bundledModels {
		source, err := os.ReadFile(path)
		if err != nil {
			t.Skip(err)
		}
		want, err := compile(source)
		if err != nil {
			t.Fatal(err)
		}
		got, err := os.ReadFile(CompiledPath(path))
		if err != nil || !bytes.Equal(got, want) {
			t.Fatalf("%s is stale; run go run ./tools/compile_model %s", CompiledPath(path), path)
		}
		if _, err := loadCompiled(CompiledPath(path), source); err != nil {
			t.Fatal(err)
		}
	}
}

func FuzzCompiledPayload(f *testing.F) {
	m := smallBPE(f)
	m.CacheWords = []string{"abab", "aaaa"}
	path := filepath.Join(f.TempDir(), "model.json")
	if err := m.Save(path); err != nil {
		f.Fatal(err)
	}
	source, _ := os.ReadFile(path)
	good, err := compile(source)
	if err != nil {
		f.Fatal(err)
	}
	f.Add(good[compiledHeader:])
	f.Fuzz(func(t *testing.T, payload []byte) {
		// Rehash so that parsing and validation, not the checksum, are tested.
		b := append(bytes.Clone(good[:compiledHeader]), payload...)
		sum := sha256.Sum256(payload)
		copy(b[compiledHeader-sha256.Size:], sum[:])
		if tok, err := parseCompiled(b, source); err == nil {
			for _, text := range compiledTexts {
				if _, err := tok.Decode(tok.Encode(text)); err != nil {
					t.Fatal(err)
				}
			}
		}
	})
}
