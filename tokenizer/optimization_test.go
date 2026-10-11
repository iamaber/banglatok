package tokenizer

import (
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"golang.org/x/text/unicode/norm"
)

func TestLoadedPieceCachePreservesSegmentation(t *testing.T) {
	models := []Model{
		{Version: 1, Algorithm: "unigram", Normalization: Normalization, Pieces: []Piece{{"a", -1}, {"b", -1}, {"ab", -10}}},
		{Version: 1, Algorithm: "bpe", Normalization: Normalization, Pieces: []Piece{{"a", 0}, {"b", 0}, {"c", 0}, {"bc", 0}, {"ab", 0}, {"abc", 0}}, Merges: []Merge{{1, 2, 3}, {0, 1, 4}, {4, 2, 5}}},
	}
	for _, m := range models {
		path := filepath.Join(t.TempDir(), "model.json")
		if err := m.Save(path); err != nil {
			t.Fatal(err)
		}
		cached, err := Load(path)
		if err != nil {
			t.Fatal(err)
		}
		plain, err := New(m)
		if err != nil {
			t.Fatal(err)
		}
		for _, text := range []string{"ab abc a", strings.Repeat("abc", 200) + " ab abc", "a 🙂 ab\x00abc", ""} {
			want := plain.Encode(text)
			got := cached.Encode(text)
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("%s %q: %v != %v", m.Algorithm, text, got, want)
			}
			if len(got) > 0 {
				got[0] = -1
			}
			if !reflect.DeepEqual(cached.Encode(text), want) {
				t.Fatal("returned IDs alias cache")
			}
		}
	}
}

func FuzzNormalizeEquivalent(f *testing.F) {
	for _, text := range []string{"", "  a b  ", "a\u00a0b", "বাংলা", "a\x00b", "e\u0301", "a\xffb", "a\u2028b", "ক\u09c7\u09be", "ড\u09bc\u09be", "ক\u09cd\u09bc", "\u09df 🙂\ufe0f"} {
		f.Add(text)
	}
	f.Fuzz(func(t *testing.T, text string) {
		want := strings.Join(strings.Fields(norm.NFC.String(strings.ToValidUTF8(text, "\uFFFD"))), " ")
		if got := Normalize(text); got != want {
			t.Fatalf("%q != %q", got, want)
		}
	})
}

// The fast path must agree with full NFC for every listed pair and for every
// Bengali triple, where vowel-sign composition and mark reordering live.
func TestAlreadyNormalizedMatchesNFC(t *testing.T) {
	var listed, bengali []rune
	for _, span := range fastRanges {
		for r := span[0]; r <= span[1]; r++ {
			listed = append(listed, r)
			if r >= 0x0980 && r <= 0x09ff {
				bengali = append(bengali, r)
			}
		}
	}
	check := func(s string) {
		if alreadyNormalized(s) {
			if want := strings.Join(strings.Fields(norm.NFC.String(s)), " "); want != s {
				t.Fatalf("%+q claimed normalized; NFC gives %+q", s, want)
			}
		}
	}
	for _, a := range listed {
		for _, b := range listed {
			check(string([]rune{a, b}))
		}
	}
	for _, a := range bengali {
		for _, b := range bengali {
			for _, c := range bengali {
				check(string([]rune{a, b, c}))
			}
		}
	}
	for _, s := range []string{"আমি বাংলা বলি।", "ami bangla boli", "meetingটা কখন?", "পড়াশোনা ভালো ❤️ 🙂", "কৌশল"} {
		if !alreadyNormalized(s) {
			t.Fatalf("common text %q missed the fast path", s)
		}
	}
}
