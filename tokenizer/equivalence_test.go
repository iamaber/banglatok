package tokenizer

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"unicode/utf8"
)

var bundledModels = []string{"../src/banglatok/pretrained/bn-bpe-58k.json", "../src/banglatok/pretrained/bn-bpe-28k.json"}

func readModel(t testing.TB, path string) Model {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Skip(err)
	}
	var m Model
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	return m
}

// referenceBPE is the plain map-based merge loop the fast path must match.
func referenceBPE(m Model, word string) []int {
	ranks := map[[2]int]int{}
	for rank, merge := range m.Merges {
		ranks[[2]int{PieceOffset + merge.Left, PieceOffset + merge.Right}] = rank
	}
	chars := map[string]int{}
	for i, p := range m.Pieces {
		if utf8.RuneCountInString(p.Text) == 1 {
			chars[p.Text] = PieceOffset + i
		}
	}
	var ids []int
	for _, r := range word {
		if id, ok := chars[string(r)]; ok {
			ids = append(ids, id)
		} else {
			for _, b := range []byte(string(r)) {
				ids = append(ids, ByteOffset+int(b))
			}
		}
	}
	for {
		best := math.MaxInt
		for i := 0; i+1 < len(ids); i++ {
			if rank, ok := ranks[[2]int{ids[i], ids[i+1]}]; ok && rank < best {
				best = rank
			}
		}
		if best == math.MaxInt {
			return ids
		}
		merge := m.Merges[best]
		var next []int
		for i := 0; i < len(ids); i++ {
			if i+1 < len(ids) && ids[i] == PieceOffset+merge.Left && ids[i+1] == PieceOffset+merge.Right {
				next = append(next, PieceOffset+merge.Result)
				i++
			} else {
				next = append(next, ids[i])
			}
		}
		ids = next
	}
}

func TestMergeTableMatchesMap(t *testing.T) {
	for _, path := range bundledModels {
		m := readModel(t, path)
		tok, err := New(m)
		if err != nil {
			t.Fatal(err)
		}
		want := map[uint64]uint64{}
		for rank, merge := range m.Merges {
			want[pairKey(PieceOffset+merge.Left, PieceOffset+merge.Right)] = uint64(rank)<<32 | uint64(PieceOffset+merge.Result)
		}
		for key, merge := range want {
			if got := tok.mergeRanks.find(key); got != merge {
				t.Fatalf("%s: pair %x = %x, want %x", path, key, got, merge)
			}
		}
		rng := rand.New(rand.NewSource(1))
		limit := PieceOffset + len(m.Pieces)
		for range 1_000_000 {
			left, right := rng.Intn(limit), rng.Intn(limit)
			merge, ok := want[pairKey(left, right)]
			if !ok {
				merge = noMerge
			}
			if got := tok.merge(left, right); got != merge {
				t.Fatalf("%s: pair %d,%d = %x, want %x", path, left, right, got, merge)
			}
		}
	}
}

func smallBPE(t testing.TB) Model {
	m, err := TrainBPE(map[string]float64{"aaaaaaaa": 9, "abababab": 7, "আমিআমি": 5, "বাংলা": 4, "banglish": 3, "🙂🙂🙂": 2}, 290, 8, nil)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func randomWord(rng *rand.Rand, pieces []Piece) string {
	var b strings.Builder
	for n := rng.Intn(12); n >= 0; n-- {
		switch rng.Intn(10) {
		case 0:
			b.WriteRune(rune(0x80 + rng.Intn(0x2ff80)))
		case 1:
			b.WriteString(strings.Repeat(pieces[rng.Intn(len(pieces))].Text, 1+rng.Intn(40)))
		default:
			b.WriteString(pieces[rng.Intn(len(pieces))].Text)
		}
	}
	return strings.ReplaceAll(b.String(), " ", "")
}

// Words of every length, including runs far beyond the 64-entry stack buffer,
// must segment exactly as the reference loop does.
func TestBPEMatchesReference(t *testing.T) {
	models := []Model{readModel(t, bundledModels[1])}
	models = append(models, smallBPE(t))
	for _, m := range models {
		m.CacheWords, m.SpaceFusion, m.Version = nil, nil, 1
		tok, err := New(m)
		if err != nil {
			t.Fatal(err)
		}
		rng := rand.New(rand.NewSource(2))
		for range 3000 {
			word := randomWord(rng, m.Pieces)
			if got, want := tok.appendBPE(nil, word), referenceBPE(m, word); !reflect.DeepEqual(got, want) && len(got)+len(want) > 0 {
				t.Fatalf("%q: %v, want %v", word, got, want)
			}
		}
	}
}

func FuzzBPEMatchesReference(f *testing.F) {
	m := smallBPE(f)
	tok, err := New(m)
	if err != nil {
		f.Fatal(err)
	}
	for _, s := range []string{"", "aaaaaaaaa", "ababab", "আমিবাংলা", "\xff", "🙂🙂🙂"} {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, word string) {
		if got, want := tok.appendBPE(nil, word), referenceBPE(m, word); !reflect.DeepEqual(got, want) && len(got)+len(want) > 0 {
			t.Fatalf("%q: %v, want %v", word, got, want)
		}
	})
}

// appendWords skips normalizing text made of cached words because a single
// space keeps normalized words normalized; check that against full NFC.
func FuzzNormalizedWordsConcatenate(f *testing.F) {
	for _, pair := range [][2]string{{"a", "b"}, {"ক", "া"}, {"e", "́"}, {"́", "́"}, {"ে", "া"}, {"🙂", "️"}} {
		f.Add(pair[0], pair[1])
	}
	f.Fuzz(func(t *testing.T, a, b string) {
		if a == "" || b == "" || Normalize(a) != a || Normalize(b) != b {
			return
		}
		if s := a + " " + b; Normalize(s) != s {
			t.Fatalf("%+q", s)
		}
	})
}

// Encode verifies normalization while encoding; it must match normalizing first.
func FuzzEncodeMatchesNormalizeFirst(f *testing.F) {
	m := smallBPE(f)
	m.Version, m.SpaceFusion = 2, []int{PieceOffset, PieceOffset + 1, ByteOffset + 0xe0}
	m.CacheWords = []string{"aaaa", "abab", "আমি", "́", "া", "🙂🙂"}
	path := filepath.Join(f.TempDir(), "model.json")
	if err := m.Save(path); err != nil {
		f.Fatal(err)
	}
	tok, err := Load(path)
	if err != nil {
		f.Fatal(err)
	}
	for _, s := range []string{"", " ", "aaaa abab", "aaaa  abab", "aaaa ", " aaaa", "á ́", "ক া", "কা", "আমি\tআমি", "\xff aaaa", "é aaaa"} {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, text string) {
		want, _ := tok.appendWords(nil, Normalize(text), false)
		if got := tok.Encode(text); !reflect.DeepEqual(got, want) && len(got)+len(want) > 0 {
			t.Fatalf("%+q: %v, want %v", text, got, want)
		}
	})
}

func TestBundledCacheWordsConcatenate(t *testing.T) {
	for _, path := range bundledModels {
		m := readModel(t, path)
		words := append([]string{"́", "া", "্", "‍", "a", "ক"}, m.CacheWords...)
		for _, p := range m.Pieces {
			if !strings.Contains(p.Text, " ") && Normalize(p.Text) == p.Text {
				words = append(words, p.Text)
			}
		}
		for i, w := range words {
			for _, other := range []string{words[(i*7919)%len(words)], "́", "া", "ক"} {
				if s := w + " " + other; Normalize(s) != s {
					t.Fatalf("%+q", s)
				}
				if s := other + " " + w; Normalize(s) != s {
					t.Fatalf("%+q", s)
				}
			}
		}
	}
}
