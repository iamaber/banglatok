package tokenizer

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
	"unicode/utf8"
)

// IDs 0..3 are reserved; 4..259 are literal UTF-8 bytes, never text aliases.
const ByteOffset = 4
const PieceOffset = 260

type Piece struct {
	Text  string  `json:"text"`
	Score float64 `json:"score"`
}

type Model struct {
	Version       int      `json:"version"`
	Algorithm     string   `json:"algorithm"`
	Normalization string   `json:"normalization"`
	Pieces        []Piece  `json:"pieces"`
	Merges        []Merge  `json:"merges,omitempty"`
	CacheWords    []string `json:"cache_words,omitempty"`
	SpaceFusion   []int    `json:"space_fusion,omitempty"`
}

type node struct {
	next map[byte]int
	id   int
}

// Tokenizer is immutable after construction and safe for concurrent Encode calls.
type Tokenizer struct {
	model           Model
	trie            []node
	mergeRanks      mergeTable
	characters      map[rune]int
	denseCharacters []int32
	cache           *wordCache
	spaceFusion     []int
}

func New(m Model) (*Tokenizer, error) {
	if uint64(len(m.Pieces))+uint64(len(m.SpaceFusion)) > math.MaxUint32-PieceOffset {
		return nil, fmt.Errorf("vocabulary exceeds 32-bit token IDs")
	}
	if (m.Version != 1 && m.Version != 2) || (m.Algorithm != "unigram" && m.Algorithm != "bpe") || m.Normalization != Normalization {
		return nil, fmt.Errorf("unsupported model contract")
	}
	if len(m.SpaceFusion) > 0 && (m.Version != 2 || m.Algorithm != "bpe") {
		return nil, fmt.Errorf("space fusion requires version 2 BPE")
	}
	for _, word := range m.CacheWords {
		if word == "" || len(word) > 1024 || strings.Contains(word, " ") || word != Normalize(word) {
			return nil, fmt.Errorf("invalid cache word")
		}
	}
	t := &Tokenizer{model: m, trie: []node{{id: -1}}}
	t.model.Pieces = append([]Piece(nil), m.Pieces...)
	t.model.CacheWords = append([]string(nil), m.CacheWords...)
	t.model.SpaceFusion = append([]int(nil), m.SpaceFusion...)
	if len(m.SpaceFusion) > 0 {
		base := PieceOffset + len(m.Pieces)
		t.spaceFusion = make([]int, base)
		for i, id := range m.SpaceFusion {
			if id < ByteOffset || id >= base || id == ByteOffset+32 || t.spaceFusion[id] != 0 {
				return nil, fmt.Errorf("invalid or duplicate space fusion ID %d", id)
			}
			t.spaceFusion[id] = base + i
		}
	}
	// Only Unigram walks the trie; BPE needs just the duplicate check.
	var seen map[string]struct{}
	if m.Algorithm == "bpe" {
		seen = make(map[string]struct{}, len(m.Pieces))
	}
	for i, p := range m.Pieces {
		if p.Text == "" || !utf8.ValidString(p.Text) || math.IsNaN(p.Score) || math.IsInf(p.Score, 0) || p.Score > 0 {
			return nil, fmt.Errorf("invalid piece %d", i)
		}
		if seen != nil {
			if _, ok := seen[p.Text]; ok {
				return nil, fmt.Errorf("duplicate piece %q", p.Text)
			}
			seen[p.Text] = struct{}{}
			continue
		}
		n := 0
		for j := range len(p.Text) {
			if t.trie[n].next == nil {
				t.trie[n].next = make(map[byte]int)
			}
			v, ok := t.trie[n].next[p.Text[j]]
			if !ok {
				v = len(t.trie)
				t.trie[n].next[p.Text[j]] = v
				t.trie = append(t.trie, node{id: -1})
			}
			n = v
		}
		if t.trie[n].id >= 0 {
			return nil, fmt.Errorf("duplicate piece %q", p.Text)
		}
		t.trie[n].id = i
	}
	if m.Algorithm == "bpe" {
		if err := t.initBPE(); err != nil {
			return nil, err
		}
	}
	return t, nil
}

// Load reads a JSON model. A compiled sidecar (see CompiledPath) replaces
// parsing and cache segmentation when it matches the JSON bytes exactly.
func Load(path string) (*Tokenizer, error) {
	source, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if t, err := loadCompiled(CompiledPath(path), source); err == nil {
		return t, nil
	}
	t, _, _, err := loadJSON(source)
	return t, err
}

// loadJSON also returns the cache IDs and per-word ends for compilation.
func loadJSON(source []byte) (*Tokenizer, []uint32, []uint32, error) {
	var m Model
	if err := json.Unmarshal(source, &m); err != nil {
		return nil, nil, nil, err
	}
	t, err := New(m)
	if err != nil {
		return nil, nil, nil, err
	}
	words := cacheWords(m)
	ids, ends := make([]uint32, 0, 2*len(words)), make([]uint32, len(words))
	var buffer []int
	for i, word := range words {
		buffer = t.AppendEncode(buffer[:0], word)
		for _, id := range buffer {
			ids = append(ids, uint32(id))
		}
		ends[i] = uint32(len(ids))
	}
	t.cache = newWordCache(words, ids, ends, t.spaceFusion)
	return t, ids, ends, nil
}

// cacheWords lists vocabulary strings and optional training words, never
// caller text. Load computes their actual segmentation rather than assuming
// that every learned piece wins against competing Unigram paths or BPE ranks.
func cacheWords(m Model) []string {
	words := make([]string, 0, len(m.Pieces)+len(m.CacheWords))
	for _, piece := range m.Pieces {
		if !strings.Contains(piece.Text, " ") && piece.Text == Normalize(piece.Text) {
			words = append(words, piece.Text)
		}
	}
	return append(words, m.CacheWords...)
}

func (m Model) Save(path string) error {
	if _, err := New(m); err != nil {
		return err
	}
	b, err := json.Marshal(m)
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0644)
}

// Encode returns deterministic IDs. Version 2 can fuse a space with the first
// token of the following word; literal markers and special-token strings round-trip.
func (t *Tokenizer) Encode(text string) []int {
	return t.AppendEncode(nil, text)
}

// AppendEncode appends the IDs of text to dst, letting callers reuse buffers.
func (t *Tokenizer) AppendEncode(dst []int, text string) []int {
	if dst == nil {
		dst = make([]int, 0, len(text)/4+8)
	}
	if ids, ok := t.appendWords(dst, text, true); ok {
		return ids
	}
	ids, _ := t.appendWords(dst, Normalize(text), false)
	return ids
}

// appendWords encodes space-separated words. With verify it gives up, so the
// caller can normalize first, unless it proves Normalize(s) == s on the way:
// cached words are normalized, a single space between normalized words has
// normalization boundaries on both sides, and alreadyNormalized checks the rest.
func (t *Tokenizer) appendWords(dst []int, s string, verify bool) ([]int, bool) {
	ids, first := dst, len(dst)
	var local [128]viterbiState
	scratch := local[:]
	for len(s) > 0 {
		w, rest, found := strings.Cut(s, " ")
		if verify && (w == "" || found && rest == "") {
			return dst, false
		}
		if len(ids) > first {
			ids = append(ids, ByteOffset+32)
		}
		start := len(ids)
		if c := t.cache.find(w); c != nil {
			// Hits carry their space-fused first ID precomputed.
			cached := t.cache.ids[c.ids : c.ids+c.count]
			if start > first && c.fused != 0 {
				ids[start-1] = int(c.fused)
				cached = cached[1:]
			}
			for _, id := range cached {
				ids = append(ids, int(id))
			}
			s = rest
			continue
		}
		if verify && !alreadyNormalized(w) {
			return dst, false
		}
		if t.model.Algorithm == "bpe" {
			ids = t.appendBPE(ids, w)
		} else {
			if len(scratch) < len(w)+1 {
				scratch = make([]viterbiState, len(w)+1)
			}
			ids = t.appendUnigram(ids, w, scratch[:len(w)+1])
		}
		if start > first && len(t.spaceFusion) > 0 {
			if fused := t.spaceFusion[ids[start]]; fused != 0 {
				ids[start-1] = fused
				copy(ids[start:], ids[start+1:])
				ids = ids[:len(ids)-1]
			}
		}
		s = rest
	}
	return ids, true
}

// Vocabulary returns learned piece texts in ID order, starting at PieceOffset,
// and the base IDs that fused IDs expand to after a space, in fused-ID order.
func (t *Tokenizer) Vocabulary() (pieces []string, spaceFusion []int) {
	pieces = make([]string, len(t.model.Pieces))
	for i, p := range t.model.Pieces {
		pieces[i] = p.Text
	}
	return pieces, append([]int(nil), t.model.SpaceFusion...)
}

// Algorithm reports "unigram" or "bpe".
func (t *Tokenizer) Algorithm() string { return t.model.Algorithm }

// Merges returns a copy of the BPE merge rules in rank order; Unigram has none.
func (t *Tokenizer) Merges() []Merge { return append([]Merge(nil), t.model.Merges...) }

// Decode rejects reserved/invalid IDs rather than silently losing text.
func (t *Tokenizer) Decode(ids []int) (string, error) {
	var b strings.Builder
	for _, id := range ids {
		base := PieceOffset + len(t.model.Pieces)
		if id >= base && id < t.VocabSize() {
			b.WriteByte(' ')
			id = t.model.SpaceFusion[id-base]
		}
		switch {
		case id >= ByteOffset && id < PieceOffset:
			b.WriteByte(byte(id - ByteOffset))
		case id >= PieceOffset && id < PieceOffset+len(t.model.Pieces):
			b.WriteString(t.model.Pieces[id-PieceOffset].Text)
		default:
			return "", fmt.Errorf("invalid or reserved token ID %d", id)
		}
	}
	if !utf8.ValidString(b.String()) {
		return "", fmt.Errorf("token sequence is not valid UTF-8")
	}
	return b.String(), nil
}

func (t *Tokenizer) VocabSize() int {
	return PieceOffset + len(t.model.Pieces) + len(t.model.SpaceFusion)
}

// Expand returns base token IDs, undoing space fusion without decoding text.
// It rejects reserved/out-of-range IDs; UTF-8 validity is checked by Decode.
func (t *Tokenizer) Expand(ids []int) ([]int, error) {
	base := PieceOffset + len(t.model.Pieces)
	result := make([]int, 0, len(ids))
	for _, id := range ids {
		if id < ByteOffset || id >= t.VocabSize() {
			return nil, fmt.Errorf("invalid or reserved token ID %d", id)
		}
		if id >= base {
			result = append(result, ByteOffset+32, t.model.SpaceFusion[id-base])
		} else {
			result = append(result, id)
		}
	}
	return result, nil
}
