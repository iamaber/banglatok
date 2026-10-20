package tokenizer

import (
	"hash/maphash"
	"strings"
)

// wordCache maps a word to its IDs without per-entry pointers: words and IDs
// live in two flat arrays and a linear-probing table holds their offsets.
type wordCache struct {
	seed  maphash.Seed
	slots []wordSlot
	shift uint
	words string
	ids   []uint32
}

// wordSlot is empty when length is 0; cached words are never empty. Offsets
// are 32-bit because words are model pieces or cache words.
type wordSlot struct {
	hash, word, length uint32
	ids, count         uint32
	// fused replaces a preceding space and the first ID, or is 0.
	fused uint32
}

// newWordCache indexes words[i], whose IDs are ids[ends[i-1]:ends[i]].
// A repeated word keeps its last IDs.
func newWordCache(words []string, ids []uint32, ends []uint32, spaceFusion []int) *wordCache {
	bits := uint(3)
	for 1<<bits < 2*len(words) {
		bits++
	}
	c := &wordCache{seed: maphash.MakeSeed(), slots: make([]wordSlot, 1<<bits), shift: 64 - bits, words: strings.Join(words, ""), ids: ids}
	mask := uint64(len(c.slots) - 1)
	offset, start := uint32(0), uint32(0)
	for i, w := range words {
		h := maphash.String(c.seed, w)
		s := wordSlot{hash: uint32(h), word: offset, length: uint32(len(w)), ids: start, count: ends[i] - start}
		if len(spaceFusion) > 0 && s.count > 0 {
			s.fused = uint32(spaceFusion[ids[start]])
		}
		j := h >> c.shift
		for c.slots[j].length != 0 && c.word(&c.slots[j]) != w {
			j = (j + 1) & mask
		}
		c.slots[j] = s
		offset, start = offset+s.length, ends[i]
	}
	return c
}

func (c *wordCache) word(s *wordSlot) string { return c.words[s.word : s.word+s.length] }

// find returns nil for a miss or a nil cache.
func (c *wordCache) find(w string) *wordSlot {
	if c == nil {
		return nil
	}
	h := maphash.String(c.seed, w)
	mask := uint64(len(c.slots) - 1)
	for i := h >> c.shift; ; i = (i + 1) & mask {
		s := &c.slots[i]
		if s.length == 0 {
			return nil
		}
		if s.hash == uint32(h) && int(s.length) == len(w) && c.word(s) == w {
			return s
		}
	}
}
