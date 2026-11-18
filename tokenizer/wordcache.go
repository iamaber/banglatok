package tokenizer

import (
	"math/bits"
	"math/rand/v2"
	"strings"
)

// wordCache maps a word to its IDs without per-entry pointers: words and IDs
// live in two flat arrays and a linear-probing table holds their offsets.
type wordCache struct {
	seed  [3]uint64
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
func newWordCache(words []string, ids []uint32, ends []uint32) *wordCache {
	bits := uint(3)
	for 1<<bits < 2*len(words) {
		bits++
	}
	c := &wordCache{seed: [3]uint64{rand.Uint64(), rand.Uint64(), rand.Uint64()}, slots: make([]wordSlot, 1<<bits), shift: 64 - bits, words: strings.Join(words, ""), ids: ids}
	mask := uint64(len(c.slots) - 1)
	offset, start := uint32(0), uint32(0)
	for i, w := range words {
		h := c.hash(w)
		s := wordSlot{hash: uint32(h), word: offset, length: uint32(len(w)), ids: start, count: ends[i] - start}
		j := h >> c.shift
		for c.slots[j].length != 0 && c.word(&c.slots[j]) != w {
			j = (j + 1) & mask
		}
		c.slots[j] = s
		offset, start = offset+s.length, ends[i]
	}
	return c
}

func (c *wordCache) fuse(spaceFusion []int) {
	if len(spaceFusion) == 0 {
		return
	}
	for i := range c.slots {
		if s := &c.slots[i]; s.count > 0 {
			s.fused = uint32(spaceFusion[c.ids[s.ids]])
		}
	}
}

func (c *wordCache) word(s *wordSlot) string { return c.words[s.word : s.word+s.length] }

func (c *wordCache) hash(w string) uint64 {
	n := len(w)
	var a, b uint64
	seed := c.seed[0]
	switch {
	case n > 16:
		for p := w; len(p) > 16; p = p[16:] {
			seed = mix(load64(p)^c.seed[1], load64(p[8:])^seed)
		}
		a, b = load64(w[n-16:]), load64(w[n-8:])
	case n >= 8:
		a, b = load64(w), load64(w[n-8:])
	case n >= 4:
		a, b = uint64(load32(w)), uint64(load32(w[n-4:]))
	case n > 0:
		a = uint64(w[0])<<16 | uint64(w[n>>1])<<8 | uint64(w[n-1])
	}
	return mix(mix(a^c.seed[1], b^seed)^c.seed[2], uint64(n)^c.seed[1])
}

func mix(a, b uint64) uint64 {
	hi, lo := bits.Mul64(a, b)
	return hi ^ lo
}

func load64(s string) uint64 {
	_ = s[7]
	return uint64(s[0]) | uint64(s[1])<<8 | uint64(s[2])<<16 | uint64(s[3])<<24 |
		uint64(s[4])<<32 | uint64(s[5])<<40 | uint64(s[6])<<48 | uint64(s[7])<<56
}

func load32(s string) uint32 {
	_ = s[3]
	return uint32(s[0]) | uint32(s[1])<<8 | uint32(s[2])<<16 | uint32(s[3])<<24
}

// find returns nil for a miss or a nil cache.
func (c *wordCache) find(w string) *wordSlot {
	if c == nil {
		return nil
	}
	h := c.hash(w)
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
