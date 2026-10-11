package tokenizer

import (
	"container/heap"
	"fmt"
	"io"
	"math"
	"sort"
	"unicode/utf8"
)

// Merge contains piece indexes (not external IDs) and is applied in list order.
type Merge struct {
	Left   int `json:"left"`
	Right  int `json:"right"`
	Result int `json:"result"`
}

// denseRunes covers Latin, Bengali and general punctuation with a slice lookup.
const denseRunes = 0x2100

func (t *Tokenizer) initBPE() error {
	t.model.Merges = append([]Merge(nil), t.model.Merges...)
	t.mergeRanks = map[uint64]int{}
	t.characters = map[rune]int{}
	t.denseCharacters = make([]int32, denseRunes)
	for i := range t.denseCharacters {
		t.denseCharacters[i] = -1
	}
	available := map[int]bool{}
	for i, p := range t.model.Pieces {
		if utf8.RuneCountInString(p.Text) == 1 {
			r, _ := utf8.DecodeRuneInString(p.Text)
			if r < denseRunes {
				t.denseCharacters[r] = int32(i)
			} else {
				t.characters[r] = i
			}
			available[i] = true
		}
	}
	for rank, m := range t.model.Merges {
		if !available[m.Left] || !available[m.Right] || m.Result < 0 || m.Result >= len(t.model.Pieces) {
			return fmt.Errorf("invalid BPE merge")
		}
		if t.model.Pieces[m.Result].Text != t.model.Pieces[m.Left].Text+t.model.Pieces[m.Right].Text {
			return fmt.Errorf("BPE merge text mismatch")
		}
		pair := pairKey(PieceOffset+m.Left, PieceOffset+m.Right)
		if _, ok := t.mergeRanks[pair]; ok {
			return fmt.Errorf("duplicate BPE merge")
		}
		t.mergeRanks[pair] = rank
		available[m.Result] = true
	}
	if len(available) != len(t.model.Pieces) {
		return fmt.Errorf("BPE vocabulary contains unreachable pieces")
	}
	return nil
}

func pairKey(left, right int) uint64 {
	return uint64(uint32(left))<<32 | uint64(uint32(right))
}

func (t *Tokenizer) character(r rune) (int, bool) {
	if r < denseRunes {
		id := t.denseCharacters[r]
		return int(id), id >= 0
	}
	id, ok := t.characters[r]
	return id, ok
}

func (t *Tokenizer) rank(left, right int) int32 {
	if rank, ok := t.mergeRanks[pairKey(left, right)]; ok {
		return int32(rank)
	}
	return math.MaxInt32
}

// appendBPE applies the lowest-ranked merge to every non-overlapping
// occurrence, matching training, until no merge applies. Pair ranks are kept
// beside the word and recomputed only next to merged tokens.
func (t *Tokenizer) appendBPE(ids []int, s string) []int {
	start := len(ids)
	for _, r := range s {
		if id, ok := t.character(r); ok {
			ids = append(ids, PieceOffset+id)
		} else {
			var encoded [utf8.UTFMax]byte
			for _, b := range encoded[:utf8.EncodeRune(encoded[:], r)] {
				ids = append(ids, ByteOffset+int(b))
			}
		}
	}
	word := ids[start:]
	if len(word) < 2 {
		return ids
	}
	var local [64]int32
	ranks := local[:]
	if len(word) > len(ranks) {
		ranks = make([]int32, len(word))
	}
	for i := 0; i+1 < len(word); i++ {
		ranks[i] = t.rank(word[i], word[i+1])
	}
	for len(word) > 1 {
		best, pos := int32(math.MaxInt32), -1
		for i, rank := range ranks[:len(word)-1] {
			if rank < best {
				best, pos = rank, i
			}
		}
		if pos < 0 {
			break
		}
		merge := t.model.Merges[best]
		left, right, result := PieceOffset+merge.Left, PieceOffset+merge.Right, PieceOffset+merge.Result
		// Compact in place; an unmerged token keeps its old right-pair rank,
		// which stays valid unless its right neighbour was merged.
		n := 0
		for i := 0; i < len(word); i++ {
			if i+1 < len(word) && word[i] == left && word[i+1] == right {
				word[n] = result
				i++
			} else {
				word[n], ranks[n] = word[i], ranks[i]
			}
			n++
		}
		word = word[:n]
		for i, id := range word {
			if id != result {
				continue
			}
			if i > 0 {
				ranks[i-1] = t.rank(word[i-1], id)
			}
			if i+1 < len(word) {
				ranks[i] = t.rank(id, word[i+1])
			}
		}
	}
	return ids[:start+len(word)]
}

type pairItem struct {
	pair     [2]int
	count    float64
	revision int
}
type pairHeap []pairItem

func (h pairHeap) Len() int { return len(h) }
func (h pairHeap) Less(i, j int) bool {
	a, b := h[i], h[j]
	if a.count != b.count {
		return a.count > b.count
	}
	if a.pair[0] != b.pair[0] {
		return a.pair[0] < b.pair[0]
	}
	return a.pair[1] < b.pair[1]
}
func (h pairHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *pairHeap) Push(v any)   { *h = append(*h, v.(pairItem)) }
func (h *pairHeap) Pop() any     { a := *h; v := a[len(a)-1]; *h = a[:len(a)-1]; return v }

// TrainBPE updates only words affected by a merge. Corpus weights, normalization,
// word boundaries, all observed characters and fallback match the Unigram run.
func TrainBPE(counts map[string]float64, vocab, maxPieceRunes int, log io.Writer) (Model, error) {
	m := Model{Version: 1, Algorithm: "bpe", Normalization: Normalization}
	if vocab <= PieceOffset || maxPieceRunes < 1 {
		return m, fmt.Errorf("invalid BPE configuration")
	}
	input, valid := prepareTrainingInput(counts)
	if !valid {
		return m, fmt.Errorf("invalid word")
	}
	alphabet := input.alphabet()
	if len(alphabet)+PieceOffset > vocab {
		return m, fmt.Errorf("vocabulary too small")
	}
	index := map[string]int{}
	for _, s := range alphabet {
		index[s] = len(m.Pieces)
		m.Pieces = append(m.Pieces, Piece{s, 0})
	}
	words := make([][]int, len(input.words))
	freq := make([]float64, len(input.words))
	for i, w := range input.words {
		freq[i] = w.count
		for _, r := range w.text {
			words[i] = append(words[i], index[string(r)])
		}
	}
	totals := map[[2]int]float64{}
	members := map[[2]int]map[int]bool{}
	revisions := map[[2]int]int{}
	done := map[[2]int]bool{}
	local := func(ids []int) map[[2]int]int {
		out := map[[2]int]int{}
		for i := 0; i+1 < len(ids); i++ {
			out[[2]int{ids[i], ids[i+1]}]++
		}
		return out
	}
	for i, ids := range words {
		for p, n := range local(ids) {
			totals[p] += float64(n) * freq[i]
			if members[p] == nil {
				members[p] = map[int]bool{}
			}
			members[p][i] = true
		}
	}
	h := &pairHeap{}
	for p, n := range totals {
		heap.Push(h, pairItem{p, n, 0})
	}
	for len(m.Pieces)+PieceOffset < vocab {
		if h.Len() == 0 {
			return m, fmt.Errorf("insufficient merge candidates for vocabulary %d", vocab)
		}
		item := heap.Pop(h).(pairItem)
		p := item.pair
		if item.revision != revisions[p] || done[p] || item.count <= 1e-12 {
			continue
		}
		text := m.Pieces[p[0]].Text + m.Pieces[p[1]].Text
		if utf8.RuneCountInString(text) > maxPieceRunes {
			done[p] = true
			continue
		}
		result, ok := index[text]
		if !ok {
			result = len(m.Pieces)
			index[text] = result
			m.Pieces = append(m.Pieces, Piece{text, 0})
		}
		m.Merges = append(m.Merges, Merge{p[0], p[1], result})
		done[p] = true
		affected := make([]int, 0, len(members[p]))
		for i := range members[p] {
			affected = append(affected, i)
		}
		sort.Ints(affected)
		changed := map[[2]int]bool{}
		for _, i := range affected {
			old := words[i]
			before := local(old)
			afterIDs := make([]int, 0, len(old))
			for j := 0; j < len(old); j++ {
				if j+1 < len(old) && old[j] == p[0] && old[j+1] == p[1] {
					afterIDs = append(afterIDs, result)
					j++
				} else {
					afterIDs = append(afterIDs, old[j])
				}
			}
			words[i] = afterIDs
			after := local(afterIDs)
			for q, n := range before {
				totals[q] -= float64(n) * freq[i]
				delete(members[q], i)
				changed[q] = true
			}
			for q, n := range after {
				totals[q] += float64(n) * freq[i]
				if members[q] == nil {
					members[q] = map[int]bool{}
				}
				members[q][i] = true
				changed[q] = true
			}
		}
		for q := range changed {
			revisions[q]++
			if !done[q] && totals[q] > 1e-12 {
				heap.Push(h, pairItem{q, totals[q], revisions[q]})
			}
		}
		if log != nil && len(m.Merges)%1000 == 0 {
			fmt.Fprintf(log, "merges=%d vocabulary=%d\n", len(m.Merges), len(m.Pieces)+PieceOffset)
		}
	}
	return m, nil
}
