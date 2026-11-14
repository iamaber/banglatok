package native

import (
	"errors"
	"math"

	"github.com/iamaber/banglatok/tokenizer"
)

// Control IDs added by Layout.Special, matching Python's BOS_ID and EOS_ID;
// padding uses PAD_ID 0.
const (
	bosID = 2
	eosID = 3
)

// ErrBatchLengths reports code-point counts that do not partition the text.
var ErrBatchLengths = errors.New("batch lengths do not match the text")

// ErrLayout reports a Layout that Python never sends.
var ErrLayout = errors.New("invalid batch layout")

// ErrBatchSize reports output larger than the native interface can return.
var ErrBatchSize = errors.New("encoded batch is too large")

// Layout shapes output rows the way Python's Tokenizer.__call__ documents.
type Layout struct {
	MaxLength int  // keep at most this many IDs per row, controls included; 0 keeps all
	Special   bool // wrap each row in <s> ... </s>
	Width     int  // 0: unpadded rows; -1: pad to the longest row; n: pad to n
}

// Batch holds encoded IDs until Write copies them out in Layout order.
type Batch struct {
	chunks  []chunk
	counts  []int
	layout  Layout
	workers int
	width   int
	size    int
}

// EncodeBatch splits text into len(runes) texts of runes[i] code points and
// encodes them with up to threads workers. text may alias caller-owned memory
// until the Batch is written or released.
func (r *Registry) EncodeBatch(handle uint64, text string, runes []uint32, threads int, layout Layout) (*Batch, error) {
	model, err := r.model(handle)
	if err != nil {
		return nil, err
	}
	if !layout.valid() {
		return nil, ErrLayout
	}
	texts, ok := splitRunes(text, runes)
	if !ok {
		return nil, ErrBatchLengths
	}
	return encodeBatch(model, texts, Threads(threads), layout)
}

func encodeBatch(model *tokenizer.Tokenizer, texts []string, threads int, layout Layout) (*Batch, error) {
	b := &Batch{counts: make([]int, len(texts)), layout: layout}
	b.chunks, b.workers = encodeChunks(model, texts, b.counts, threads)
	size := 0
	for i := range b.counts {
		n := b.row(i)
		size += n
		b.width = max(b.width, n)
	}
	switch {
	case layout.Width > 0:
		b.width = layout.Width
		fallthrough
	case layout.Width < 0:
		if b.width > 0 && len(texts) > math.MaxInt32/b.width {
			b.Release()
			return nil, ErrBatchSize
		}
		size = len(texts) * b.width
	}
	if size > math.MaxInt32 {
		b.Release()
		return nil, ErrBatchSize
	}
	b.size = size
	return b, nil
}

func (l Layout) valid() bool {
	least := 1
	if l.Special {
		least = 2
	}
	if l.MaxLength < 0 || (l.MaxLength > 0 && l.MaxLength < least) || l.Width < -1 {
		return false
	}
	// A fixed width must hold every truncated row.
	return l.Width <= 0 || (l.MaxLength > 0 && l.MaxLength <= l.Width)
}

// row returns the unpadded output length of text i.
func (b *Batch) row(i int) int {
	n, reserve := b.counts[i], 0
	if b.layout.Special {
		reserve = 2
	}
	if b.layout.MaxLength > 0 {
		n = min(n, b.layout.MaxLength-reserve)
	}
	return n + reserve
}

// Size is the number of IDs Write needs, padding included.
func (b *Batch) Size() int { return b.size }

// Width is the padded row width, or the longest row when unpadded.
func (b *Batch) Width() int { return b.width }

// Write fills out (len >= Size) with the rows and rows[i] with the unpadded
// length of row i, then releases the batch.
func (b *Batch) Write(out []uint32, rows []uint32) {
	starts := make([]int, len(b.counts))
	offset := 0
	for i := range b.counts {
		rows[i] = uint32(b.row(i))
		starts[i] = offset
		if b.layout.Width != 0 {
			offset += b.width
		} else {
			offset += int(rows[i])
		}
	}
	write := func(c *chunk) {
		ids := c.ids
		for i := c.first; i < c.last; i++ {
			raw := ids[:b.counts[i]]
			ids = ids[len(raw):]
			row := out[starts[i]:]
			n := int(rows[i])
			if b.layout.Special {
				row[0], row[n-1] = bosID, eosID
				raw = raw[:n-2]
				row = row[1:]
			} else {
				raw = raw[:n]
			}
			for j, id := range raw {
				row[j] = uint32(id)
			}
			if b.layout.Width != 0 {
				clear(out[starts[i]+n : starts[i]+b.width])
			}
		}
	}
	parallel(b.chunks, b.workers, write)
	b.Release()
}

// Release returns scratch buffers; the Batch must not be used afterwards.
func (b *Batch) Release() {
	releaseChunks(b.chunks)
	b.chunks = nil
}

// splitRunes cuts UTF-8 text into consecutive texts of the given code-point
// counts; it reports false unless they cover text exactly.
func splitRunes(text string, runes []uint32) ([]string, bool) {
	texts := make([]string, len(runes))
	start := 0
	for i, n := range runes {
		end := start
		for end < len(text) {
			if text[end]&0xC0 != 0x80 {
				if n == 0 {
					break
				}
				n--
			}
			end++
		}
		if n != 0 {
			return nil, false
		}
		texts[i] = text[start:end]
		start = end
	}
	return texts, start == len(text)
}
