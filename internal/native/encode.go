package native

import "sync"

// Scratch ID buffers are reused across calls; returned data never aliases them.
var scratch = sync.Pool{New: func() any { return new([]int) }}

// maxPooledIDs keeps one unusually large request from pinning its buffer.
const maxPooledIDs = 1 << 16

func release(buffer *[]int, ids []int) {
	if cap(ids) <= maxPooledIDs {
		*buffer = ids
		scratch.Put(buffer)
	}
}

// EncodeInto encodes one UTF-8 text into out and returns its ID count. A count
// larger than len(out) means out was too small and holds no complete result.
// text may alias caller-owned memory: nothing derived from it is retained.
func (r *Registry) EncodeInto(handle uint64, text string, out []uint32) (int, error) {
	model, err := r.model(handle)
	if err != nil {
		return 0, err
	}
	buffer := scratch.Get().(*[]int)
	ids := model.AppendEncode((*buffer)[:0], text)
	if len(ids) <= len(out) {
		for i, id := range ids {
			out[i] = uint32(id)
		}
	}
	release(buffer, ids)
	return len(ids), nil
}
