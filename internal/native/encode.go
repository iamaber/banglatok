package native

import (
	"encoding/binary"
	"sync"
	"unsafe"
)

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

// EncodeBinary accepts repeated little-endian uint32 byte lengths followed by
// UTF-8 text and encodes it with up to threads workers (see Threads). data may
// alias caller-owned memory for the duration of the call.
// The response is native-endian uint32 words: status 1 followed by an error
// message in the remaining bytes, or status 0, one ID count per text, then all
// IDs in input order.
func (r *Registry) EncodeBinary(handle uint64, data []byte, threads int) []byte {
	model, err := r.model(handle)
	if err != nil {
		return EncodeError(err.Error())
	}
	var texts []string
	for len(data) > 0 {
		if len(data) < 4 {
			return EncodeError("truncated text length")
		}
		n := uint64(binary.LittleEndian.Uint32(data))
		data = data[4:]
		if n > uint64(len(data)) {
			return EncodeError("truncated text")
		}
		texts = append(texts, unsafe.String(unsafe.SliceData(data), int(n)))
		data = data[int(n):]
	}
	counts := make([]int, len(texts))
	chunks := encodeChunks(model, texts, counts, Threads(threads))
	defer releaseChunks(chunks)
	total := 0
	for _, c := range chunks {
		total += len(c.ids)
	}
	output := make([]byte, 4*(1+len(texts)+total))
	words := unsafe.Slice((*uint32)(unsafe.Pointer(unsafe.SliceData(output))), len(output)/4)
	for i, count := range counts {
		words[1+i] = uint32(count)
	}
	ids := words[1+len(texts):]
	for _, c := range chunks {
		for i, id := range c.ids {
			ids[i] = uint32(id)
		}
		ids = ids[len(c.ids):]
	}
	return output
}

// EncodeError builds the EncodeBinary error response for message.
func EncodeError(message string) []byte {
	output := binary.NativeEndian.AppendUint32(nil, 1)
	return append(output, message...)
}
