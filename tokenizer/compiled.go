package tokenizer

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"strings"
)

// A compiled sidecar holds a JSON model in binary form with the cached word
// segmentations, so loading skips JSON parsing and re-encoding. Its header
// pins the SHA-256 of the exact JSON bytes and of its own payload; any
// mismatch, including a stale or corrupt file, falls back to the JSON.
const compiledMagic = "BNTOKBIN"

// compiledVersion covers the layout and the segmentation the payload caches.
// Bump it whenever encoding output for a word could change.
const compiledVersion = 1

const compiledHeader = len(compiledMagic) + 4 + 2*sha256.Size

// CompiledPath is the sidecar location for a JSON model path.
func CompiledPath(path string) string {
	return strings.TrimSuffix(path, ".json") + ".bntok"
}

// WriteCompiled writes the sidecar for the JSON model at path.
func WriteCompiled(path string) error {
	source, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	b, err := compile(source)
	if err != nil {
		return err
	}
	return os.WriteFile(CompiledPath(path), b, 0644)
}

func compile(source []byte) ([]byte, error) {
	t, ids, ends, err := loadJSON(source)
	if err != nil {
		return nil, err
	}
	m := t.model
	var p []byte
	u32 := func(v int) { p = binary.LittleEndian.AppendUint32(p, uint32(v)) }
	str := func(s string) { u32(len(s)); p = append(p, s...) }
	u32(m.Version)
	str(m.Algorithm)
	str(m.Normalization)
	u32(len(m.Pieces))
	for _, piece := range m.Pieces {
		str(piece.Text)
		p = binary.LittleEndian.AppendUint64(p, math.Float64bits(piece.Score))
	}
	u32(len(m.Merges))
	for _, merge := range m.Merges {
		u32(merge.Left)
		u32(merge.Right)
		u32(merge.Result)
	}
	u32(len(m.SpaceFusion))
	for _, id := range m.SpaceFusion {
		u32(id)
	}
	u32(len(m.CacheWords))
	for _, word := range m.CacheWords {
		str(word)
	}
	u32(len(ids))
	for _, id := range ids {
		u32(int(id))
	}
	u32(len(ends))
	for _, end := range ends {
		u32(int(end))
	}
	sourceSum, payloadSum := sha256.Sum256(source), sha256.Sum256(p)
	b := make([]byte, 0, compiledHeader+len(p))
	b = append(b, compiledMagic...)
	b = binary.LittleEndian.AppendUint32(b, compiledVersion)
	b = append(append(append(b, sourceSum[:]...), payloadSum[:]...), p...)
	return b, nil
}

// compiledReader decodes a payload; any overrun sets bad and yields zeros.
type compiledReader struct {
	data string
	bad  bool
}

func (r *compiledReader) u32() uint32 {
	if len(r.data) < 4 {
		r.bad = true
		return 0
	}
	v := uint32(r.data[0]) | uint32(r.data[1])<<8 | uint32(r.data[2])<<16 | uint32(r.data[3])<<24
	r.data = r.data[4:]
	return v
}

func (r *compiledReader) u64() uint64 {
	return uint64(r.u32()) | uint64(r.u32())<<32
}

// count reads a length whose items take at least size bytes each, so a
// corrupt count cannot force a huge allocation.
func (r *compiledReader) count(size int) int {
	n := int(r.u32())
	if n > len(r.data)/size {
		r.bad = true
		return 0
	}
	return n
}

func (r *compiledReader) str() string {
	n := r.count(1)
	s := r.data[:n]
	r.data = r.data[n:]
	return s
}

func loadCompiled(path string, source []byte) (*Tokenizer, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return parseCompiled(b, source)
}

func parseCompiled(b, source []byte) (*Tokenizer, error) {
	sourceSum := sha256.Sum256(source)
	if len(b) < compiledHeader || string(b[:len(compiledMagic)]) != compiledMagic ||
		binary.LittleEndian.Uint32(b[len(compiledMagic):]) != compiledVersion {
		return nil, fmt.Errorf("unsupported compiled model")
	}
	sums, payload := b[len(compiledMagic)+4:compiledHeader], b[compiledHeader:]
	if payloadSum := sha256.Sum256(payload); !bytes.Equal(sums[:sha256.Size], sourceSum[:]) || !bytes.Equal(sums[sha256.Size:], payloadSum[:]) {
		return nil, fmt.Errorf("stale or corrupt compiled model")
	}
	r := &compiledReader{data: string(payload)}
	m := Model{Version: int(r.u32()), Algorithm: r.str(), Normalization: r.str()}
	m.Pieces = make([]Piece, r.count(12))
	for i := range m.Pieces {
		m.Pieces[i] = Piece{r.str(), math.Float64frombits(r.u64())}
	}
	m.Merges = make([]Merge, r.count(12))
	for i := range m.Merges {
		m.Merges[i] = Merge{int(r.u32()), int(r.u32()), int(r.u32())}
	}
	m.SpaceFusion = make([]int, r.count(4))
	for i := range m.SpaceFusion {
		m.SpaceFusion[i] = int(r.u32())
	}
	m.CacheWords = make([]string, r.count(4))
	for i := range m.CacheWords {
		m.CacheWords[i] = r.str()
	}
	ids := make([]uint32, r.count(4))
	for i := range ids {
		ids[i] = r.u32()
	}
	ends := make([]uint32, r.count(4))
	for i := range ends {
		ends[i] = r.u32()
	}
	if r.bad || len(r.data) != 0 {
		return nil, fmt.Errorf("malformed compiled model")
	}
	// JSON omits empty lists; keep the model identical to a JSON load.
	if len(m.Merges) == 0 {
		m.Merges = nil
	}
	if len(m.SpaceFusion) == 0 {
		m.SpaceFusion = nil
	}
	if len(m.CacheWords) == 0 {
		m.CacheWords = nil
	}
	t, err := New(m)
	if err != nil {
		return nil, err
	}
	words := cacheWords(m)
	if len(ends) != len(words) {
		return nil, fmt.Errorf("malformed compiled cache")
	}
	start := uint32(0)
	for _, end := range ends {
		if end <= start || end > uint32(len(ids)) {
			return nil, fmt.Errorf("malformed compiled cache")
		}
		start = end
	}
	if start != uint32(len(ids)) {
		return nil, fmt.Errorf("malformed compiled cache")
	}
	limit := uint32(PieceOffset + len(m.Pieces))
	for _, id := range ids {
		if id < ByteOffset || id >= limit {
			return nil, fmt.Errorf("malformed compiled cache")
		}
	}
	// The checksums rule out corruption; also re-encode a spread of words in
	// case the payload came from a build whose segmentation differs.
	start = 0
	step := max(1, len(words)/256)
	for i := range words {
		if i%step == 0 {
			got := t.Encode(words[i])
			want := ids[start:ends[i]]
			if len(got) != len(want) {
				return nil, fmt.Errorf("compiled cache disagrees with encoder")
			}
			for j, id := range got {
				if uint32(id) != want[j] {
					return nil, fmt.Errorf("compiled cache disagrees with encoder")
				}
			}
		}
		start = ends[i]
	}
	t.cache = newWordCache(words, ids, ends, t.spaceFusion)
	return t, nil
}
