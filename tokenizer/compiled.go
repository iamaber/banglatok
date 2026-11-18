package tokenizer

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
)

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

func (r *compiledReader) u32s() []uint32 {
	out := make([]uint32, r.count(4))
	for i := range out {
		v := r.data[4*i : 4*i+4]
		out[i] = uint32(v[0]) | uint32(v[1])<<8 | uint32(v[2])<<16 | uint32(v[3])<<24
	}
	r.data = r.data[4*len(out):]
	return out
}

func loadSidecar(path string) (*Tokenizer, error) {
	b, err := os.ReadFile(CompiledPath(path))
	if err != nil {
		return nil, err
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return parseCompiled(b, bufio.NewReaderSize(f, 256<<10))
}

func parseCompiled(b []byte, source io.Reader) (*Tokenizer, error) {
	if len(b) < compiledHeader || string(b[:len(compiledMagic)]) != compiledMagic ||
		binary.LittleEndian.Uint32(b[len(compiledMagic):]) != compiledVersion {
		return nil, fmt.Errorf("unsupported compiled model")
	}
	sums, payload := b[len(compiledMagic)+4:compiledHeader], b[compiledHeader:]
	sourceOK, payloadOK := make(chan bool, 1), make(chan bool, 1)
	go func() {
		h := sha256.New()
		_, err := io.Copy(h, source)
		sourceOK <- err == nil && bytes.Equal(sums[:sha256.Size], h.Sum(nil))
	}()
	go func() {
		sum := sha256.Sum256(payload)
		payloadOK <- bytes.Equal(sums[sha256.Size:], sum[:])
	}()
	t, err := decodeCompiled(payload)
	sourceMatches, payloadMatches := <-sourceOK, <-payloadOK
	if !sourceMatches || !payloadMatches {
		return nil, fmt.Errorf("stale or corrupt compiled model")
	}
	return t, err
}

func decodeCompiled(payload []byte) (*Tokenizer, error) {
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
	ids := r.u32s()
	ends := r.u32s()
	if r.bad || len(r.data) != 0 {
		return nil, fmt.Errorf("malformed compiled model")
	}
	// JSON omits empty lists; keep the model identical to a JSON load.
	if len(m.Pieces) == 0 {
		m.Pieces = nil
	}
	if len(m.Merges) == 0 {
		m.Merges = nil
	}
	if len(m.SpaceFusion) == 0 {
		m.SpaceFusion = nil
	}
	if len(m.CacheWords) == 0 {
		m.CacheWords = nil
	}
	var words []string
	var cache *wordCache
	cacheErr := make(chan error, 1)
	go func() {
		words = cacheWords(m)
		err := checkCompiledCache(words, ids, ends, uint32(PieceOffset+len(m.Pieces)))
		if err == nil {
			cache = newWordCache(words, ids, ends)
		}
		cacheErr <- err
	}()
	t, err := newOwned(m)
	if e := <-cacheErr; err == nil {
		err = e
	}
	if err != nil {
		return nil, err
	}
	// The checksums rule out corruption; also re-encode a spread of words in
	// case the payload came from a build whose segmentation differs.
	start := uint32(0)
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
	cache.fuse(t.spaceFusion)
	t.cache = cache
	return t, nil
}

func checkCompiledCache(words []string, ids, ends []uint32, limit uint32) error {
	if len(ends) != len(words) {
		return fmt.Errorf("malformed compiled cache")
	}
	start := uint32(0)
	for _, end := range ends {
		if end <= start || end > uint32(len(ids)) {
			return fmt.Errorf("malformed compiled cache")
		}
		start = end
	}
	if start != uint32(len(ids)) {
		return fmt.Errorf("malformed compiled cache")
	}
	for _, id := range ids {
		if id < ByteOffset || id >= limit {
			return fmt.Errorf("malformed compiled cache")
		}
	}
	return nil
}
