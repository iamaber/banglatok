package native

import (
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/iamaber/banglatok/tokenizer"
)

// Work sizes in input bytes, measured on Apple M5: below minParallelBytes a
// batch is encoded on the calling goroutine because worker start-up costs more
// than it saves; each worker gets at least minWorkerBytes; chunks are at least
// minChunkBytes so scheduling stays cheap while uneven texts still balance.
const (
	minParallelBytes = 8 << 10
	minWorkerBytes   = 2 << 10
	minChunkBytes    = 1 << 10
)

// chunk is a contiguous run of texts encoded by one worker.
type chunk struct {
	first, last int
	buffer      *[]int
	ids         []int
}

// Threads resolves a requested worker count; n <= 0 means GOMAXPROCS.
func Threads(n int) int {
	if n <= 0 {
		return runtime.GOMAXPROCS(0)
	}
	return n
}

// encodeChunks encodes texts with up to threads workers. Chunks hold IDs in
// input order and counts[i] receives the ID count of texts[i]. Callers must
// pass the chunks to releaseChunks.
func encodeChunks(model *tokenizer.Tokenizer, texts []string, counts []int, threads int) []chunk {
	chunks, workers := split(texts, threads)
	encode := func(c *chunk) {
		c.buffer = scratch.Get().(*[]int)
		ids := (*c.buffer)[:0]
		for i := c.first; i < c.last; i++ {
			start := len(ids)
			ids = model.AppendEncode(ids, texts[i])
			counts[i] = len(ids) - start
		}
		c.ids = ids
	}
	if len(chunks) == 1 {
		encode(&chunks[0])
		return chunks
	}
	var next atomic.Int64
	var done sync.WaitGroup
	for range workers {
		done.Add(1)
		go func() {
			defer done.Done()
			for i := int(next.Add(1) - 1); i < len(chunks); i = int(next.Add(1) - 1) {
				encode(&chunks[i])
			}
		}()
	}
	done.Wait()
	return chunks
}

// split partitions texts into chunks of similar byte size, about four per
// worker so that uneven text costs still balance, and picks the worker count.
func split(texts []string, threads int) ([]chunk, int) {
	total := 0
	for _, text := range texts {
		total += len(text)
	}
	workers := min(threads, total/minWorkerBytes, len(texts))
	if workers <= 1 || total < minParallelBytes {
		return []chunk{{first: 0, last: len(texts)}}, 1
	}
	target := max(total/(4*workers), minChunkBytes)
	chunks := make([]chunk, 0, total/target+1)
	first, size := 0, 0
	for i, text := range texts {
		size += len(text)
		if size >= target {
			chunks = append(chunks, chunk{first: first, last: i + 1})
			first, size = i+1, 0
		}
	}
	if first < len(texts) {
		chunks = append(chunks, chunk{first: first, last: len(texts)})
	}
	return chunks, min(workers, len(chunks))
}

func releaseChunks(chunks []chunk) {
	for _, c := range chunks {
		release(c.buffer, c.ids)
	}
}
