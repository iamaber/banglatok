// Command pgo_profile writes the CPU profile used for profile-guided
// optimization of the shared library. It loads each model through the native
// registry and encodes a JSONL corpus with both single-text and batch calls:
//
//	go run ./tools/pgo_profile -corpus data/processed/train.jsonl \
//	  -out cmd/bntok-shared/default.pgo src/banglatok/pretrained/*.json
package main

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"runtime/pprof"

	"github.com/iamaber/banglatok/internal/native"
)

func main() {
	corpus := flag.String("corpus", "", "JSONL file with a text field")
	every := flag.Int("every", 1, "use every n-th record")
	output := flag.String("out", "default.pgo", "profile path")
	flag.Parse()
	if *corpus == "" || flag.NArg() == 0 || *every < 1 {
		flag.Usage()
		os.Exit(2)
	}
	texts, err := read(*corpus, *every)
	check(err)
	f, err := os.Create(*output)
	check(err)
	check(pprof.StartCPUProfile(f))
	var registry native.Registry
	for _, path := range flag.Args() {
		request, _ := json.Marshal(map[string]string{"op": "load", "path": path})
		var loaded struct {
			Handle uint64 `json:"handle"`
			Error  string `json:"error"`
		}
		check(json.Unmarshal(registry.Call(request), &loaded))
		if loaded.Error != "" {
			check(fmt.Errorf("%s: %s", path, loaded.Error))
		}
		out := make([]uint32, 1<<16)
		for _, text := range texts {
			_, err := registry.EncodeInto(loaded.Handle, text, out)
			check(err)
		}
		for i := 0; i < len(texts); i += 64 {
			var batch []byte
			for _, text := range texts[i:min(i+64, len(texts))] {
				batch = binary.LittleEndian.AppendUint32(batch, uint32(len(text)))
				batch = append(batch, text...)
			}
			registry.EncodeBinary(loaded.Handle, batch)
		}
	}
	pprof.StopCPUProfile()
	check(f.Close())
}

func read(path string, every int) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var texts []string
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1<<16), 64<<20)
	for n := 0; scanner.Scan(); n++ {
		if n%every != 0 {
			continue
		}
		var record struct {
			Text string `json:"text"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &record); err != nil {
			return nil, err
		}
		texts = append(texts, record.Text)
	}
	return texts, scanner.Err()
}

func check(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
