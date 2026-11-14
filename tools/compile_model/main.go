// Command compile_model writes the compiled sidecar beside each JSON model.
//
//	go run ./tools/compile_model src/banglatok/pretrained/*.json
package main

import (
	"fmt"
	"os"

	"github.com/iamaber/banglatok/tokenizer"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: compile_model MODEL.json...")
		os.Exit(2)
	}
	for _, path := range os.Args[1:] {
		if err := tokenizer.WriteCompiled(path); err != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", path, err)
			os.Exit(1)
		}
		fmt.Println(tokenizer.CompiledPath(path))
	}
}
