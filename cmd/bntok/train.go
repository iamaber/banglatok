package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"github.com/iamaber/banglatok/corpus"
	"github.com/iamaber/banglatok/tokenizer"
)

func train(f *flag.FlagSet, args []string) error {
	algorithm := f.String("algorithm", "unigram", "unigram or bpe")
	input := f.String("input", "data/processed/train.jsonl", "training records")
	out := f.String("out", "models/unigram-32000.json", "output model")
	vocab := f.Int("vocab", 32000, "total vocabulary including 260 reserved/byte IDs")
	seed := f.Int("seed-size", 64000, "initial vocabulary size")
	em := f.Int("em", 2, "EM iterations per pruning round")
	length := f.Int("max-piece-runes", 12, "maximum learned piece length")
	balanced := f.Bool("balanced", false, "use 25/25/25/25 rather than 40/25/25/10 sentence mixture")
	if err := f.Parse(args); err != nil {
		return err
	}
	mixture := map[string]float64{"bangla": .4, "english": .25, "banglish": .25, "code-mixed": .1}
	if *balanced {
		for k := range mixture {
			mixture[k] = .25
		}
	}
	counts, err := corpus.WordCounts(*input, mixture)
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "training on %d distinct words\n", len(counts))
	started := time.Now()
	var m tokenizer.Model
	switch *algorithm {
	case "unigram":
		m, err = tokenizer.Train(counts, tokenizer.TrainConfig{VocabSize: *vocab, SeedSize: *seed, MaxPieceRunes: *length, EMIterations: *em, Log: os.Stderr})
	case "bpe":
		m, err = tokenizer.TrainBPE(counts, *vocab, *length, os.Stderr)
	default:
		return fmt.Errorf("unknown algorithm %q", *algorithm)
	}
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(*out), 0755); err != nil {
		return err
	}
	if err := m.Save(*out); err != nil {
		return err
	}
	hash, err := corpus.FileHash(*input)
	if err != nil {
		return err
	}
	modelHash, err := corpus.FileHash(*out)
	if err != nil {
		return err
	}
	meta, err := os.Create(*out + ".meta.json")
	if err != nil {
		return err
	}
	defer meta.Close()
	return writeJSON(meta, map[string]any{"algorithm": *algorithm, "vocabulary": len(m.Pieces) + tokenizer.PieceOffset, "seed_size": *seed, "em_iterations": *em, "max_piece_runes": *length, "mixture": mixture, "weight_unit": "equal sentence mass; inverse word-count within each sentence", "train_sha256": hash, "model_sha256": modelHash, "go": runtime.Version(), "os": runtime.GOOS, "arch": runtime.GOARCH, "training_seconds": time.Since(started).Seconds()})
}
