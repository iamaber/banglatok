package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/iamaber/banglatok/corpus"
	"github.com/iamaber/banglatok/tokenizer"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: bntok prepare|train|encode|evaluate [flags]")
	}
	f := flag.NewFlagSet(args[0], flag.ContinueOnError)
	switch args[0] {
	case "audit":
		dir := f.String("input", "data/processed", "prepared directory")
		raw := f.String("raw", "data/raw", "raw official held-out sources")
		if err := f.Parse(args[1:]); err != nil {
			return err
		}
		report, err := corpus.Audit(*dir)
		if err != nil {
			return err
		}
		n, err := corpus.AuditOfficial(*dir, *raw)
		if err != nil {
			return err
		}
		report["official_heldout_keys_checked"] = n
		report["official_heldout_train_overlap"] = 0
		return writeJSON(os.Stdout, report)
	case "fetch":
		manifest := f.String("manifest", "data/sources.lock.json", "frozen download manifest")
		root := f.String("raw", "data/raw", "download directory")
		if err := f.Parse(args[1:]); err != nil {
			return err
		}
		return corpus.Fetch(*manifest, *root)
	case "prepare":
		manifest := f.String("manifest", "data/sources.lock.json", "frozen download manifest")
		raw := f.String("raw", "data/raw", "download directory")
		out := f.String("out", "data/processed", "split directory")
		if err := f.Parse(args[1:]); err != nil {
			return err
		}
		if err := corpus.VerifyManifest(*manifest, *raw); err != nil {
			return err
		}
		r, err := corpus.LoadSources(*raw)
		if err != nil {
			return err
		}
		s, err := corpus.Prepare(r, *out)
		if err != nil {
			return err
		}
		return writeJSON(os.Stdout, s)
	case "prepare-records":
		input := f.String("input", "", "provenance-preserving corpus records JSONL")
		out := f.String("out", "", "new split directory; existing paths are refused")
		if err := f.Parse(args[1:]); err != nil {
			return err
		}
		if *input == "" || *out == "" {
			return fmt.Errorf("prepare-records requires -input and -out")
		}
		if _, err := os.Stat(*out); !os.IsNotExist(err) {
			return fmt.Errorf("output exists or cannot be inspected; use a new split directory")
		}
		var records []corpus.Record
		if err := corpus.Read(*input, func(r corpus.Record) error {
			records = append(records, r)
			return nil
		}); err != nil {
			return err
		}
		s, err := corpus.Prepare(records, *out)
		if err != nil {
			return err
		}
		return writeJSON(os.Stdout, s)
	case "train":
		return train(f, args[1:])
	case "encode":
		path := f.String("model", "models/unigram-32000.json", "model file")
		text := f.String("text", "আমি আজ ভালো আছি। ami aj bhalo achi", "text to encode")
		if err := f.Parse(args[1:]); err != nil {
			return err
		}
		t, err := tokenizer.Load(*path)
		if err != nil {
			return err
		}
		ids := t.Encode(*text)
		decoded, err := t.Decode(ids)
		if err != nil {
			return err
		}
		return writeJSON(os.Stdout, map[string]any{"ids": ids, "decoded": decoded, "tokens": len(ids)})
	case "evaluate":
		return evaluate(f, args[1:])
	default:
		return fmt.Errorf("unknown command %q", args[0])
	}
}
