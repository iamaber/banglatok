package main

import (
	"bytes"
	"testing"
)

func TestWriteJSONFormatting(t *testing.T) {
	var output bytes.Buffer
	if err := writeJSON(&output, map[string]string{"text": "বাংলা <>&"}); err != nil {
		t.Fatal(err)
	}
	want := "{\n  \"text\": \"বাংলা \\u003c\\u003e\\u0026\"\n}\n"
	if got := output.String(); got != want {
		t.Fatalf("writeJSON = %q, want %q", got, want)
	}
}
