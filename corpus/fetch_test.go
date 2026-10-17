package corpus

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestManifestReadErrors(t *testing.T) {
	for name, check := range map[string]func(string, string) error{
		"verify": VerifyManifest,
		"fetch":  Fetch,
	} {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			manifest := filepath.Join(dir, "lock.json")
			if err := check(manifest, dir); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("expected missing manifest error, got %v", err)
			}
			if err := os.WriteFile(manifest, []byte("["), 0644); err != nil {
				t.Fatal(err)
			}
			var syntaxErr *json.SyntaxError
			if err := check(manifest, dir); !errors.As(err, &syntaxErr) {
				t.Fatalf("expected JSON syntax error, got %v", err)
			}
			if err := os.WriteFile(manifest, []byte("[]"), 0644); err != nil {
				t.Fatal(err)
			}
			if err := check(manifest, dir); err == nil || err.Error() != "empty source manifest" {
				t.Fatalf("expected empty source manifest, got %v", err)
			}
		})
	}
}

func TestManifestValidationDifferences(t *testing.T) {
	for _, tc := range []struct {
		name string
		data string
		url  string
	}{
		{"HTTP URL", "hello", "http://example.invalid/sample.txt"},
		{"empty file", "", "https://example.invalid/sample.txt"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "sample.txt")
			if err := os.WriteFile(path, []byte(tc.data), 0644); err != nil {
				t.Fatal(err)
			}
			hash, err := FileHash(path)
			if err != nil {
				t.Fatal(err)
			}
			manifest := writeDownloadManifest(t, dir, Download{Path: "sample.txt", URL: tc.url, Bytes: int64(len(tc.data)), SHA256: hash})
			if err := VerifyManifest(manifest, dir); err != nil {
				t.Fatal(err)
			}
			if err := Fetch(manifest, dir); err == nil || err.Error() != `invalid manifest entry "sample.txt"` {
				t.Fatalf("expected invalid manifest entry, got %v", err)
			}
		})
	}
}

func TestVerifyManifestEntryOrder(t *testing.T) {
	dir := t.TempDir()
	manifest := writeDownloadManifest(t, dir, Download{Path: "missing.txt"}, Download{Path: "../invalid"})
	if err := VerifyManifest(manifest, dir); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("expected first file's open error, got %v", err)
	}
}

func TestFetchKeepsEarlierDownload(t *testing.T) {
	dir := t.TempDir()
	entry := Download{
		Path:   "nested/sample.txt",
		URL:    "https://example.invalid/sample.txt",
		SHA256: "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
		Bytes:  5,
	}
	manifest := writeDownloadManifest(t, dir, entry, Download{Path: "../invalid"})
	previous := http.DefaultTransport
	t.Cleanup(func() { http.DefaultTransport = previous })
	calls := 0
	http.DefaultTransport = downloadTransport(func(*http.Request) (*http.Response, error) {
		calls++
		return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader("hello"))}, nil
	})
	for range 2 {
		if err := Fetch(manifest, dir); err == nil || err.Error() != `invalid manifest entry "../invalid"` {
			t.Fatalf("expected later entry's validation error, got %v", err)
		}
		data, err := os.ReadFile(filepath.Join(dir, entry.Path))
		if err != nil || string(data) != "hello" {
			t.Fatalf("earlier download was not saved: %q, %v", data, err)
		}
	}
	if calls != 1 {
		t.Fatalf("expected one download followed by cache reuse, got %d requests", calls)
	}
}

type downloadTransport func(*http.Request) (*http.Response, error)

func (f downloadTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}

func writeDownloadManifest(t *testing.T, dir string, entries ...Download) string {
	t.Helper()
	data, err := json.Marshal(entries)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "lock.json")
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatal(err)
	}
	return path
}
