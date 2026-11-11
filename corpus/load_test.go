package corpus

import (
	"archive/zip"
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/parquet-go/parquet-go"
)

// TestLoadSourcesFixtures exercises the network-fed ingest and its
// column-to-bucket mapping with local archives.
func TestLoadSourcesFixtures(t *testing.T) {
	raw := t.TempDir()
	for _, dir := range []string{"banglatlit", "tatoeba", "wikipedia"} {
		if err := os.MkdirAll(filepath.Join(raw, dir), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	header := "id,text_transliterated,text_bengali\n"
	files := map[string]string{
		"train.csv":      header + "1,ami bhalo achi,আমি ভালো আছি\n2,hello বাংলা সবাই,শুধু বাংলা লেখা\n",
		"validation.csv": header,
		"test.csv":       header,
	}
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(raw, "banglatlit", name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	writeZip(t, filepath.Join(raw, "tatoeba", "ben-eng.zip"), "ben.txt",
		"Hello\tহ্যালো\tattr one\nThank you\tধন্যবাদ\tattr two\n")
	writeZip(t, filepath.Join(raw, "tatoeba", "fra-eng.zip"), "fra.txt",
		"Hello\tBonjour\tattr one\n")

	id := sampledID(t)
	for shard := 0; shard < 2; shard++ {
		text := "একটি বাংলা অনুচ্ছেদ যেখানে আটটির বেশি শব্দ রয়েছে"
		writeParquet(t, filepath.Join(raw, "wikipedia", fmt.Sprintf("bn-%d.parquet", shard)), []wikiRow{
			{ID: id, URL: "https://example.org/wiki/" + strconv.Itoa(shard), Title: "fixture", Text: text},
		})
	}

	records, err := LoadSources(raw)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []Record{
		{Text: "ami bhalo achi", Bucket: "banglish", Source: "banglatlit"},
		{Text: "আমি ভালো আছি", Bucket: "bangla", Source: "banglatlit"},
		{Text: "hello বাংলা সবাই", Bucket: "code-mixed", Source: "banglatlit"},
		{Text: "শুধু বাংলা লেখা", Bucket: "bangla", Source: "banglatlit"},
		{Text: "Hello", Bucket: "english", Source: "tatoeba-ben"},
		{Text: "হ্যালো", Bucket: "bangla", Source: "tatoeba-ben"},
		{Text: "Hello", Bucket: "english", Source: "tatoeba-fra"},
		{Bucket: "bangla", Source: "wikipedia-bn-20231101"},
	} {
		if !hasRecord(records, want) {
			t.Errorf("missing record %+v", want)
		}
	}
	// Column 0 is the English side of both Tatoeba archives; the French text
	// in column 1 must never be ingested.
	if hasRecord(records, Record{Text: "Bonjour"}) {
		t.Error("French column was ingested")
	}
	// Two BanglaTLit rows (two buckets each), two Bengali pairs (English and
	// Bengali), one French archive (English only), and one article per shard.
	if got := len(records); got != 11 {
		t.Fatalf("got %d records, want 11", got)
	}
}

func hasRecord(records []Record, want Record) bool {
	for _, r := range records {
		if (want.Text == "" || r.Text == want.Text) &&
			(want.Bucket == "" || r.Bucket == want.Bucket) &&
			(want.Source == "" || r.Source == want.Source) {
			return true
		}
	}
	return false
}

func writeZip(t *testing.T, path, name, body string) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	w := zip.NewWriter(f)
	entry, err := w.Create(name)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := entry.Write([]byte(body)); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}

func writeParquet(t *testing.T, path string, rows []wikiRow) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	w := parquet.NewGenericWriter[wikiRow](f)
	if _, err := w.Write(rows); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}

// sampledID returns an article ID that passes LoadSources' deterministic
// 20% sample (sha256(ID)[0] % 5 == 0).
func sampledID(t *testing.T) string {
	t.Helper()
	for i := 0; ; i++ {
		id := strconv.Itoa(i)
		h := sha256.Sum256([]byte(id))
		if h[0]%5 == 0 {
			return id
		}
	}
}
