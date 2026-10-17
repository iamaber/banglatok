package corpus

import (
	"crypto/sha256"
	"encoding/hex"
	"io"
	"os"
)

func FileHash(path string) (string, error) {
	hash, _, err := fileDigest(path)
	return hash, err
}

func fileDigest(path string) (string, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", 0, err
	}
	defer f.Close()
	h := sha256.New()
	n, err := io.Copy(h, f)
	if err != nil {
		return "", 0, err
	}
	return hex.EncodeToString(h.Sum(nil)), n, nil
}
