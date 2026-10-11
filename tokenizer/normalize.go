// Package tokenizer implements a joint Unicode Unigram tokenizer in pure Go.
package tokenizer

import (
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

const Normalization = "nfc-whitespace-v1"

// Normalize applies NFC, trims whitespace and collapses Unicode whitespace.
// Case, joiners, punctuation, accents and spelling variants are preserved.
// Invalid UTF-8 is replaced with U+FFFD before normalization.
func Normalize(s string) string {
	if alreadyNormalized(s) {
		return s
	}
	s = norm.NFC.String(strings.ToValidUTF8(s, "\uFFFD"))
	previousSpace := true
	for _, r := range s {
		if unicode.IsSpace(r) {
			if r != ' ' || previousSpace {
				return strings.Join(strings.Fields(s), " ")
			}
			previousSpace = true
		} else {
			previousSpace = false
		}
	}
	return strings.TrimSuffix(s, " ")
}

// Per-rune NFC facts for the scripts this tokenizer targets. A rune is listed
// only if NFC leaves it unchanged on its own.
const (
	nfcListed = 1 << iota
	nfcBoundaryBefore
	nfcBoundaryAfter
	// Combining class zero without a boundary before (for example Bengali
	// U+09BE): it can only compose with the immediately preceding rune.
	nfcStarterMaybe
)

// fastRanges are ASCII, Bengali, danda, joiners, punctuation, symbols and
// emoji. ASCII whitespace other than U+0020 is left unlisted; none of the
// other ranges contains Unicode whitespace.
var fastRanges = [][2]rune{
	{0x00, 0x7f}, {0x0964, 0x0965}, {0x0980, 0x09ff}, {0x200c, 0x200d}, {0x2010, 0x2027},
	{0x2030, 0x205e}, {0x20a0, 0x20c0}, {0x2100, 0x27bf}, {0xfe00, 0xfe0f}, {0x1f000, 0x1faff},
}

type nfcSegment struct {
	first rune
	flags []uint8
}

// nfcLowLimit bounds the directly indexed table; rarer ranges use segments.
const nfcLowLimit = 0x2800

var (
	nfcLow      [nfcLowLimit]uint8
	nfcSegments []nfcSegment
	// composingPairs holds every listed pair whose second rune is a starter
	// that composes with the first, such as U+09C7 U+09BE.
	composingPairs [][2]rune
)

func init() {
	var maybe []rune
	for _, span := range fastRanges {
		flags := nfcLow[:]
		first := rune(0)
		if span[0] >= nfcLowLimit {
			first, flags = span[0], make([]uint8, span[1]-span[0]+1)
			nfcSegments = append(nfcSegments, nfcSegment{first, flags})
		}
		for r := span[0]; r <= span[1]; r++ {
			text := string(r)
			if norm.NFC.String(text) != text || (r < utf8.RuneSelf && unicode.IsSpace(r) && r != ' ') {
				continue
			}
			p := norm.NFC.PropertiesString(text)
			value := uint8(nfcListed)
			if p.BoundaryBefore() {
				value |= nfcBoundaryBefore
			} else if p.CCC() == 0 {
				value |= nfcStarterMaybe
				maybe = append(maybe, r)
			}
			if p.BoundaryAfter() {
				value |= nfcBoundaryAfter
			}
			flags[r-first] = value
		}
	}
	for _, span := range fastRanges {
		for a := span[0]; a <= span[1]; a++ {
			for _, b := range maybe {
				if pair := string([]rune{a, b}); nfcFlags(a) != 0 && norm.NFC.String(pair) != pair {
					composingPairs = append(composingPairs, [2]rune{a, b})
				}
			}
		}
	}
}

func nfcFlags(r rune) uint8 {
	if r < nfcLowLimit {
		return nfcLow[r]
	}
	for _, segment := range nfcSegments {
		if offset := r - segment.first; offset >= 0 && int(offset) < len(segment.flags) {
			return segment.flags[offset]
		}
	}
	return 0
}

func composes(a, b rune) bool {
	for _, pair := range composingPairs {
		if pair == [2]rune{a, b} {
			return true
		}
	}
	return false
}

// alreadyNormalized proves, without allocating, that Normalize(s) == s for
// common Bangla/English/Banglish text. False means "unknown", not "changes".
// NFC cannot change a string of listed runes when every adjacent pair has a
// normalization boundary between them. Marks with nonzero combining classes
// have no boundary on either side, so any pair that could reorder fails. A
// class-zero rune without a boundary before can only compose with the rune
// immediately before it, which composingPairs answers exactly.
func alreadyNormalized(s string) bool {
	previous, previousFlags := rune(-1), uint8(nfcBoundaryAfter)
	for i := 0; i < len(s); {
		r, width := rune(s[i]), 1
		if r >= utf8.RuneSelf {
			r, width = utf8.DecodeRuneInString(s[i:])
		}
		i += width
		flags := nfcFlags(r)
		if flags == 0 || (r == ' ' && (previous == ' ' || previous < 0)) {
			return false
		}
		if previousFlags&nfcBoundaryAfter == 0 && flags&nfcBoundaryBefore == 0 &&
			(flags&nfcStarterMaybe == 0 || composes(previous, r)) {
			return false
		}
		previous, previousFlags = r, flags
	}
	return previous != ' '
}
