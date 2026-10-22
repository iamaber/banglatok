// Package main exposes the Go tokenizer through owned C buffers.
package main

/*
#include <stdint.h>
#include <stdlib.h>

// Unlike C.malloc, failure returns NULL instead of crashing the process.
static uint32_t *bntok_alloc_ids(size_t n) { return malloc(n * sizeof(uint32_t)); }
*/
import "C"

import (
	"errors"
	"math"
	"unsafe"

	"github.com/iamaber/banglatok/internal/native"
)

var registry native.Registry

// Status codes returned by bntok_encode_into in place of an ID count.
// bntok_encode_batch also returns encodeLengths and encodeTooLarge.
const (
	encodeClosed   = -1
	encodeInvalid  = -2
	encodeLengths  = -3
	encodeTooLarge = -4
)

// bntok_encode_into encodes one UTF-8 text into a caller-owned uint32 buffer.
// It returns the ID count; a count above capacity means retry with that size.
// Input and output memory are only used during the call.
//
//export bntok_encode_into
func bntok_encode_into(handle C.ulonglong, data *C.char, length C.size_t, out *C.uint32_t, capacity C.size_t) C.longlong {
	if uint64(length) > math.MaxInt32 || uint64(capacity) > math.MaxInt32 || (data == nil && length != 0) || (out == nil && capacity != 0) {
		return encodeInvalid
	}
	text := unsafe.String((*byte)(unsafe.Pointer(data)), int(length))
	ids := unsafe.Slice((*uint32)(unsafe.Pointer(out)), int(capacity))
	n, err := registry.EncodeInto(uint64(handle), text, ids)
	if err != nil {
		return encodeClosed
	}
	return C.longlong(n)
}

// bntok_encode_batch encodes len(runes) consecutive texts of data, given as
// code-point counts, with up to threads workers (<= 0: every CPU) and shapes
// the rows as native.Layout describes. It overwrites runes with each row's
// unpadded length and returns the number of IDs written, or a negative status.
// IDs go to out when capacity suffices; otherwise to a malloc'd *allocated
// that the caller frees with bntok_free. Memory is only used during the call.
//
//export bntok_encode_batch
func bntok_encode_batch(handle C.ulonglong, data *C.char, length C.size_t, runes *C.uint32_t, count C.size_t, threads C.int,
	maxLength C.longlong, special C.int, width C.longlong, out *C.uint32_t, capacity C.size_t, allocated **C.uint32_t) C.longlong {
	if uint64(length) > math.MaxInt32 || uint64(count) > math.MaxInt32 || uint64(capacity) > math.MaxInt32 || allocated == nil ||
		(data == nil && length != 0) || (runes == nil && count != 0) || (out == nil && capacity != 0) {
		return encodeInvalid
	}
	*allocated = nil
	text := unsafe.String((*byte)(unsafe.Pointer(data)), int(length))
	rows := unsafe.Slice((*uint32)(unsafe.Pointer(runes)), int(count))
	layout := native.Layout{MaxLength: int(maxLength), Special: special != 0, Width: int(width)}
	batch, err := registry.EncodeBatch(uint64(handle), text, rows, int(threads), layout)
	switch {
	case errors.Is(err, native.ErrBatchLengths):
		return encodeLengths
	case errors.Is(err, native.ErrLayout):
		return encodeInvalid
	case errors.Is(err, native.ErrBatchSize):
		return encodeTooLarge
	case err != nil:
		return encodeClosed
	}
	size := batch.Size()
	if size > int(capacity) {
		out = C.bntok_alloc_ids(C.size_t(size))
		if out == nil {
			batch.Release()
			return encodeTooLarge
		}
		*allocated = out
	}
	batch.Write(unsafe.Slice((*uint32)(unsafe.Pointer(out)), size), rows)
	return C.longlong(size)
}

// bntok_encode encodes a length-prefixed batch with up to threads workers;
// threads <= 0 uses every available CPU.
//
//export bntok_encode
func bntok_encode(handle C.ulonglong, data *C.char, length C.size_t, threads C.int, outputLength *C.size_t) *C.char {
	if outputLength == nil {
		return nil
	}
	var response []byte
	if uint64(length) > math.MaxInt32 || (data == nil && length != 0) {
		response = native.EncodeError("invalid encode input size")
	} else {
		response = registry.EncodeBinary(uint64(handle), unsafe.Slice((*byte)(unsafe.Pointer(data)), int(length)), int(threads))
	}
	*outputLength = C.size_t(len(response))
	return (*C.char)(C.CBytes(response))
}

//export bntok_call
func bntok_call(request *C.char) *C.char {
	return C.CString(string(registry.Call([]byte(C.GoString(request)))))
}

//export bntok_free
func bntok_free(response *C.char) {
	C.free(unsafe.Pointer(response))
}

func main() {}
