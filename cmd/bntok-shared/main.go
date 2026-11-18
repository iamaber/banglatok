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
func bntok_encode_into(handle C.ulonglong, data *C.char, length C.int, out *C.uint32_t, capacity C.int) C.longlong {
	if length < 0 || capacity < 0 || (data == nil && length != 0) || (out == nil && capacity != 0) {
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

// bntok_encode_batch encodes count consecutive texts of data, given as byte
// lengths, with up to threads workers (every CPU when threads <= 0), and
// shapes the rows as native.Layout describes. It overwrites lengths with each
// row's unpadded length and returns the number of IDs written, or a negative
// status. IDs go to out when capacity is large enough, otherwise to a malloc'd
// *allocated that the caller frees with bntok_free. Memory is only used during
// the call.
//
//export bntok_encode_batch
func bntok_encode_batch(handle C.ulonglong, data *C.char, length C.int, lengths *C.uint32_t, count C.int, threads C.int,
	maxLength C.longlong, special C.int, width C.longlong, out *C.uint32_t, capacity C.int, allocated **C.uint32_t) C.longlong {
	if length < 0 || count < 0 || capacity < 0 || allocated == nil ||
		(data == nil && length != 0) || (lengths == nil && count != 0) || (out == nil && capacity != 0) {
		return encodeInvalid
	}
	*allocated = nil
	text := unsafe.String((*byte)(unsafe.Pointer(data)), int(length))
	rows := unsafe.Slice((*uint32)(unsafe.Pointer(lengths)), int(count))
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

//export bntok_call
func bntok_call(request *C.char) *C.char {
	return C.CString(string(registry.Call([]byte(C.GoString(request)))))
}

//export bntok_free
func bntok_free(response *C.char) {
	C.free(unsafe.Pointer(response))
}

func main() {}
