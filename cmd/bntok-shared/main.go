// Package main exposes the Go tokenizer through owned C buffers.
package main

/*
#include <stdint.h>
#include <stdlib.h>
*/
import "C"

import (
	"math"
	"unsafe"

	"github.com/iamaber/banglatok/internal/native"
)

var registry native.Registry

// Status codes returned by bntok_encode_into in place of an ID count.
const (
	encodeClosed  = -1
	encodeInvalid = -2
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
