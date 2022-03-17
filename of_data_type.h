#ifndef __OF_DATA_TYPE_H__
#define __OF_DATA_TYPE_H__
#include <iostream>

enum DataType : int {
  kInvalidDataType = 0,
  kChar = 1,
  kFloat = 2,
  kDouble = 3,
  kInt8 = 4,
  kInt32 = 5,
  kInt64 = 6,
  kUInt8 = 7,
  kOFRecord = 8,
  kFloat16 = 9,
  kTensorBuffer = 10,
  kBFloat16 = 11,
  kBool = 12,
  kUInt16 = 13,
  kUInt32 = 14,
  kUInt64 = 15,
  kUInt128 = 16,
  kInt16 = 17,
  kInt128 = 18,
  kComplex32 = 19,
  kComplex64 = 20,
  kComplex128 = 21
};


size_t GetSizeOfDataType(DataType data_type) {
  switch (data_type) {
    // 8-bit
    case kChar: return 1;
    case kInt8: return 1;
    case kUInt8: return 1;
    case kBool: return 1;

    // 16-bit
    case kInt16: return 2;
    case kUInt16: return 2;
    case kFloat16: return 2;
    case kBFloat16: return 2;

    // 32-bit
    case kInt32: return 4;
    case kUInt32: return 4;
    case kFloat: return 4;
    case kComplex32: return 4;

    // 64-bit
    case kInt64: return 8;
    case kUInt64: return 8;
    case kDouble: return 8;
    case kComplex64: return 8;

    // 128-bit
    case kInt128: return 16;
    case kUInt128: return 16;
    case kComplex128: return 16;


    default: return 0;
  }
}


#endif