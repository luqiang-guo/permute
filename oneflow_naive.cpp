#include <cstdint>
#include <iostream>
#include <vector>
#include <chrono>
#include "of_data_type.h"
#include "nd_index_offset_helper.h"
#include "op_permute.h"
#include <random>


constexpr size_t kMaxMovementSize = 16;
constexpr size_t kMaxNumDims = 8;

template<size_t num_dims, typename IndexType>
struct PermuteKernelParams {
  NdIndexOffsetHelper<IndexType, num_dims> src_index_helper;
  NdIndexOffsetHelper<IndexType, num_dims> dst_index_helper;
  int permutation[num_dims]{};
  IndexType count{};
  const void* src{};
  void* dst{};
};

template<size_t num_dims, typename IndexType>
PermuteKernelParams<num_dims, IndexType> MakePermuteParams(const int64_t* src_dims, const void* src,
                                                           const int* permutation, void* dst,
                                                           size_t count) {
  PermuteKernelParams<num_dims, IndexType> params;
  params.src_index_helper = NdIndexOffsetHelper<IndexType, num_dims>(src_dims);
  int64_t dst_dims[num_dims];
  for (size_t i = 0; i < num_dims; ++i) { dst_dims[i] = src_dims[permutation[i]]; }
  params.dst_index_helper = NdIndexOffsetHelper<IndexType, num_dims>(dst_dims);
  for (size_t i = 0; i < num_dims; ++i) { params.permutation[i] = permutation[i]; }
  params.src = src;
  params.dst = dst;
  params.count = static_cast<IndexType>(count);
  return params;
}


template<size_t num_dims, size_t movement_size, typename IndexType>
void PermuteKernel(PermuteKernelParams<num_dims, IndexType> params) {
  auto t0 = std::chrono::high_resolution_clock::now();   
  using T = typename std::aligned_storage<movement_size, movement_size>::type;
  const T* src = reinterpret_cast<const T*>(params.src);
  T* dst = reinterpret_cast<T*>(params.dst);
  for (IndexType i = 0; i < params.count; ++i) {
    IndexType src_index[num_dims];
    IndexType dst_index[num_dims];
    params.dst_index_helper.OffsetToNdIndex(i, dst_index);
    for (size_t dim = 0; dim < num_dims; ++dim) {
      src_index[params.permutation[dim]] = dst_index[dim];
    }
    IndexType src_offset = params.src_index_helper.NdIndexToOffset(src_index);
    dst[i] = src[src_offset];
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  printf("time = %ld \n", time);
}

template<size_t num_dims, size_t movement_size, typename IndexType>
void LaunchKernel(const int64_t* src_dims, const void* src, const int* permutation,
                  void* dst, size_t count) {
  PermuteKernelParams<num_dims, IndexType> params =
      MakePermuteParams<num_dims, IndexType>(src_dims, src, permutation, dst, count);
  PermuteKernel<num_dims, movement_size, IndexType>(params);
}

# define INT32_MAX		(2147483647)

template<size_t num_dims, size_t movement_size>
void DispatchIndexType(const int64_t* src_dims, const void* src,
                       const int* permutation, void* dst) {
  size_t count = 1;
  for (size_t i = 0; i < num_dims; ++i) { count *= src_dims[i]; }
  if (count < INT32_MAX) {
    LaunchKernel<num_dims, movement_size, int32_t>(src_dims, src, permutation, dst, count);
  } else {
    LaunchKernel<num_dims, movement_size, int64_t>( src_dims, src, permutation, dst, count);
  }
}

template<size_t num_dims>
void DispatchMovementSize(size_t movement_size, const int64_t* src_dims,
                          const void* src, const int* permutation, void* dst) {
  void (*func)(const int64_t* /*src_dims*/, const void* /*src*/,
               const int* /*permutation*/, void* /*dst*/) = nullptr;
  if (movement_size == 1) {
    func = DispatchIndexType<num_dims, 1>;
  } else if (movement_size == 2) {
    func = DispatchIndexType<num_dims, 2>;
  } else if (movement_size == 4) {
    func = DispatchIndexType<num_dims, 4>;
  } else if (movement_size == 8) {
    func = DispatchIndexType<num_dims, 8>;
  } else if (movement_size == 16) {
    func = DispatchIndexType<num_dims, 16>;
  } else {
    // UNIMPLEMENTED();
  }
  func(src_dims, src, permutation, dst);
}


void LaunchWithSimplified( size_t movement_size, size_t num_dims,
                          const int64_t* src_dims, const void* src, const int* permutation,
                          void* dst) {
  void (*func)(size_t /*movement_size*/, const int64_t* /*src_dims*/,
               const void* /*src*/, const int* /*permutation*/, void* /*dst*/) = nullptr;
  if (num_dims == 1) {
    func = DispatchMovementSize<1>;
  } else if (num_dims == 2) {
    func = DispatchMovementSize<2>;
  } else if (num_dims == 3) {
    func = DispatchMovementSize<3>;
  } else if (num_dims == 4) {
    func = DispatchMovementSize<4>;
  } else if (num_dims == 5) {
    func = DispatchMovementSize<5>;
  } else if (num_dims == 6) {
    func = DispatchMovementSize<6>;
  } else if (num_dims == 7) {
    func = DispatchMovementSize<7>;
  } else if (num_dims == 8) {
    func = DispatchMovementSize<8>;
  } else {
    // UNIMPLEMENTED();
  }
  func( movement_size, src_dims, src, permutation, dst);
}


void SimplifyThenLaunch(DataType data_type, size_t num_dims,
                        const int64_t* src_dims, const void* src, const int* permutation,
                        void* dst) {
//   CHECK_LE(num_dims, kMaxNumDims);
//   CHECK_GT(num_dims, 0);
  size_t simplified_num_dims = 0;
  int64_t simplified_src_dims[kMaxNumDims];
  int simplified_permutation[kMaxNumDims];
  size_t movement_size = 0;
  SimplifyPermutation<kMaxNumDims, kMaxMovementSize>(
      num_dims, src_dims, permutation, &simplified_num_dims, simplified_src_dims,
      simplified_permutation, GetSizeOfDataType(data_type), src, dst, &movement_size);
  LaunchWithSimplified(movement_size, simplified_num_dims, simplified_src_dims, src,
                       simplified_permutation, dst);
}

void of_permute(DataType data_type, size_t num_dims, const int64_t* src_dims,
              const void* src, const int* permutation, void* dst) {
    SimplifyThenLaunch(data_type, num_dims, src_dims, src, permutation, dst);
}

#define C 3
#define H 224
#define W 224
#define LEN (C * H * W)
#define TEST_NUM 1000
#define TYEP float


void test_correctness()
{
  TYEP * in = (TYEP *)aligned_alloc(32, 8 * sizeof(TYEP));
  TYEP * out = (TYEP *)aligned_alloc(32, 8 * sizeof(TYEP));
  std::vector<int64_t> src_dims{2, 2, 2};
  std::vector<int32_t> perm{1, 2, 0};
  for(int i = 0; i < 8; i++)
  {
    in[i] = i;
  }
  of_permute(kFloat, 3, src_dims.data(), (void *)in, perm.data(), (void *)out);
  for(int i = 0; i < 8; i++)
  {
    printf("%f ", out[i]);
  }
  printf("\n");
}

void test_speed(TYEP * in, TYEP * out)
{

  std::vector<int64_t> src_dims{H, W, C};
  std::vector<int32_t> perm{1, 2, 0};
  of_permute(kFloat, 3, src_dims.data(), (void *)in, perm.data(), (void *)out);
}


template<typename T>
void alloc_buff(T** src, T** dst)
{
    *src = (T*)malloc(LEN * TEST_NUM * sizeof(T));
    *dst = (T*)malloc(LEN * TEST_NUM * sizeof(T));

    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_real_distribution<float> distribution(-1, 1);
    
    for(int64_t i; i < LEN*TEST_NUM; i++)
    {
        (*src)[i] = distribution(rng);
    }
}

template<typename func>
void  timing(func& f)
{
    double sum=0.0;
    float * src;
    float * dst;
    alloc_buff<float>(&src, &dst);

    for(int i = 0; i < TEST_NUM; i ++)
    {
        auto t1 = std::chrono::high_resolution_clock::now();   
        float * tmp_src = src + LEN*i;
        float * tmp_dst = dst + LEN*i;

        f(tmp_src, tmp_dst);
        auto t2 = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        sum += time;
        printf("i = %d, time = %ld \n", i, time);
    }   

    printf("avg = %lf \n", sum/TEST_NUM);
}



int main(void)
{
  test_correctness();
  timing(test_speed);
}