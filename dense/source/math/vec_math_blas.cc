#include "math/vec_math.h"
#include <cblas.h>
#include <cmath>
#include <vector>

namespace vec {

// A的形状[M, K]，B的形状[K, N]，C的形状[M, N]
void matmul_blas(const float *A, size_t A_stride, const float *B,
                 size_t B_stride, const float *bias, float *C, size_t C_stride,
                 size_t M, size_t K, size_t N) {
  // C += A * B
  // 将 beta 设置为 1.0f，表示将乘法结果累加到 C 的原有值上
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A,
              A_stride, B, B_stride,
              1.0f, // 累加到 C 的原有值
              C, C_stride);

  // 加上 bias
  if (bias) {
    for (size_t m = 0; m < M; ++m) {
      cblas_saxpy(N, 1.0f, bias, 1, &C[m * C_stride], 1);
    }
  }
}

// A的形状[K,M],B的形状[K,N],C的形状[M,N]
void matmul_A_transpose_blas(const float *A, size_t A_stride, const float *B,
                             size_t B_stride, const float *bias, float *C,
                             size_t C_stride, size_t K, size_t M, size_t N) {
  // C += A_trans * B
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, 1.0f, A,
              A_stride, B, B_stride,
              1.0f, // 累加到 C 的原有值
              C, C_stride);

  // 加上 bias
  if (bias) {
    for (size_t m = 0; m < M; ++m) {
      cblas_saxpy(N, 1.0f, bias, 1, &C[m * C_stride], 1);
    }
  }
}

// A的形状[M,K],B的形状[N,K],C的形状[M,N]
void matmul_B_transpose_blas(const float *A, size_t A_stride, const float *B,
                             size_t B_stride, const float *bias, float *C,
                             size_t C_stride, size_t M, size_t N, size_t K) {
  // C += A * B_trans
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, A,
              A_stride, B, B_stride,
              1.0f, // 累加到 C 的原有值
              C, C_stride);

  // 加上 bias
  if (bias) {
    for (size_t m = 0; m < M; ++m) {
      cblas_saxpy(N, 1.0f, bias, 1, &C[m * C_stride], 1);
    }
  }
}

// A的形状是[K,M],B的形状是[N,K],C的形状是[M,N]
void matmul_A_B_transpose_blas(const float *A, size_t A_stride, const float *B,
                               size_t B_stride, const float *bias, float *C,
                               size_t C_stride, size_t K, size_t M, size_t N) {
  // C += A_trans * B_trans
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans, M, N, K, 1.0f, A, A_stride,
              B, B_stride,
              1.0f, // 累加到 C 的原有值
              C, C_stride);

  // 加上 bias
  if (bias) {
    for (size_t m = 0; m < M; ++m) {
      cblas_saxpy(N, 1.0f, bias, 1, &C[m * C_stride], 1);
    }
  }
}

void saxpy_blas(int N, float alpha, const float *x, int incx, float *y,
                int incy) {
  cblas_saxpy(N, alpha, x, incx, y, incy);
}

void saxpby_blas(int N, float alpha, const float *x, int incx, float beta,
                 float *y, int incy) {
  cblas_saxpby(N, alpha, x, incx, beta, y, incy);
}

void sgemv_blas(int M, int N, float alpha, const float *a, int lda,
                const float *x, int incx, float beta, float *y, int incy) {
  cblas_sgemv(CblasRowMajor, CblasNoTrans, M, N, alpha, a, lda, x, incx, beta,
              y, incy);
}

void sgemv_transpose_blas(int M, int N, float alpha, const float *a, int lda,
                          const float *x, int incx, float beta, float *y,
                          int incy) {
  cblas_sgemv(CblasRowMajor, CblasTrans, M, N, alpha, a, lda, x, incx, beta, y,
              incy);
}

void scopy_blas(int n, const float *x, int incx, float *y, int incy) {
  cblas_scopy(n, x, incx, y, incy);
}

// X = alpha * X
void sscal_blas(int N, float alpha, float *X, int incX) {
  cblas_sscal(N, alpha, X, incX);
}

float sdot_blas(int n, const float *x, int incx, const float *y, int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}

void ssqrt_blas(int n, const float *x, float *y) {
  const int vector_limit = n >> 3;
  int j = 0;
  for (; j < vector_limit * 8; j += 8) {
    __m256 g = _mm256_loadu_ps(&x[j]);
    g = _mm256_sqrt_ps(g);
    _mm256_storeu_ps(&y[j], g);
  }
  for (; j < n; ++j) {
    y[j] = sqrtf(x[j]);
  }
}

void sexp_blas(int n, const float *x, float *y) {
  const int vector_limit = n >> 3; // n / 8
  int j = 0;

  // AVX2 向量化处理
  for (; j < vector_limit * 8; j += 8) {
    __m256 g = _mm256_loadu_ps(&x[j]);
    g = _mm256_exp_ps(g);
    _mm256_storeu_ps(&y[j], g);
  }

  // 处理剩余元素
  for (; j < n; ++j) {
    y[j] = expf(x[j]);
  }
}

void shdm_blas(int n, const float *a, const float *b, float *result) {
  const int vector_limit = n >> 3; // n / 8
  int i = 0;

  // AVX2 向量化处理
  for (; i < vector_limit * 8; i += 8) {
    __m256 a_vec = _mm256_loadu_ps(&a[i]);
    __m256 b_vec = _mm256_loadu_ps(&b[i]);
    __m256 result_vec = _mm256_mul_ps(a_vec, b_vec);
    _mm256_storeu_ps(&result[i], result_vec);
  }

  // 处理剩余元素
  for (; i < n; ++i) {
    result[i] = a[i] * b[i];
  }
}

void mat_softmax_forward_blas(float *A, size_t M, size_t N) {
  std::vector<float> ones(N, 1.0f);

  for (size_t m = 0; m < M; ++m) {
    auto A_bt = A + m * N;
    float maxval = -INFINITY;
    for (size_t n = 0; n < N; ++n) {
      if (A_bt[n] > maxval) {
        maxval = A_bt[n];
      }
    }
    // saxpy: y = y + (-alpha) * ones
    // 实现： A_bt[n] - maxval
    saxpy_blas(N, -maxval, ones.data(), 1, A_bt, 1);
    // std::exp
    sexp_blas(N, A_bt, A_bt);
    // 计算 sum
    float sum = sdot_blas(N, A_bt, 1, ones.data(), 1);
    sscal_blas(N, 1.0f / sum, A_bt, 1);
  }
}

void mat_softmax_backward_blas(float *dout, const float *inp, const float *din,
                               size_t M, size_t N) {
  std::vector<float> ones(N, 1.0f);
  std::vector<float> temp(N, 0.0f);

  for (size_t m = 0; m < M; ++m) {
    auto inp_bt = inp + m * N;
    auto din_bt = din + m * N;
    auto dout_bt = dout + m * N;

    float sum = sdot_blas(N, din_bt, 1, inp_bt, 1);
    scopy_blas(N, din_bt, 1, temp.data(), 1);

    // din_bt[n] - sum
    saxpy_blas(N, -sum, ones.data(), 1, temp.data(), 1);
    shdm_blas(N, inp_bt, temp.data(), dout_bt);
  }
}

void transpose_2d_blas(int m, int n, const float *A, float *B) {
  cblas_somatcopy(CblasRowMajor, CblasTrans, m, n, 1.0f, A, n, B, m);
}

} // namespace vec