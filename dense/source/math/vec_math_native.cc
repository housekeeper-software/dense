#include "math/vec_math.h"
#include <cmath>

namespace vec {

// A的形状[M, K]，B的形状[K, N]，C的形状[M, N]
void matmul_native(const float *A, size_t A_stride, const float *B,
                   size_t B_stride, const float *bias, float *C,
                   size_t C_stride, size_t M, size_t K, size_t N) {
  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      float sum = bias ? bias[n] : 0.0f;
      for (size_t k = 0; k < K; ++k) {
        sum += A[m * A_stride + k] * B[k * B_stride + n];
      }
      C[m * C_stride + n] += sum;
    }
  }
}

// A的形状[K,M],B的形状[K,N],C的形状[M,N]
void matmul_A_transpose_native(const float *A, size_t A_stride, const float *B,
                               size_t B_stride, const float *bias, float *C,
                               size_t C_stride, size_t K, size_t M, size_t N) {
  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      float sum = bias ? bias[n] : 0.0f;
      for (size_t k = 0; k < K; ++k) {
        sum += A[k * A_stride + m] * B[k * B_stride + n];
      }
      C[m * C_stride + n] += sum;
    }
  }
}

// A的形状[M,K],B的形状[N,K],C的形状[M,N]
void matmul_B_transpose_native(const float *A, size_t A_stride, const float *B,
                               size_t B_stride, const float *bias, float *C,
                               size_t C_stride, size_t M, size_t N, size_t K) {
  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      float sum = bias ? bias[n] : 0.0f;
      for (size_t k = 0; k < K; ++k) {
        sum += A[m * A_stride + k] * B[n * B_stride + k];
      }
      C[m * C_stride + n] += sum;
    }
  }
}

// A的形状是[K,M],B的形状是[N,K],C的形状是[M,N]
void matmul_A_B_transpose_native(const float *A, size_t A_stride,
                                 const float *B, size_t B_stride,
                                 const float *bias, float *C, size_t C_stride,
                                 size_t K, size_t M, size_t N) {
  // m-n-k 循环顺序：直接计算每个C[m,n]的完整值
  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      float sum = bias ? bias[n] : 0.0f;
      for (size_t k = 0; k < K; ++k) {
        sum += A[k * A_stride + m] * B[n * B_stride + k];
      }
      C[m * C_stride + n] += sum;
    }
  }
}

void saxpy_native(int N, float alpha, const float *x, int incx, float *y,
                  int incy) {
  int i, x_i = 0, y_i = 0;
  if (incx == 1 && incy == 1) {
    for (i = 0; i < N; ++i) {
      y[i] += alpha * x[i];
    }
  } else {
    for (i = 0; i < N; ++i) {
      y[y_i] += alpha * x[x_i];
      x_i += incx;
      y_i += incy;
    }
  }
}

void saxpby_native(int N, float alpha, const float *x, int incx, float beta,
                   float *y, int incy) {
  int i, x_i = 0, y_i = 0;
  if (incx == 1 && incy == 1) {
    for (i = 0; i < N; ++i) {
      y[i] += (alpha * x[i] + beta * y[i]);
    }
  } else {
    for (i = 0; i < N; ++i) {
      y[y_i] += (alpha * x[x_i] + beta * y[y_i]);
      x_i += incx;
      y_i += incy;
    }
  }
}

void sgemv_native(int M, int N, float alpha, const float *a, int lda,
                  const float *x, int incx, float beta, float *y, int incy) {
  for (int m = 0; m < M; ++m) {
    float sum = 0.0f;
    for (int n = 0; n < N; ++n) {
      sum += (alpha * a[m * lda + n] * x[n * incx]);
    }
    y[m * incy] = sum + beta * y[m * incy];
  }
}

void sgemv_transpose_native(int M, int N, float alpha, const float *a, int lda,
                            const float *x, int incx, float beta, float *y,
                            int incy) {
  for (int n = 0; n < N; ++n) {
    float sum = 0.0f;
    for (int m = 0; m < M; ++m) {
      sum += (alpha * a[m * lda + n] * x[m * incx]);
    }
    y[n * incy] = sum + beta * y[n * incy];
  }
}

void scopy_native(int n, const float *x, int incx, float *y, int incy) {
  for (int i = 0; i < n; ++i) {
    y[i * incy] = x[i * incx];
  }
}

void sscal_native(int N, float alpha, float *X, int incX) {
  for (int i = 0; i < N; ++i) {
    X[i * incX] *= alpha;
  }
}

float sdot_native(int n, const float *x, int incx, const float *y, int incy) {
  float sum = 0.0f;
  for (int i = 0; i < n; ++i) {
    sum += x[i * incx] * y[i * incy];
  }
  return sum;
}

void ssqrt_native(int n, float *x, float *y) {
  for (int i = 0; i < n; ++i) {
    y[i] = std::sqrtf(x[i]);
  }
}

void sexp_native(int n, const float *x, float *y) {
  for (int i = 0; i < n; ++i) {
    y[i] = std::exp(x[i]);
  }
}

void shdm_native(int n, const float *a, const float *b, float *result) {
  for (int i = 0; i < n; ++i) {
    result[i] = a[i] * b[i];
  }
}

void mat_softmax_forward_native(float *A, size_t M, size_t N) {
  for (size_t m = 0; m < M; ++m) {
    auto A_bt = A + m * N;
    float maxval = -INFINITY;
    for (size_t n = 0; n < N; ++n) {
      if (A_bt[n] > maxval) {
        maxval = A_bt[n];
      }
    }

    float sum = 0.0f;
    for (size_t n = 0; n < N; ++n) {
      A_bt[n] = std::exp(A_bt[n] - maxval);
      sum += A_bt[n];
    }
    for (size_t n = 0; n < N; ++n) {
      A_bt[n] /= sum;
    }
  }
}

void mat_softmax_backward_native(float *dout, const float *inp,
                                 const float *din, size_t M, size_t N) {
  for (size_t m = 0; m < M; ++m) {
    auto inp_bt = inp + m * N;
    auto din_bt = din + m * N;
    auto dout_bt = dout + m * N;
    float sum = 0.0f;
    for (size_t n = 0; n < N; ++n) {
      sum += (din_bt[n] * inp_bt[n]);
    }
    for (size_t n = 0; n < N; ++n) {
      dout_bt[n] = (inp_bt[n] * (din_bt[n] - sum));
    }
  }
}

} // namespace vec