#ifndef MATH_VEC_MATH_H_
#define MATH_VEC_MATH_H_

#include <memory>

namespace vec {

/*
  以下矩阵乘法，对于输出使用的是累积，
  以便支持累积梯度更新，所以，当不需要累积的时候，需要对输出先清零
*/

// 标准矩阵乘法 C = A × B
// A的形状[M, K]，B的形状[K, N]，C的形状[M, N]
// bias的形状是[N]，如果有偏置则添加到输出中
void matmul_native(const float *A, size_t A_stride, const float *B,
                   size_t B_stride, const float *bias, float *C,
                   size_t C_stride, size_t M, size_t K, size_t N);
void matmul_blas(const float *A, size_t A_stride, const float *B,
                 size_t B_stride, const float *bias, float *C, size_t C_stride,
                 size_t M, size_t K, size_t N);

// A转置矩阵乘法 C = A^T × B
// A的形状[K,M],B的形状[K,N],C的形状[M,N]
// bias的形状是[N]，如果有偏置则添加到输出中
void matmul_A_transpose_native(const float *A, size_t A_stride, const float *B,
                               size_t B_stride, const float *bias, float *C,
                               size_t C_stride, size_t K, size_t M, size_t N);
void matmul_A_transpose_blas(const float *A, size_t A_stride, const float *B,
                             size_t B_stride, const float *bias, float *C,
                             size_t C_stride, size_t K, size_t M, size_t N);

// B转置矩阵乘法 C = A × B^T
// A的形状[M,K],B的形状[N,K],C的形状[M,N]
// bias的形状是[N]，如果有偏置则添加到输出中
void matmul_B_transpose_native(const float *A, size_t A_stride, const float *B,
                               size_t B_stride, const float *bias, float *C,
                               size_t C_stride, size_t M, size_t N, size_t K);
void matmul_B_transpose_blas(const float *A, size_t A_stride, const float *B,
                             size_t B_stride, const float *bias, float *C,
                             size_t C_stride, size_t M, size_t N, size_t K);

// A和B都转置的矩阵乘法 C = A^T × B^T
// A的形状是[K,M],B的形状是[N,K],C的形状是[M,N]
// bias的形状是[N]，如果有偏置则添加到输出中
void matmul_A_B_transpose_native(const float *A, size_t A_stride,
                                 const float *B, size_t B_stride,
                                 const float *bias, float *C, size_t C_stride,
                                 size_t K, size_t M, size_t N);
void matmul_A_B_transpose_blas(const float *A, size_t A_stride, const float *B,
                               size_t B_stride, const float *bias, float *C,
                               size_t C_stride, size_t K, size_t M, size_t N);

// 二维张量计算softmax
void mat_softmax_forward_native(float *A, size_t M, size_t N);
void mat_softmax_forward_blas(float *A, size_t M, size_t N);

// dout:是输出梯度
// inp: 是 mat_softmax_forward 前向传播输出
// din: 是输入梯度
void mat_softmax_backward_native(float *dout, const float *inp,
                                 const float *din, size_t M, size_t N);
void mat_softmax_backward_blas(float *dout, const float *inp, const float *din,
                               size_t M, size_t N);

// y = alpha*x + y
void saxpy_native(int N, float alpha, const float *x, int incx, float *y,
                  int incy);
void saxpy_blas(int N, float alpha, const float *x, int incx, float *y,
                int incy);

// y = alpha*x + beta*y
void saxpby_native(int N, float alpha, const float *x, int incx, float beta,
                   float *y, int incy);

void saxpby_blas(int N, float alpha, const float *x, int incx, float beta,
                 float *y, int incy);

// y←α⋅A⋅x+β⋅y, 矩阵和向量乘法
void sgemv_native(int M, int N, float alpha, const float *a, int lda,
                  const float *x, int incx, float beta, float *y, int incy);
void sgemv_blas(int M, int N, float alpha, const float *a, int lda,
                const float *x, int incx, float beta, float *y, int incy);

// y←α⋅A^T⋅x+β⋅y 转置矩阵和向量乘法
void sgemv_transpose_native(int M, int N, float alpha, const float *a, int lda,
                            const float *x, int incx, float beta, float *y,
                            int incy);
void sgemv_transpose_blas(int M, int N, float alpha, const float *a, int lda,
                          const float *x, int incx, float beta, float *y,
                          int incy);

// y = x
void scopy_native(int n, const float *x, int incx, float *y, int incy);
void scopy_blas(int n, const float *x, int incx, float *y, int incy);

// X = alpha * X
void sscal_native(int N, float alpha, float *X, int incX);
void sscal_blas(int N, float alpha, float *X, int incX);

// 内积，输出一个浮点数
float sdot_native(int n, const float *x, int incx, const float *y, int incy);
float sdot_blas(int n, const float *x, int incx, const float *y, int incy);

void ssqrt_native(int n, const float *x, float *y);
void ssqrt_blas(int n, const float *x, float *y);

void sexp_native(int n, const float *x, float *y);
void sexp_blas(int n, const float *x, float *y);

// 两个向量哈达玛积，就是逐位相乘
void shdm_native(int n, const float *a, const float *b, float *result);
void shdm_blas(int n, const float *a, const float *b, float *result);

void transpose_2d_blas(int m, int n, const float *A, float *B);

} // namespace vec

#endif // MATH_VEC_MATH_H_