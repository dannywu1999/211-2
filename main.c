#include <stdio.h>
#include <stdlib.h>
#include <lapacke.h>
#include <time.h>
#include <memory.h>
#include <math.h>
#include <cblas.h>
#include <string.h>

int mydgetrf(double *A, int *PVT, int n)
{
    int i, maxIndex;
    double max;
    double *temprow = (double*) malloc(sizeof(double) * n);
    for (i = 0; i < n; i++)
    {
        // pivoting
        maxIndex = i;
        max = fabs(A[i*n + i]);
        
        int j;
        for (j = i+1; j < n; j++)
        {
            if (fabs(A[j*n + i]) > max)
            {
                maxIndex = j;
                max = fabs(A[j*n + i]);
            }
        }
        if (max == 0)
        {
            printf("LU factorization failed: coefficient matrix is singular.\n");
            return -1;
        }
        else
        {
            if (maxIndex != i)
            {
                // save pivoting information
                int temp = PVT[i];
                PVT[i] = PVT[maxIndex];
                PVT[maxIndex] = temp;
                // swap rows
                memcpy(temprow, A + i*n, n * sizeof(double));
                memcpy(A + i*n, A + maxIndex*n, n * sizeof(double));
                memcpy(A + maxIndex*n, temprow, n * sizeof(double));
            }
        }

        // factorization
        for (j = i+1; j < n; j++)
        {
            A[j*n + i] = A[j*n + i] / A[i*n + i];
            int k;
            for (k = i+1; k < n; k++)
            {
                A[j*n + k] -= A[j*n +i] * A[i*n + k];
            }
        }
    }
    free(temprow);
    return 0;
}

void mydtrsm(char UPLO, double *A, double *B, int n, int *PVT)
{
    double *y = (double*) malloc(n * sizeof(double));
    int i, j;
    double sum;
    if (UPLO == 'L')
    {
        y[0] = B[PVT[0]];
        for (i = 1; i < n; i++)
        {
            sum = 0.0;
            for (j = 0; j < i; j++)
            {
                sum += y[j] * A[i*n + j];
            }
            y[i] = B[PVT[i]] - sum;
        }
    }
    else if (UPLO == 'U')
    {
        y[n - 1] = B[n - 1] / A[(n-1)*n + n-1];
        for (i = n-2; i >= 0; i--)
        {
            sum = 0;
            for (j = i+1; j < n; j++)
            {
                sum += y[j] * A[i*n + j];
            }
            y[i] = (B[i] - sum) / A[i*n + i];
        }
    }

    memcpy(B, y, sizeof(double) * n);
    free(y);
}

void dgemm3_cache_mod(double *a, double *b, double *c, int n, int i, int j, int k, int blocksize)
{
    int i1 = i, j1 = j, k1 = k;
    int ni = i + blocksize > n ? n : i + blocksize;
    int nj = j + blocksize > n ? n : j + blocksize;
    int nk = k + blocksize > n ? n : k + blocksize;

    for (i1 = i; i1 < ni; i1 += 3)
    {
        for (j1 = j; j1 < nj; j1 += 3)
        {
            int t = i1 * n + j1;
            int tt = t + n;
            int ttt = tt + n;
            register double c00 = c[t];
            register double c01 = c[t + 1];
            register double c02 = c[t + 2];
            register double c10 = c[tt];
            register double c11 = c[tt + 1];
            register double c12 = c[tt + 2];
            register double c20 = c[ttt];
            register double c21 = c[ttt + 1];
            register double c22 = c[ttt + 2];

            for (k1 = k; k1 < nk; k1 += 3)
            {
		int l;
                for (l = 0; l < 3; l++)
                {
                    int ta = i1 * n + k1 + l;
                    int tta = ta + n;
                    int ttta = tta + n;
                    int tb = k1 * n + j1 + l * n;
                    register double a0 = a[ta];
                    register double a1 = a[tta];
                    register double a2 = a[ttta];
                    register double b0 = b[tb];
                    register double b1 = b[tb + 1];
                    register double b2 = b[tb + 2];

                    c00 -= a0 * b0;
                    c01 -= a0 * b1;
                    c02 -= a0 * b2;
                    c10 -= a1 * b0;
                    c11 -= a1 * b1;
                    c12 -= a1 * b2;
                    c20 -= a2 * b0;
                    c21 -= a2 * b1;
                    c22 -= a2 * b2;
                }
            }
            c[t] = c00;
            c[t + 1] = c01;
            c[t + 2] = c02;
            c[tt] = c10;
            c[tt + 1] = c11;
            c[tt + 2] = c12;
            c[ttt] = c20;
            c[ttt + 1] = c21;
            c[ttt + 2] = c22;
        }
    }
}

int mydgetrf_block(double *A, int *PVT, int n, int b)
{
    int ib, i, j, k, maxIndex;
    double max, sum;
    double *temprow = (double*) malloc(sizeof(double) * n);

    for (ib = 0; ib < n; ib += b)
    {
        for (i = ib; i < ib+b && i < n; i++)
        {
            // pivoting
            maxIndex = i;
            max = fabs(A[i*n + i]);
            
            int j;
            for (j = i+1; j < n; j++)
            {
                if (fabs(A[j*n + i]) > max)
                {
                    maxIndex = j;
                    max = fabs(A[j*n + i]);
                }
            }
            if (max == 0)
            {
                printf("LU factorization failed: coefficient matrix is singular.\n");
                return -1;
            }
            else
            {
                if (maxIndex != i)
                {
                    // save pivoting information
                    int temp = PVT[i];
                    PVT[i] = PVT[maxIndex];
                    PVT[maxIndex] = temp;
                    // swap rows
                    memcpy(temprow, A + i*n, n * sizeof(double));
                    memcpy(A + i*n, A + maxIndex*n, n * sizeof(double));
                    memcpy(A + maxIndex*n, temprow, n * sizeof(double));
                }
            }

            // factorization
            for (j = i+1; j < n; j++)
            {
                A[j*n + i] = A[j*n + i] / A[i*n + i];
                int k;
                for (k = i+1; k < ib+b && k < n; k++)
                {
                    A[j*n + k] -= A[j*n +i] * A[i*n + k];
                }
            }
        }

        // update A(ib:end, end+1:n)
        for (i = ib; i < ib+b && i < n; i++)
        {
            for (j = ib+b; j < n; j++)
            {
                sum = 0;
                for (k = ib; k < i; k++)
                {
                    sum += A[i*n + k] * A[k*n + j];
                }
                A[i*n + j] -= sum;
            }
        }

        // update A(end+1:n, end+1:n)
        for (i = ib+b; i < n; i += b)
        {
            for (j = ib+b; j < n; j += b)
            {
                dgemm3_cache_mod(A, A, A, n, i, j, ib, b);
            }
        }
    }
    return 0;
}

int checkDiff(double *A, double *B, int n)
{
    double diff = 0.0;
    int i;
    for (i = 0; i < n; i++)
    {
        if (fabs(A[i] - B[i]) > diff)
            diff = fabs(A[i] - B[i]);
    }
    if (diff < 1e-2)
        return 1;
    else 
        return 0;
}

int main(int argc, const char** argv)
{
#ifdef PART_ONE
    int Ns[] = {1000, 2000, 3000, 4000, 5000};
#endif
#ifdef PART_TWO
    int Ns[] = {1008, 2016, 3024, 4032, 5040};
#endif

    int Nsize = 5;

    int i;
    for (i = 0; i < Nsize; i++)
    {
        // Initialization of matrixes
        printf("n=%d...\n", Ns[i]);
        double *A_lapack, *B_lapack;
        double *A_fq, *B_fq;
        double *A_fq_block, *B_fq_block;
        A_lapack = (double*) malloc(sizeof(double) * Ns[i] * Ns[i]);    // col_major
        A_fq = (double*) malloc(sizeof(double) * Ns[i] * Ns[i]);        // row_major
        A_fq_block = (double*) malloc(sizeof(double) * Ns[i] * Ns[i]);  // row_major
        B_lapack = (double*) malloc(sizeof(double) * Ns[i]);
        B_fq = (double*) malloc(sizeof(double) * Ns[i]);
        B_fq_block = (double*) malloc(sizeof(double) * Ns[i]);

        int _i, _j;
        for (_i = 0; _i < Ns[i]; _i++)
        {
            for (_j = 0; _j < Ns[i]; _j++)
            {
                A_lapack[_j*Ns[i] + _i] = A_fq[_i*Ns[i] + _j] = A_fq_block[_i*Ns[i] + _j] = (double) (abs(rand()) % 10000);
            }
            B_lapack[_i] = B_fq[_i] = B_fq_block[_i] = (double) (abs(rand()) % 10000);
        }

        /**********************************
         * LAPACK VERSION
         */
        printf("Lapack calculating...");
        struct timespec begin, end;
        clock_gettime(CLOCK_MONOTONIC, &begin);

        // LU factorization with partial pivoting (LUP)
        int info;
        int lda = Ns[i];
        int ldb = Ns[i];
        int *ipiv = (int*) malloc(sizeof(int) * Ns[i]);
        LAPACKE_dgetrf(LAPACK_COL_MAJOR, Ns[i], Ns[i], A_lapack, lda, ipiv);

        // Apply IPIV array to B
        for (_i = 0; _i < Ns[i]; _i++)
        {
            double t = B_lapack[ipiv[_i] - 1];
            B_lapack[ipiv[_i] - 1] = B_lapack[_i];
            B_lapack[_i] = t;
        }

        // forward substitution
        cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, Ns[i], 1, 1.0, A_lapack, lda, B_lapack, ldb);
        // backward substitution
        cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, Ns[i], 1, 1.0, A_lapack, lda, B_lapack, ldb);

        clock_gettime(CLOCK_MONOTONIC, &end);
        double _time = end.tv_sec - begin.tv_sec + (end.tv_nsec - begin.tv_nsec) / 1e9;
        printf("%lf s\n", _time);

        /**********************************
         * FQ VERSION
         */
        printf("FQ calculating...");
        // fflush(stdout);
        clock_gettime(CLOCK_MONOTONIC, &begin);

        for(_i = 0; _i < Ns[i]; _i++)
            ipiv[_i] = _i;
        _i = mydgetrf(A_fq, ipiv, Ns[i]);
        if (_i != 0)
            return 0;
        mydtrsm('L', A_fq, B_fq, Ns[i], ipiv);
        mydtrsm('U', A_fq, B_fq, Ns[i], ipiv);

        clock_gettime(CLOCK_MONOTONIC, &end);
        _time = end.tv_sec - begin.tv_sec + (end.tv_nsec - begin.tv_nsec) / 1e9;
        printf("%lf s, check...%s.\n", _time, (checkDiff(B_lapack, B_fq, Ns[i]) == 1 ? "passed" : "failed"));

#ifdef PART_TWO
        /**********************************
         * FQ BLOCK VERSION
         */
        printf("FQ BLOCK calculating...");
        // fflush(stdout);
        clock_gettime(CLOCK_MONOTONIC, &begin);

        for(_i = 0; _i < Ns[i]; _i++)
            ipiv[_i] = _i;
        int blocksize = 126;
        _i = mydgetrf_block(A_fq_block, ipiv, Ns[i], blocksize);
        if (_i != 0)
            return 0;
        mydtrsm('L', A_fq_block, B_fq_block, Ns[i], ipiv);
        mydtrsm('U', A_fq_block, B_fq_block, Ns[i], ipiv);

        clock_gettime(CLOCK_MONOTONIC, &end);
        _time = end.tv_sec - begin.tv_sec + (end.tv_nsec - begin.tv_nsec) / 1e9;
        printf("%lf s, check...%s.\n", _time, (checkDiff(B_lapack, B_fq_block, Ns[i]) == 1 ? "passed" : "failed"));
#endif
    }
}
