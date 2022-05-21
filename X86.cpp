#include <iostream>
#include<omp.h>
#include <stdio.h>
#include <stdlib.h>
#include<stdio.h>
#include <windows.h>
#include<time.h>
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2

using namespace std;
int n = 50;
int NUM_THREADS = 6;
float** A;
float** B;
float** C;
float** D;
float** E;
void m_reset(int n)
{

    A = new float* [n];
    for (int i = 0; i < n; i++)
        A[i] = new float[n];
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < i; j++)
            A[i][j] = 0;
        A[i][i] = 1.0;
        for (int j = i + 1; j < n; j++)
        {
            A[i][j] = rand();
        }

    }
    for (int k = 0; k < n; k++)
    {
        for (int i = k + 1; i < n; i++)
        {
            for (int j = 0; j < n; j++)
                A[i][j] += A[k][j];
        }
    }
}
void deepcopy()
{
    B = new float* [n];
    for (int i = 0; i < n; i++)
        B[i] = new float[n];
    C = new float* [n];
    for (int i = 0; i < n; i++)
        C[i] = new float[n];
    D = new float* [n];
    for (int i = 0; i < n; i++)
        D[i] = new float[n];
    E = new float* [n];
    for (int i = 0; i < n; i++)
        E[i] = new float[n];
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            B[i][j] = A[i][j];
            C[i][j] = A[i][j];
            D[i][j] = A[i][j];
            E[i][j] = A[i][j];
        }
    }
}
void chuanxing()
{
    long long head, tail, freq; // timers
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for (int k = 0; k < n; k++)
    {
        for (int j = k + 1; j < n; j++)
        {
            C[k][j] /= C[k][k];
        }
        C[k][k] = 1.0;
        for (int i = k + 1; i < n; i++)
        {
            for (int j = k + 1; j < n; j++)
            {
                C[i][j] -= C[i][k] * C[k][j];
            }
            C[i][k] = 0;
        }

    }
    /*	 cout<<"串行"<<endl;
        for(int i=0;i<n;i++)
        {
            for(int j=0;j<n;j++)
            {
                cout<<C[i][j]<<" ";
                if(j==n-1)
                    cout<<endl;
            }
        }*/
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << (tail - head) * 1000.0 / freq << "          ";

}
void omp()
{
    long long head, tail, freq; // timers
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    omp_set_num_threads(NUM_THREADS);
#pragma omp parallel
    {
        for (int k = 0; k < n; ++k) {
            // 串行部分，也可以尝试并行化
#pragma omp single
            {
                float tmp = A[k][k];
                for (int j = k + 1; j < n; ++j)
                {
                    A[k][j] = A[k][j] / tmp;
                }
                A[k][k] = 1.0;
            }
            // 并行部分，使用行划分
#pragma omp for schedule(static,NUM_THREADS)
            for (int i = k + 1; i < n; ++i)
            {
                float tmp = A[i][k];
                for (int j = k + 1; j < n; ++j)
                {
                    A[i][j] = A[i][j] - tmp * A[k][j];
                }
                A[i][k] = 0.0;
            }
            // 离开for循环时，各个线程默认同步，进入下一行的处理
        }
    }
  /*      cout<<"并行"<<endl;
        for(int i=0;i<n;i++)
        {
            for(int j=0;j<n;j++)
            {
                cout<<A[i][j]<<" ";
                if(j==n-1)
                    cout<<endl;
            }
        }*/
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << (tail - head) * 1000.0 / freq << "          ";
}
void ompdymanic()
{
    long long head, tail, freq; // timers
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    omp_set_num_threads(NUM_THREADS);
#pragma omp parallel
    {
        for (int k = 0; k < n; ++k) {
            // 串行部分，也可以尝试并行化
#pragma omp single
            {
                float tmp = D[k][k];
                for (int j = k + 1; j < n; ++j)
                {
                    D[k][j] =D[k][j] / tmp;
                }
                D[k][k] = 1.0;
            }
            // 并行部分，使用行划分
#pragma omp for schedule(dynamic,NUM_THREADS)
            for (int i = k + 1; i < n; ++i)
            {
                float tmp = D[i][k];
                for (int j = k + 1; j < n; ++j)
                {
                    D[i][j] = D[i][j] - tmp *D[k][j];
                }
                D[i][k] = 0.0;
               
            }
            // 离开for循环时，各个线程默认同步，进入下一行的处理
        }
    }
   /* cout<<"并行"<<endl;
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<n;j++)
        {
            cout<<A[i][j]<<" ";
            if(j==n-1)
                cout<<endl;
        }
    }*/
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << (tail - head) * 1000.0 / freq << "          ";
}
void ompsimd()
{
    long long head, tail, freq; // timers
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    omp_set_num_threads(NUM_THREADS);
#pragma omp parallel
    {
        for (int k = 0; k < n; ++k) {
            // 串行部分，也可以尝试并行化
#pragma omp single
            {
                float tmp = B[k][k];
                for (int j = k + 1; j < n; ++j)
                {
                    B[k][j] = B[k][j] / tmp;
                }
                B[k][k] = 1.0;
            }
            // 并行部分，使用行划分
#pragma omp for schedule(static,NUM_THREADS)
            for (int i = k + 1; i < n; ++i)
            {
                __m128 vaik = _mm_set1_ps(B[i][k]);
                int j = k + 1;
                for (j = k + 1; j + 4 <= n; j += 4)
                {
                    __m128 vakj = _mm_loadu_ps(&B[k][j]);
                    __m128 vaij = _mm_loadu_ps(&B[i][j]);
                    __m128 vx = _mm_mul_ps(vakj, vaik);
                    vaij = _mm_sub_ps(vaij, vx);

                    _mm_storeu_ps(&B[i][j], vaij);

                }
                while (j < n)
                {
                    B[i][j] -= B[k][j] * B[i][k];
                    j++;
                }
                B[i][k] = 0;
            }
            // 离开for循环时，各个线程默认同步，进入下一行的处理
        }
    }
    /*    cout<<"并行+simd"<<endl;
        for(int i=0;i<n;i++)
        {
            for(int j=0;j<n;j++)
            {
                cout<<B[i][j]<<" ";
                if(j==n-1)
                    cout<<endl;
            }
        }*/
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << (tail - head) * 1000.0 / freq << "          ";
}
int main()
{
    cout << NUM_THREADS << endl;
   
    while (n<1000)
    {
 
        cout << n<< "           ";
        m_reset(n);
        deepcopy();
        chuanxing();
        omp();
        ompdymanic();
        ompsimd();
        cout << endl;
        n += 100;
        //NUM_THREADS--;
    }
    return 0;
}
