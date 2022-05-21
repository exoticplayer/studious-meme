#include <iostream>
#include <pthread.h>
#include<time.h>
#include<arm_neon.h>

using namespace std;
int n = 50;
int NUM_THREADS = 8;
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
    struct timespec sts,ets;
    timespec_get(&sts,TIME_UTC);
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
  /*  	 cout<<"串行"<<endl;
        for(int i=0;i<n;i++)
        {
            for(int j=0;j<n;j++)
            {
                cout<<C[i][j]<<" ";
                if(j==n-1)
                    cout<<endl;
            }
        }*/
    timespec_get(&ets,TIME_UTC);
    time_t dsec=ets.tv_sec-sts.tv_sec;
    long dnsec=ets.tv_nsec-sts.tv_nsec;
    if(dnsec<0)
    {
        dsec--;
        dnsec+=1000000000ll;
    }
   printf("%lld.%09lld",dsec,dnsec);

}
void omp()
{
    struct timespec sts,ets;
    timespec_get(&sts,TIME_UTC);
//    omp_set_num_threads(NUM_THREADS);
#pragma omp parallel num_threads(NUM_THREADS)
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
        //cout<<"并行"<<endl;
        //for(int i=0;i<n;i++)
        //{
        //    for(int j=0;j<n;j++)
        //    {
        //        cout<<A[i][j]<<" ";
        //        if(j==n-1)
        //            cout<<endl;
        //    }
        //}
    timespec_get(&ets,TIME_UTC);
    time_t dsec=ets.tv_sec-sts.tv_sec;
    long dnsec=ets.tv_nsec-sts.tv_nsec;
    if(dnsec<0)
    {
        dsec--;
        dnsec+=1000000000ll;
    }
   printf("%lld.%09lld",dsec,dnsec);
}
void ompdymanic()
{
    struct timespec sts,ets;
    timespec_get(&sts,TIME_UTC);
//    omp_set_num_threads(NUM_THREADS);
#pragma omp parallel num_threads(NUM_THREADS)
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
    //cout<<"并行"<<endl;
    //for(int i=0;i<n;i++)
    //{
    //    for(int j=0;j<n;j++)
    //    {
    //        cout<<A[i][j]<<" ";
    //        if(j==n-1)
    //            cout<<endl;
    //    }
    //}
    timespec_get(&ets,TIME_UTC);
    time_t dsec=ets.tv_sec-sts.tv_sec;
    long dnsec=ets.tv_nsec-sts.tv_nsec;
    if(dnsec<0)
    {
        dsec--;
        dnsec+=1000000000ll;
    }
   printf("%lld.%09lld",dsec,dnsec);
}
void ompsimd()
{
    struct timespec sts,ets;
    timespec_get(&sts,TIME_UTC);
//    omp_set_num_threads(NUM_THREADS);
#pragma omp parallel num_threads(NUM_THREADS)
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
                float32x4_t vaik = vdupq_n_f32(B[i][k]);
                int j = k + 1;
                for (j = k + 1; j + 4 <= n; j += 4)
                {
                    float32x4_t vakj = vld1q_f32(&B[k][j]);
                    float32x4_t vaij = vld1q_f32(&B[i][j]);
                    float32x4_t vx = vmulq_f32(vakj, vaik);
                    vaij = vsubq_f32(vaij, vx);

                    vst1q_f32(&B[i][j], vaij);

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
        //cout<<"并行+simd"<<endl;
        //for(int i=0;i<n;i++)
        //{
        //    for(int j=0;j<n;j++)
        //    {
        //        cout<<B[i][j]<<" ";
        //        if(j==n-1)
        //            cout<<endl;
        //    }
        //}
    timespec_get(&ets,TIME_UTC);
    time_t dsec=ets.tv_sec-sts.tv_sec;
    long dnsec=ets.tv_nsec-sts.tv_nsec;
    if(dnsec<0)
    {
        dsec--;
        dnsec+=1000000000ll;
    }
   printf("%lld.%09lld",dsec,dnsec);
}
int main()
{
    cout << "new" << endl;
    n=400;
    while (NUM_THREADS>1)
    {

        cout << NUM_THREADS << "           ";
        m_reset(n);
        deepcopy();

        chuanxing();
        cout<<"           ";
        omp();
        cout<<"            ";
        ompdymanic();
        cout<<"           " ;
        ompsimd();
        cout << endl;
        NUM_THREADS--;
    }
    return 0;
}
