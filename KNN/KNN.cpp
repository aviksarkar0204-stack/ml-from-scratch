#include <stdio.h>
#include <math.h>
#define N 10
#define K 3

int main()
{
    double x[] = {1,2,3,4,5,6,7,8,9,10};
    double y[] = {0,0,0,0,0,1,1,1,1,1};
    double test = 4.5;
    double dist[N];
    for (int i = 0; i < N; i++)
    {
        dist[i] = fabs(x[i] - test);
    }
    int idx[N];
    for (int i = 0; i < N; i++)
    {
        idx[i] = i;
    }
    for (int i = 0; i < N-1; i++)
    {
        for (int j = i+1; j < N; j++)
        {
            if (dist[idx[j]] < dist[idx[i]])
            {
                int tmp = idx[i];
                idx[i]  = idx[j];
                idx[j]  = tmp;
            }
        }
    }
    int votes = 0;
    for (int i = 0; i < K; i++)
    {
        votes += (int)y[idx[i]];
    }
    int prediction = votes > K/2 ? 1 : 0;
    printf("Test point: %.1f\n\n", test);
    printf("K=%d nearest neighbors:\n", K);
    for (int i = 0; i < K; i++)
        printf("  x: %.0f | dist: %.2f | class: %.0f\n",
               x[idx[i]], dist[idx[i]], y[idx[i]]);

    printf("\nVotes for class 1: %d / %d\n", votes, K);
    printf("Prediction: class %d\n", prediction);
}