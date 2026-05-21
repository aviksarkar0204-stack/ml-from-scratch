#include <stdio.h>
#include <math.h>
#define N 10
#define C 2
#define F 2

int main()
{
    double X[N][F] = {{1, 1},   //← spam
        {1, 0},   //← spam
        {1, 1},   //← spam
        {1, 1},   //← spam
        {1, 0},   //← spam
        {0, 0},   //← not spam
        {0, 0},   //← not spam
        {0, 1},   //← not spam
        {0, 0},   //← not spam
        {0, 1}};   //← not spam

    int y[N] = {1,1,1,1,1,0,0,0,0,0};
    double test[F] = {1, 1};
    double feat_count[C][F];
    int class_count[C];
    double prior[C];

    for (int c = 0; c < C; c++) {
        class_count[c] = 0;
        prior[c] = 0;
        for (int f = 0; f < F; f++)
            feat_count[c][f] = 0;
    }

    for (int i = 0; i < N; i++) {
        int c = y[i];
        class_count[c]++;
        for (int f = 0; f < F; f++)
            feat_count[c][f] += X[i][f];  // X[i][f] is 0 or 1
    }

    for (int c = 0; c < C; c++)
        prior[c] = (double)class_count[c] / N;


    return 0;
}