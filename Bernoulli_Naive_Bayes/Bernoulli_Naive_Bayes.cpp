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
    double test[F] = {0,0};
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

    double scores[C];
    for (int c = 0; c < C; c++) {
        scores[c] = log(prior[c]);
        for (int f = 0; f < F; f++) {
            double p = (feat_count[c][f] + 1) / (class_count[c] + 2);
            scores[c] += test[f] * log(p) + (1 - test[f]) * log(1 - p);
        }
    }

    int prediction = scores[0] > scores[1] ? 0 : 1;

    printf("Score class 0 (not spam): %f\n", scores[0]);
    printf("Score class 1 (spam):     %f\n", scores[1]);
    printf("Prediction: class %d %s\n", prediction, prediction == 1 ? "(spam)" : "(not spam)");


    return 0;
}