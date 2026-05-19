#include <stdio.h>
#include <math.h>
#define N 10
#define C 2
#define F 2

int main()
{
#define F 3
    double X[N][F] = {
        {3, 2, 0},  {4, 1, 0},  {5, 3, 0},  {2, 2, 0},  {6, 4, 0},  // spam
        {0, 0, 3},  {0, 0, 2},  {0, 0, 4},  {0, 0, 2},  {0, 0, 3}   // not spam
    };
    int y[N] = {1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
    double test[F] = {0, 0, 3};
    double feat_count[C][F];
    double total[C];
    double prior[C];

    for (int c = 0; c < C; c++) {
        total[c] = 0;
        prior[c] = 0;
        for (int f = 0; f < F; f++)
            feat_count[c][f] = 0;
    }

    int class_count[C] = {0, 0};
    for (int i = 0; i < N; i++) {
        int c = y[i];
        class_count[c]++;
        for (int f = 0; f < F; f++) {
            feat_count[c][f] += X[i][f];
            total[c]         += X[i][f];
        }
    }

    for (int c = 0; c < C; c++)
        prior[c] = (double)class_count[c] / N;

    double scores[C];
    for (int c = 0; c < C; c++) {
        scores[c] = log(prior[c]);
        for (int f = 0; f < F; f++) {
            // Laplace smoothing: add 1 to count, add F to total
            double prob = (feat_count[c][f] + 1) / (total[c] + F);
            scores[c] += test[f] * log(prob);
        }
    }

    int prediction = scores[0] > scores[1] ? 0 : 1;


    printf("feat_count[0]: %f %f %f\n", feat_count[0][0], feat_count[0][1], feat_count[0][2]);
    printf("feat_count[1]: %f %f %f\n", feat_count[1][0], feat_count[1][1], feat_count[1][2]);
    printf("total[0]: %f  total[1]: %f\n", total[0], total[1]);
    printf("prior[0]: %f  prior[1]: %f\n", prior[0], prior[1]);
    printf("\nScore class 0 (not spam): %f\n", scores[0]);
    printf("Score class 1 (spam):     %f\n", scores[1]);
    printf("Prediction: class %d %s\n", prediction, prediction == 1 ? "(spam)" : "(not spam)");

}