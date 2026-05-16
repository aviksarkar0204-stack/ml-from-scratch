#include<stdio.h>
#include<math.h>

#define N 10  // Number of samples
#define F 2   // Number of features
#define C 2   // Number of classes


int main()
{
    double x[N][F] = {
        {150, 48},
        {155, 52},
        {160, 55},
        {165, 60},
        {170, 65},
        {175, 70},
        {180, 75},
        {185, 82},
        {190, 88},
        {195, 95}
    };
    int y[N] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    double test[F] = {185, 80};

    double mean[C][F];
    double var[C][F];
    double prior[C];
    for (int c = 0; c < C; c++) {
        prior[c] = 0;
        for (int f = 0; f < F; f++) {
            mean[c][f] = 0;
            var[c][f]  = 0;
        }
    }
    int count[C] = {0, 0};
    for (int i = 0; i < N; i++) {
        int c = y[i];
        count[c]++;
        for (int f = 0; f < F; f++)
            mean[c][f] += x[i][f];
    }
    for (int c = 0; c < C; c++) {
        prior[c] = (double)count[c] / N;
        for (int f = 0; f < F; f++)
            mean[c][f] /= count[c];
    }
    for (int i = 0; i < N; i++) {
        int c = y[i];
        for (int f = 0; f < F; f++) {
            double diff = x[i][f] - mean[c][f];
            var[c][f] += diff * diff;
        }
    }

    for (int c = 0; c < C; c++)
        for (int f = 0; f < F; f++)
            var[c][f] /= count[c];
    // compute log score for each class
    double scores[C];
    for (int c = 0; c < C; c++) {
        scores[c] = log(prior[c]);   // start with log prior
        for (int f = 0; f < F; f++) {
            double diff = test[f] - mean[c][f];
            // log of Gaussian PDF
            scores[c] += -0.5 * log(2 * M_PI * var[c][f])
                       - (diff * diff) / (2 * var[c][f]);
        }
    }

    // pick class with highest score
    int prediction = scores[0] > scores[1] ? 0 : 1;

    printf("Score class 0: %f\n", scores[0]);
    printf("Score class 1: %f\n", scores[1]);
    printf("Prediction: class %d\n", prediction);
    return 0;
}