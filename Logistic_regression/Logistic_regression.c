#include <math.h>
#include<stdio.h>
#define N 10
#define LR 0.01
#define EPOCH 5000

int main()
{
    double x[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    double y[] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    double w = 0;
    double b = 0;

    for (int epoch = 0; epoch < EPOCH; epoch++)
    {
        double loss = 0;
        for (int i = 0; i < N; i++)
        {
            double z = w*x[i] + b;
            double pred = 1.0/(1.0 + exp(-z));
            loss += -(y[i] * log(pred) + (1 - y[i]) * log(1 - pred));
        }
        loss /= N;
        if (epoch % 100 == 0)
        {
            printf("epoch: %d, w: %f, b: %f, loss: %f\n", epoch, w, b, loss);
        }
        if (loss < 0.000001)
        {
            break;
        }
        double dw = 0;
        double db = 0;
        for (int i = 0; i < N; i++)
        {
            double z = w*x[i] + b;
            double pred = 1.0/(1.0 + exp(-z));
            dw += (pred - y[i]) * x[i];
            db += (pred - y[i]);
        }
        dw /= N;
        db /= N;
        w = w - LR * dw;
        b = b - LR * db;
    }
    printf("\nFinal: w = %f, b = %f\n", w, b);
    printf("\nPredictions:\n");
    for (int i = 0; i < N; i++)
    {
        double z = w * x[i] + b;
        double pred = 1.0 / (1.0 + exp(-z));
        int class = pred >= 0.5 ? 1 : 0;
        printf("x: %.0f | prob: %.4f | class: %d | expected: %.0f\n",
               x[i], pred, class, y[i]);
    }
}
