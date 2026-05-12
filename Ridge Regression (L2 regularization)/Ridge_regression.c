#include <stdio.h>
#define N 10
#define LR 0.01
#define EPOCH 5000
#define LAMBDA 0.01
int main()
{
    double x[] = {1,2,3,4,5,6,7,8,9,10};
    double y[] = {1,3,5,7,9,11,13,15,17,19};
    double w = 0;
    double b = 0;
    for (int epoch = 0; epoch < EPOCH; epoch++)
    {
        double loss = 0;
        for (int i = 0; i < N; i++)
        {
            double pred = w*x[i] + b;
            loss += (y[i] - pred)*(y[i] - pred);
        }
        loss /= N;
        loss += LAMBDA * w * w;
        if (loss < 0.000001)
        {
            break;
        }
        double dw = 0;
        double db = 0;
        for (int i = 0; i < N; i++)
        {
            double pred = w*x[i] + b;
            dw += -2.0/N * x[i] * (y[i] - pred);
            db += -2.0/N * (y[i] - pred);
        }
        dw += 2 * LAMBDA * w;
        w -= LR * dw;
        b -= LR * db;
        if (epoch % 1000 == 0)
        {
            printf("epoch: %d, w: %f, b: %f, loss: %f\n", epoch, w, b, loss);
        }
    }
    printf("\nFinal: w = %f, b = %f\n", w, b);
    printf("\nPredictions:\n");
    for (int i = 0; i < N; i++)
    {
        double pred = w * x[i] + b;
        printf("x: %.0f | pred: %.2f | actual: %.0f\n", x[i], pred, y[i]);
    }
}