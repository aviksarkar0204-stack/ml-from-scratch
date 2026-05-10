#include <stdio.h>
#define DEGREE 2
#define N 10
#define LR 0.00001
#define EPOCHS 1000000

int main()
{
    double x[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    double y[] = {1, 4, 9, 16, 25, 36, 49, 64, 81, 100};
    double w[DEGREE];
    for (int i = 0; i < DEGREE; i++)
    {
        w[i] = 0.0;
    }
    double b = 0.0;
    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        double loss = 0.0;
        for (int i = 0; i < N; i++)
        {
            double pred = w[0]*x[i] + w[1]*x[i]*x[i] + b;
            loss += (y[i] - pred)*(y[i] - pred);
        }
        loss /= N;
        double dw[DEGREE];
        for (int i = 0; i < DEGREE; i++)
        {
            dw[i] = 0.0;
        }
        double db = 0;
        for (int i = 0 ; i < N; i++)
        {
            double pred = w[0]*x[i] + w[1]*x[i]*x[i] + b;
            dw[0] += -2.0/N * x[i] * (y[i] - pred);
            dw[1] += -2.0/N * x[i]*x[i] * (y[i] - pred);
            db += -2.0/N * (y[i] - pred);
        }
        for (int i = 0; i < DEGREE; i++)
        {
            w[i] -= LR * dw[i];
        }
        b -= LR * db;
        if (epoch % 10000 == 0)
            printf("epoch: %d | loss: %f | w0: %f | w1: %f | b: %f\n",
                   epoch, loss, w[0], w[1], b);
    }
    printf("\nPredictions:\n");
    for (int i = 0; i < N; i++)
    {
        double pred = w[0]*x[i] + w[1]*x[i]*x[i] + b;
        printf("x: %.0f | pred: %.2f | actual: %.0f\n", x[i], pred, y[i]);
    }
}