#include<stdio.h>
#define N 5
#define LR 0.01
#define EPOCH 5000

int main()
{
    double x[] = {1,2,3,4,5};
    double y[] = {2,4,6,8,10};
    double w = 0;               //weight,start at zero
    double b = 0;               //bais,start at zero

    for (int epoch = 0; epoch < EPOCH; epoch++)
    {
        double loss = 0;

        for (int i = 0; i < N; i++)
        {
            double pred = w*x[i] + b;
            loss += (y[i] - pred)*(y[i] - pred);
        }

        loss /= N;
        double dw = 0;
        double db = 0;

        if (loss < 0.000001)
        {
            break;
        }

        if (epoch % 100 == 0)
        {
            printf("epoch: %d, w: %f, b: %f, loss: %f\n", epoch, w, b, loss);
        }

        for (int i = 0; i < N; i++)
        {
            double pred = w*x[i] + b;
            dw += -2.0/N * x[i] * (y[i] - pred);
            db += -2.0/N * (y[i] - pred);
        }
        w = w - LR * dw;
        b = b - LR * db;
    }
}