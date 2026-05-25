#include<stdio.h>
#include<math.h>

#define N 6
#define F 2
#define LR 0.001
#define LAMDA 0.01
#define EPOCH 1000

int main()
{
    double x[N][F] = {{1,1}, {1,2}, {2,1},
     {4,4}, {4,5}, {5,4}};
    int y[N] = {-1, -1, -1, 1, 1, 1};
    double w[F] = {0, 0};
    double b = 0;
    for (int epoch = 0; epoch < EPOCH; epoch++)
    {
        for (int i = 0; i < N; i++)
        {

            double score = b;
            for (int f = 0; f < F; f++)
                score += w[f] * x[i][f];


            if (y[i] * score >= 1)
            {

                for (int f = 0; f < F; f++)
                    w[f] -= LR * (2 * LAMDA * w[f]);
            }
            else
            {

                for (int f = 0; f < F; f++)
                    w[f] -= LR * (2 * LAMDA * w[f] - y[i] * x[i][f]);
                b -= LR * (-y[i]);
            }
        }

        if (epoch % 100 == 0)
        {
            printf("Epoch %d | w: [%.4f, %.4f] | b: %.4f\n",
                   epoch, w[0], w[1], b);
        }
    }
    printf("\nPredictions:\n");
    for (int i = 0; i < N; i++)
    {
        double score = b;
        for (int f = 0; f < F; f++)
            score += w[f] * x[i][f];
        int pred = score >= 0 ? 1 : -1;
        printf("  (%.0f, %.0f) -> score: %.4f | pred: %d | actual: %d %s\n",
               x[i][0], x[i][1], score, pred, y[i],
               pred == y[i] ? "OK" : "WRONG");
    }
    return 0;
}
