#include <stdio.h>
#include <math.h>
#define N 6
#define K 2
#define F 2
#define MAX_ITER 100

int main()
{
    double X[N][F] = {{1,1},
        {1,2},
        {2,1},
        {8,8},
        {8,9},
        {9,8}};
    double centroids[K][F];
    int labels[N];
    for (int k = 0; k < K; k++)
        for (int f = 0; f < F; f++)
            centroids[k][f] = X[k][f];
    for (int iter = 0; iter < MAX_ITER; iter++)
    {

        for (int i = 0; i < N; i++)
        {
            double min_dist = 1e9;
            for (int k = 0; k < K; k++)
            {
                double dist = 0;
                for (int f = 0; f < F; f++)
                {
                    double diff = X[i][f] - centroids[k][f];
                    dist += diff * diff;
                }
                dist = sqrt(dist);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    labels[i] = k;
                }
            }
        }

        double new_centroids[K][F];
        int count[K];


        for (int k = 0; k < K; k++) {
            count[k] = 0;
            for (int f = 0; f < F; f++)
                new_centroids[k][f] = 0;
        }

        for (int i = 0; i < N; i++) {
            int k = labels[i];
            count[k]++;
            for (int f = 0; f < F; f++)
                new_centroids[k][f] += X[i][f];
        }

        for (int k = 0; k < K; k++)
            for (int f = 0; f < F; f++)
                new_centroids[k][f] /= count[k];

        int converged = 1;
        for (int k = 0; k < K; k++)
            for (int f = 0; f < F; f++)
                if (fabs(new_centroids[k][f] - centroids[k][f]) > 1e-6)
                    converged = 0;

        for (int k = 0; k < K; k++)
            for (int f = 0; f < F; f++)
                centroids[k][f] = new_centroids[k][f];

        if (converged) {
            printf("Converged at iteration %d\n", iter);
            break;
        }
    }
    printf("\nFinal centroids:\n");
    for (int k = 0; k < K; k++)
        printf("  Centroid %d: (%.2f, %.2f)\n", k, centroids[k][0], centroids[k][1]);

    printf("\nPoint assignments:\n");
    for (int i = 0; i < N; i++)
        printf("  (%.0f, %.0f) → cluster %d\n", X[i][0], X[i][1], labels[i]);
}