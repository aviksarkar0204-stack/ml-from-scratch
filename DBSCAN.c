#include<stdio.h>
#include<math.h>

#define N 9
#define EPS 2.0
#define MIN_SAMPLES 2

double distance(double *a, double *b) {
    double dx = a[0] - b[0];
    double dy = a[1] - b[1];
    return sqrt(dx*dx + dy*dy);
}

int get_neighbors(double X[][2], int idx, int *neighbors) {
    int count = 0;
    for (int i = 0; i < N; i++)
        if (distance(X[idx], X[i]) <= EPS)
            neighbors[count++] = i;
    return count;
}

int main()
{
    double X[N][2] = {
        {1,1}, {1,2}, {2,1}, {2,2},                     //cluster 0
        {8,8}, {8,9}, {9,8}, {9,9},                    //cluster 1
        {50,50}};                                              //Outliers

    int labels[N];
    for (int i = 0; i < N; i++)
    {
        labels[i] = -1;
    }

    int cluster_id = 0;

    for (int i = 0; i < N; i++)
    {
        if (labels[i] != -1) continue;  // already visited

        int neighbors[N];
        int n_neighbors = get_neighbors(X, i, neighbors);

        if (n_neighbors < MIN_SAMPLES) {
            labels[i] = -1;  // noise
            continue;
        }

        // start new cluster
        labels[i] = cluster_id;

        // expand cluster
        int queue[N], q_start = 0, q_end = 0;
        for (int k = 0; k < n_neighbors; k++)
            queue[q_end++] = neighbors[k];

        while (q_start < q_end)
        {
            int pt = queue[q_start++];
            if (labels[pt] == -1) labels[pt] = cluster_id;
            if (labels[pt] != -1 && labels[pt] != cluster_id) continue;

            labels[pt] = cluster_id;

            int pt_neighbors[N];
            int n_pt = get_neighbors(X, pt, pt_neighbors);

            if (n_pt >= MIN_SAMPLES)
                for (int k = 0; k < n_pt; k++)
                    if (labels[pt_neighbors[k]] == -1)
                        queue[q_end++] = pt_neighbors[k];
        }

        cluster_id++;
    }
    printf("DBSCAN Results:\n");
    for (int i = 0; i < N; i++)
        printf("  (%.0f, %.0f) -> cluster %d%s\n",
               X[i][0], X[i][1], labels[i],
               labels[i] == -1 ? " (noise)" : "");

    printf("\nTotal clusters found: %d\n", cluster_id);

    return 0;
}


