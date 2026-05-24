#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

#define N 6
#define F 2
#define MAX_DEPTH 3
#define NUM_TREES 10
#define MAX_FEATURES 1

typedef struct Node
{
    int feature;
    double threshold;
    int class;
    struct Node *left;
    struct Node *right;
}Node;

double gini(int *labels, int n)
{
    if (n==0)
    {
        return 0;
    }
    int count[2] = {0,0};
    for (int i=0;i<n;i++)
    {
        count[labels[i]]++;
    }
    double impurity = 1.0;
    for (int c = 0; c < 2; c++)
    {
        double p = (double)count[c] / n;
        impurity -= p * p;
    }
    return impurity;
}

void best_split(double X[][F], int *labels, int n, int *best_feat, double *best_thresh)
{
    double best_gini = 1e9;
    *best_feat = -1;
    *best_thresh = 0;
    int features[F] = {0, 1};
    for (int i = F-1; i > 0; i--) {
        int j = rand() % (i+1);
        int tmp = features[i];
        features[i] = features[j];
        features[j] = tmp;
    }
    for (int fi = 0; fi < MAX_FEATURES; fi++) {
        int f = features[fi];
        for (int i = 0; i < n; i++) {         // ← add this
            double thresh = X[i][f];
            int left_labels[N], right_labels[N];
            int nl = 0, nr = 0;
            for (int j = 0; j < n; j++) {
                if (X[j][f] < thresh) left_labels[nl++] = labels[j];
                else                  right_labels[nr++] = labels[j];
            }
            double g = ((double)nl/n) * gini(left_labels, nl)
                     + ((double)nr/n) * gini(right_labels, nr);
            if (g < best_gini) {
                best_gini    = g;
                *best_feat   = f;
                *best_thresh = thresh;
            }
        }
    }
}

Node* build_tree(double X[][F], int *labels, int n, int depth)
{
    Node *node = (Node*)malloc(sizeof(Node));

    int count[2] = {0, 0};
    for (int i = 0; i < n; i++) count[labels[i]]++;

    if (depth >= MAX_DEPTH || count[0] == 0 || count[1] == 0)
    {
        node->feature  = -1;
        node->left     = NULL;
        node->right    = NULL;
        node->class    = count[1] > count[0] ? 1 : 0;
        return node;
    }

    int best_feat;
    double best_thresh;
    best_split(X, labels, n, &best_feat, &best_thresh);

    double X_left[N][F],  X_right[N][F];
    int    l_labels[N],   r_labels[N];
    int    nl = 0,        nr = 0;

    for (int i = 0; i < n; i++)
    {
        if (X[i][best_feat] < best_thresh)
        {
            for (int f = 0; f < F; f++) X_left[nl][f] = X[i][f];
            l_labels[nl++] = labels[i];
        }
        else
        {
            for (int f = 0; f < F; f++) X_right[nr][f] = X[i][f];
            r_labels[nr++] = labels[i];
        }
    }

    node->feature   = best_feat;
    node->threshold = best_thresh;
    node->class     = -1;

    node->left  = build_tree(X_left,  l_labels, nl, depth + 1);
    node->right = build_tree(X_right, r_labels, nr, depth + 1);

    return node;
}

int predict(Node *node, double *x)
{
    if (node->feature == -1)
        return node->class;
    if (x[node->feature] < node->threshold)
        return predict(node->left, x);
    else
        return predict(node->right, x);
}

int main()
{
    srand(time(NULL));

    double X[N][F] = {{2,3},{3,3},{3,4},{6,7},{7,8},{8,7}};
    int y[N] = {0, 0, 0, 1, 1, 1};

    Node *forest[NUM_TREES];
    for (int t = 0; t < NUM_TREES; t++)
    {
        double X_boot[N][F];
        int    y_boot[N];
        for (int i = 0; i < N; i++)
        {
            int idx = rand() % N;
            for (int f = 0; f < F; f++) X_boot[i][f] = X[idx][f];
            y_boot[i] = y[idx];
        }
        forest[t] = build_tree(X_boot, y_boot, N, 0);
    }

    printf("Predictions:\n");
    for (int i = 0; i < N; i++)
    {
        int votes[2] = {0, 0};
        for (int t = 0; t < NUM_TREES; t++)
            votes[predict(forest[t], X[i])]++;
        int pred = votes[1] > votes[0] ? 1 : 0;
        printf("  (%.0f, %.0f) -> predicted: %d | actual: %d %s\n",
               X[i][0], X[i][1], pred, y[i],
               pred == y[i] ? "OK" : "WRONG");
    }

    printf("\nNew points:\n");
    double test1[F] = {1, 2};
    double test2[F] = {7, 7};

    int v1[2] = {0,0}, v2[2] = {0,0};
    for (int t = 0; t < NUM_TREES; t++) {
        v1[predict(forest[t], test1)]++;
        v2[predict(forest[t], test2)]++;
    }
    printf("  (1, 2) -> class %d\n", v1[1] > v1[0] ? 1 : 0);
    printf("  (7, 7) -> class %d\n", v2[1] > v2[0] ? 1 : 0);

    return 0;
}
