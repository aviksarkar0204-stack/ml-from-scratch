import numpy as np
x = np.array([[1, 0],
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 0],
        [0, 0],
        [0, 0],
        [0, 1],
        [0, 0],
        [0, 1]],dtype=float)
y = np.array([1,1,1,1,1,0,0,0,0,0])
classes = np.unique(y)
test = np.array([1,0],dtype=float)

feat_count = np.zeros((len(classes), x.shape[1]))
total      = np.zeros(len(classes))
prior      = np.zeros(len(classes))

for c in classes:
    x_c = x[y == c]
    feat_count[c] = np.sum(x_c, axis=0)
    total[c] = np.sum(x_c)
    prior[c] = len(x_c) / len(x)

scores = []
for c in classes:
    c = int(c)
    log_prior = np.log(prior[c])
    prob = (feat_count[c] + 1) / (total[c] + x.shape[1])
    log_likelihood = np.sum(test * np.log(prob))
    scores.append(log_prior + log_likelihood)

prediction = int(classes[np.argmax(scores)])
print(f"Score class 0 (not spam): {scores[0]:.6f}")
print(f"Score class 1 (spam):     {scores[1]:.6f}")
print(f"Prediction: class {prediction}")
