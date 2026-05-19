import numpy as np
x = np.array([[150, 48],
              [155, 52],
              [160,55],
              [165, 60],
              [170, 65],
              [175, 70],
              [180, 75],
              [185, 82],
              [190, 88],
              [195, 95]],dtype=float)
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
test = np.array([160, 55],dtype=float)

classes = np.unique(y)
mean  = np.zeros((len(classes), x.shape[1]))
var   = np.zeros((len(classes), x.shape[1]))
prior = np.zeros(len(classes))

for c in classes:
    x_c       = x[y == c]        # filter rows belonging to class c
    mean[c]   = np.mean(x_c, axis=0)
    var[c]    = np.var(x_c, axis=0)
    prior[c]  = len(x_c) / len(x)

scores = []
for c in classes:
    log_prior = np.log(prior[c])
    log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var[c]) + ((test - mean[c])**2 / var[c]))
    scores.append(log_prior + log_likelihood)

prediction = int(classes[np.argmax(scores)])

print(f"Score class 0: {scores[0]:.6f}")
print(f"Score class 1: {scores[1]:.6f}")
print(f"Prediction: class {prediction}")


