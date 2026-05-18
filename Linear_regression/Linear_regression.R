N <- 5
LR <- 0.01
EPOCH <- 5000

x <- c(1, 2, 3, 4, 5)
y <- c(2, 4, 6, 8, 10)

w <- 0
b <- 0

for (epoch in 0:(EPOCH - 1)) {
    # Calculate loss
    pred <- w * x + b
    loss <- sum((y - pred)^2) / N

    if (loss < 0.000001) {
        break
    }

    if (epoch %% 100 == 0) {
        cat(sprintf("epoch: %d, w: %f, b: %f, loss: %f\n", epoch, w, b, loss))
    }

    pred <- w * x + b
    dw <- sum(-2.0 / N * x * (y - pred))
    db <- sum(-2.0 / N * (y - pred))

    w <- w - LR * dw
    b <- b - LR * db
}

cat(sprintf("Final: w: %f, b: %f, loss: %f\n", w, b, loss))