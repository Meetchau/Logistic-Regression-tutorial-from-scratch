import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

np.random.seed(42)
X = np.random.rand(200, 1) * 10
noise = np.random.randn(200, 1) * 2
y = (X + noise > 5).astype(float)

# polynomial features
X_poly = np.hstack([X, X**2, X**3, X**4, X**5])

# scale features — this is the fix for the explosion
X_poly = (X_poly - X_poly.mean(axis=0)) / X_poly.std(axis=0)

# split BEFORE training
X_train, y_train = X_poly[:150], y[:150]
X_val, y_val = X_poly[150:], y[150:]

w = np.zeros((5, 1))
b = 0.0
lr = 0.01

for i in range(1000):
    z = np.dot(X_train, w) + b
    a = sigmoid(z)

    lambda_ = 0.001

    w -= lr * (np.dot(X_train.T, (a - y_train)) / len(y_train)+ 2* lambda_ * w)
    b -= lr * np.mean(a - y_train)

    if i % 100 == 0:
        p_safe = np.clip(a, 1e-10, 1 - 1e-10)
        loss = -np.mean(y_train * np.log(p_safe) + (1 - y_train) * np.log(1 - p_safe))
        print(f"iteration {i}: loss={loss:.4f}")

train_preds = (sigmoid(np.dot(X_train, w) + b) > 0.5).astype(float)
val_preds = (sigmoid(np.dot(X_val, w) + b) > 0.5).astype(float)

print("train accuracy:", (train_preds == y_train).mean())
print("val accuracy:", (val_preds == y_val).mean())
print("lambda:", lambda_, "weights:", w.flatten().round(4))


def cross_validate(X, y, lambda_, k=5, epochs=1000, lr=0.01):
    fold_size = len(X) // k
    val_scores = []

    for fold in range(k):
        # carve out validation fold
        val_start = fold * fold_size
        val_end = val_start + fold_size

        X_val = X[val_start:val_end]
        y_val = y[val_start:val_end]
        X_train = np.vstack([X[:val_start], X[val_end:]])
        y_train = np.vstack([y[:val_start], y[val_end:]])

        # train
        w = np.zeros((X.shape[1], 1))
        b = 0.0

        for i in range(epochs):
            z = np.dot(X_train, w) + b
            a = sigmoid(z)
            w -= lr * (np.dot(X_train.T, (a - y_train)) / len(y_train) + 2 * lambda_ * w)
            b -= lr * np.mean(a - y_train)

        # evaluate on val fold
        preds = (sigmoid(np.dot(X_val, w) + b) > 0.5).astype(float)
        val_scores.append((preds == y_val).mean())

    return np.mean(val_scores)

# sweep lambdas
for lambda_ in [0.001, 0.01, 0.1, 1.0]:
    score = cross_validate(X_poly, y, lambda_=lambda_)
    print(f"lambda={lambda_}: avg val accuracy={score:.3f}")