import numpy as np
import scipy.io as sio


mat = sio.loadmat("WLDataCW.mat")
data = mat['data']
label = mat['label']

print("Data shape:", data.shape)
print("Label shape:", label.shape)


n_electrodes = data.shape[0]
n_points = data.shape[1]
n_samples = data.shape[2]
half = n_points // 2  

bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 128)]
n_bands = len(bands)

X_all = np.zeros((n_electrodes * n_bands, n_samples))

for s in range(n_samples):
    idx = 0
    for e in range(n_electrodes):
        row = data[e, :, s]
        power1 = np.abs(np.fft.fft(row[:half])[:half]) ** 2
        power2 = np.abs(np.fft.fft(row[half:])[:half]) ** 2
        for low, high in bands:
            X_all[idx, s] = (np.mean(power1[low:high]) + np.mean(power2[low:high])) / 2
            idx += 1

print("Feature matrix X shape:", X_all.shape)


Y_all = np.zeros((2, n_samples))
Y_all[0, label[0] == 0] = 1
Y_all[1, label[0] == 1] = 1

print("Label matrix Y shape:", Y_all.shape)


n_folds = 5
fold_size = n_samples // n_folds
indices = np.arange(n_samples)

folds = []
for i in range(n_folds):
    test_idx = indices[i * fold_size : (i + 1) * fold_size]
    train_idx = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
    folds.append((train_idx, test_idx))



def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def initialize_parameters(n_features, n_classes=2):
    np.random.seed(42)
    W = np.random.randn(n_classes, n_features) * 0.01
    b = np.zeros((n_classes, 1))
    return W, b


def forward_propagation(X, W, b):
    Z = np.dot(W, X) + b
    A = sigmoid(Z)
    return A


def compute_cost(A, Y):
    m = Y.shape[1]
    cost = -1.0 / m * np.sum(Y * np.log(A + 1e-8) + (1 - Y) * np.log(1 - A + 1e-8))
    return cost


def backward_propagation(X, A, Y):
    m = X.shape[1]
    dZ = A - Y
    dW = (1.0 / m) * np.dot(dZ, X.T)
    db = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)
    return dW, db


def update_parameters(W, b, dW, db, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return W, b


def predict(X, W, b):
    A = forward_propagation(X, W, b)
    predictions = (A == np.max(A, axis=0, keepdims=True)).astype(int)
    return predictions


def compute_accuracy(predictions, Y):
    correct = np.sum(np.all(predictions == Y, axis=0))
    return correct / Y.shape[1] * 100.0



learning_rate = 0.01
iterations = 3000
accuracies = []

for fold_idx, (train_idx, test_idx) in enumerate(folds):
    print(f"\n--- Fold {fold_idx + 1} ---")

    X_train = X_all[:, train_idx]
    Y_train = Y_all[:, train_idx]
    X_test = X_all[:, test_idx]
    Y_test = Y_all[:, test_idx]


    mean = np.mean(X_train, axis=1, keepdims=True)
    std = np.std(X_train, axis=1, keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    W, b = initialize_parameters(X_train.shape[0])

    for i in range(iterations):
        A = forward_propagation(X_train, W, b)
        cost = compute_cost(A, Y_train)
        dW, db = backward_propagation(X_train, A, Y_train)
        W, b = update_parameters(W, b, dW, db, learning_rate)

        if i % 500 == 0:
            print(f"  Iter {i}, Cost: {cost:.6f}")


    preds = predict(X_test, W, b)
    acc = compute_accuracy(preds, Y_test)
    accuracies.append(acc)
    print(f"  Accuracy: {acc:.2f}%")


print("\n========== RESULTS ==========")
for i, acc in enumerate(accuracies):
    print(f"Fold {i + 1}: {acc:.2f}%")
print(f"Mean Accuracy: {np.mean(accuracies):.2f}%")
print(f"Std: {np.std(accuracies):.2f}%")