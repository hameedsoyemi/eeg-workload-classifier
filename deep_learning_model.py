import numpy as np
import scipy.io as sio
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense,Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


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

X_all = np.zeros((n_samples, n_electrodes, len(bands)))

for s in range(n_samples):
    for e in range(n_electrodes):
        row = data[e, :, s]
        power1 = np.abs(np.fft.fft(row[:half])[:half]) ** 2
        power2 = np.abs(np.fft.fft(row[half:])[:half]) ** 2
        for b_idx, (low, high) in enumerate(bands):
            X_all[s, e, b_idx] = (np.mean(power1[low:high]) + np.mean(power2[low:high])) / 2

print("Feature matrix X shape:", X_all.shape)

Y_all = np.zeros((n_samples, 2))
Y_all[label[0] == 0, 0] = 1
Y_all[label[0] == 1, 1] = 1

print("Label matrix Y shape:", Y_all.shape)

n_folds = 5
fold_size = n_samples // n_folds
indices = np.arange(n_samples)

folds = []
for i in range(n_folds):
    test_idx = indices[i * fold_size : (i + 1) * fold_size]
    train_idx = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
    folds.append((train_idx, test_idx))


configs = {
    "Baseline (32/64 filters, LR=0.001)": {
        "filters_1": 32, "filters_2": 64, "dense_units": 32,
        "dropout": 0.3, "lr": 0.001
    },
    "More Filters (64/128, LR=0.001)": {
        "filters_1": 64, "filters_2": 128, "dense_units": 64,
        "dropout": 0.5, "lr": 0.001
    },
    "Higher LR (64/128, LR=0.01)": {
        "filters_1": 64, "filters_2": 128, "dense_units": 64,
        "dropout": 0.5, "lr": 0.01
    },
    "Less Dropout (64/128, dropout=0.2)": {
        "filters_1": 64, "filters_2": 128, "dense_units": 128,
        "dropout": 0.2, "lr": 0.001
    },
}



all_results = {}

for config_name, cfg in configs.items():
    print(f"\n{'='*50}")
    print(f"Config: {config_name}")
    print(f"{'='*50}")

    accuracies = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds):

        X_train, X_test = X_all[train_idx], X_all[test_idx]
        Y_train, Y_test = Y_all[train_idx], Y_all[test_idx]

        mean = np.mean(X_train, axis=0, keepdims=True)
        std = np.std(X_train, axis=0, keepdims=True) + 1e-8
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        model = Sequential([
            Input(shape=(62, 5)),
            Conv1D(cfg["filters_1"], kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Conv1D(cfg["filters_2"], kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(cfg["dense_units"], activation='relu'),
            Dropout(cfg["dropout"]),
            Dense(2, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=cfg["lr"]),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        model.fit(
            X_train, Y_train,
            validation_data=(X_test, Y_test),
            epochs=100,
            batch_size=16,
            callbacks=[EarlyStopping(monitor='val_loss', patience=10,
                                      restore_best_weights=True, verbose=0)],
            verbose=0
        )

        _, acc = model.evaluate(X_test, Y_test, verbose=0)
        accuracies.append(acc * 100)
        print(f"  Fold {fold_idx + 1}: {acc * 100:.2f}%")

    all_results[config_name] = accuracies
    print(f"  Mean: {np.mean(accuracies):.2f}% (±{np.std(accuracies):.2f}%)")


print(f"\n{'='*50}")
print("RESULTS SUMMARY")
print(f"{'='*50}")

for name, accs in all_results.items():
    print(f"\n{name}:")
    for i, a in enumerate(accs):
        print(f"  Fold {i + 1}: {a:.2f}%")
    print(f"  Mean: {np.mean(accs):.2f}% (±{np.std(accs):.2f}%)")

best = max(all_results, key=lambda k: np.mean(all_results[k]))
print(f"\nBest config: {best} -> {np.mean(all_results[best]):.2f}%")