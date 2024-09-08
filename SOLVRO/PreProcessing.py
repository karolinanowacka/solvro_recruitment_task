import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def load_data():
    X_train = np.load('/Users/karolinanowacka/Downloads/solvro-rekrutacja-zimowa-ml-2022/X_train.npy')
    y_train = np.load('/Users/karolinanowacka/Downloads/solvro-rekrutacja-zimowa-ml-2022/y_train.npy')
    X_val = np.load('/Users/karolinanowacka/Downloads/solvro-rekrutacja-zimowa-ml-2022/X_val.npy')
    y_val = np.load('/Users/karolinanowacka/Downloads/solvro-rekrutacja-zimowa-ml-2022/y_val.npy')
    X_test = np.load('/Users/karolinanowacka/Downloads/solvro-rekrutacja-zimowa-ml-2022/X_test.npy')

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"unique labels in y_train: {np.unique(y_train)}")

    return X_train, y_train, X_val, y_val, X_test

def remove_duplicates(data, labels):
    data_reshaped = data.reshape(data.shape[0], -1)
    data_array = np.array([tuple(row) for row in data_reshaped])
    uniques, indices = np.unique(data_array, axis = 0, return_index = True) 
    indices = np.sort(indices)
    unique_data = data_reshaped[indices]
    labels = labels[indices]
    cleaned_data = unique_data.reshape(unique_data.shape[0], 300, 2)
    return cleaned_data, labels

def calculate_avg(data):
    X_avg = np.mean(data[:, :,0], axis = 1)
    y_avg = np.mean(data[:, :,1], axis = 1)

    return X_avg, y_avg

def calculate_std(data):
    X_std = np.std(data[:, :, 0], axis = 1)
    y_std = np.std(data[:, :, 1], axis = 1)
    return X_std, y_std


def remove_outliers(X_train, y_train, threshold=3):
    mean = np.mean(X_train, axis = 0)
    std_dev = np.std(X_train, axis = 0)

    std_dev = np.where(std_dev == 0, 1, std_dev)
    
    z_scores = (X_train - mean) / std_dev
    mask = np.abs(z_scores) < threshold
    mask = mask.all(axis = 2).all(axis = 1)
    return X_train[mask], y_train[mask]

def one_hot_encoding_to_numerical(labels):
    return np.argmax(labels, axis = 1)

def scale_data(fitting_data, data_to_scale):
    scaler = StandardScaler()
    fitting_data = fitting_data.reshape(-1,2)
    scaler = scaler.fit(fitting_data) 
    # :--p 
    # Data Leakage: It’s crucial to fit 
    # StandardScaler only on the training data 
    # and then apply the same transformation to 
    # the testing data to avoid data leakage and 
    # ensure model generalization.
    # :--p
    reshaped_data_to_scale = data_to_scale.reshape(-1, 2)
    scaled_data = scaler.transform(reshaped_data_to_scale)
    scaled_data = scaled_data.reshape(data_to_scale.shape)

    return scaled_data

def plot_trajectories(X, num_samples):
    for i in range(num_samples):
        plt.figure(figsize = (10,10))
        x_coords = X[i, :, 0]
        y_coords = X[i, :, 1]

        plt.plot(x_coords, y_coords, color = 'blue', label = f'Sample {i}')
        plt.scatter(x_coords[0], y_coords[0], color = 'green', s = 50, label = 'Start', zorder = 5)
        plt.scatter(x_coords[-1], y_coords[-1], color = 'red', s = 50, label = 'End', zorder = 5)
       
        plt.title('Particle Trajectory')
        plt.xlabel('x coordinate')
        plt.ylabel('y coordinate')
        plt.legend()
        plt.show()