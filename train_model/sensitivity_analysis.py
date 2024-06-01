import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Set global model parameters
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### Sensitivity Analysis using fractional increments of the maximum and minimum
### of each input variable to assess change over a baseline
def perform_sensitivity_analysis(net, features, rf_predicted, selected_features=None):
    # Calculate a baseline with all features included
    x_array = [np.mean(rf_predicted[feature]) for feature in features]
    x_array = torch.from_numpy(np.array(x_array)).to(device).unsqueeze(0)
    baseline = net.predict(x_array.float().data.cpu().numpy())

    if selected_features is None:
        selected_features = features
    sf = pd.DataFrame({'Variable': selected_features})

    increment = 0.05
    for k in range(int(1 / increment)):
        positive_col = '+' + str(round((k + 1) * increment, 2))
        negative_col = '-' + str(round((k + 1) * increment, 2))
        positive_sensitivities = []
        negative_sensitivities = []

        for j, feature in enumerate(selected_features):
            # Adjust features one by one
            x_array_positive = [np.mean(rf_predicted[feat]) + ((k + 1) * increment * (
                        np.max(rf_predicted[feat]) - np.mean(rf_predicted[feat]))) if feat == feature else np.mean(
                rf_predicted[feat]) for feat in features]

            x_array_negative = [np.mean(rf_predicted[feat]) - ((k + 1) * increment * (
                        np.mean(rf_predicted[feat]) - np.min(rf_predicted[feat]))) if feat == feature else np.mean(
                rf_predicted[feat]) for feat in features]

            # Convert to tensor and make predictions
            x_array_positive = torch.from_numpy(np.array(x_array_positive)).to(device).unsqueeze(0)
            x_array_negative = torch.from_numpy(np.array(x_array_negative)).to(device).unsqueeze(0)
            adjustment_positive = net.predict(x_array_positive.float().data.cpu().numpy())
            adjustment_negative = net.predict(x_array_negative.float().data.cpu().numpy())

            # Calculate sensitivities
            positive_sensitivities.append(abs((adjustment_positive - baseline) / baseline)[0][0])
            negative_sensitivities.append(abs((adjustment_negative - baseline) / baseline)[0][0])

        # Store sensitivities in the DataFrame
        sf[positive_col] = positive_sensitivities
        sf[negative_col] = negative_sensitivities

    return sf


def plot_sensitivities(sf, columns_to_plot):
    indices_names = sf.loc[sf['Variable'].isin(columns_to_plot)].index
    sf = sf[sorted(sf.columns[1:], key=float)]

    # Extracting the sensitivity values for plotting
    sensitivity_values = sf.loc[indices_names]
    x_ticks = np.array([float(col) for col in sf.columns])

    # Plotting
    plt.figure(figsize=(10, 6))
    for i in range(len(columns_to_plot)):
        plt.plot(x_ticks, sensitivity_values.iloc[i], label=columns_to_plot[i])

    plt.xlabel('Fractional Increments')
    plt.ylabel('Sensitivity')
    # plt.ylim(0, 1.2)
    plt.legend()
    plt.grid(True)
    plt.show()