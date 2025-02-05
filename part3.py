import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the data from the file
file_path = 'proj_step_3.xlsx'  # Replace with your file path
data = pd.read_excel(file_path)

# Filter Phase 1 data and select samples
data_phase1 = data[data['Phase'] == 'Phase 1']  # Filter Phase 1 data
samples = data_phase1[(data_phase1['Hour'] >= 8) & (data_phase1['Minute'] <= 5)]  # Select first 5 minutes of each hour
samples = samples.head(90)  # Limit to 90 samples

# Step 1: Calculate sample averages and ranges
samples.loc[:, 'Sample Mean'] = samples[[1, 2, 3, 4]].mean(axis=1)
samples.loc[:, 'Sample Range'] = samples[[1, 2, 3, 4]].max(axis=1) - samples[[1, 2, 3, 4]].min(axis=1)
print(samples['Sample Mean'].head())  # Debugging: Verify first few sample means
print(samples['Sample Range'].head())  # Debugging: Verify first few sample ranges

# Step 2: Calculate overall averages and ranges
X_bar = samples['Sample Mean'].mean()
R_bar = samples['Sample Range'].mean()
print(f"X̄ (Mean of Sample Means): {X_bar:.2f}")
print(f"R̄ (Mean of Sample Ranges): {R_bar:.2f}")

# Constants for n=4 (from SPC tables)
A2 = 0.729
D3 = 0  # For n=4, D3 = 0
D4 = 2.282

# Calculate control limits for the mean chart
UCL_X = X_bar + A2 * R_bar
LCL_X = X_bar - A2 * R_bar

# Calculate control limits for the range chart
UCL_R = D4 * R_bar
LCL_R = D3 * R_bar

# Step 3: Plot control charts
# Mean chart
plt.figure(figsize=(12, 6))
plt.plot(samples.index, samples['Sample Mean'], marker='o', label='Sample Means')
plt.axhline(y=UCL_X, color='r', linestyle='--', label='UCL (Mean)')
plt.axhline(y=LCL_X, color='r', linestyle='--', label='LCL (Mean)')
plt.axhline(y=X_bar, color='g', linestyle='-', label='Mean')
plt.title('X̄ Chart')
plt.xlabel('Sample Number')
plt.ylabel('Sample Mean')
plt.legend()
plt.savefig('xbar_chart.png')

# Range chart
plt.figure(figsize=(12, 6))
plt.plot(samples.index, samples['Sample Range'], marker='o', label='Sample Ranges')
plt.axhline(y=UCL_R, color='r', linestyle='--', label='UCL (Range)')
plt.axhline(y=LCL_R, color='r', linestyle='--', label='LCL (Range)')
plt.axhline(y=R_bar, color='g', linestyle='-', label='Mean (Range)')
plt.title('R Chart')
plt.xlabel('Sample Number')
plt.ylabel('Sample Range')
plt.legend()
plt.savefig('r_chart.png')

# Step 4: Analyze relationships between % Fat, Velocity, and Sample Means
X = samples[['% Fat', 'Velocity']]
y = samples['Sample Mean']

# Perform linear regression
model = LinearRegression()
model.fit(X, y)

# Optimize parameters to minimize sample mean
def optimize_parameters():
    best_fat, best_velocity = None, None
    min_mean = float('inf')
    for fat in np.linspace(data['% Fat'].min(), data['% Fat'].max(), 50):
        for velocity in np.linspace(data['Velocity'].min(), data['Velocity'].max(), 50):
            input_df = pd.DataFrame({'% Fat': [fat], 'Velocity': [velocity]})
            predicted_mean = model.predict(input_df)[0]
            if predicted_mean < min_mean:
                min_mean = predicted_mean
                best_fat, best_velocity = fat, velocity
    return best_fat, best_velocity, min_mean

best_fat, best_velocity, min_mean = optimize_parameters()
print(f"Optimized % Fat: {best_fat:.2f}, Optimized Velocity: {best_velocity:.2f}, Minimum Predicted Mean: {min_mean:.2f}")

# Step 5: Recalculate control limits and material loss for optimized parameters
optimized_samples = data_phase1[(data_phase1['% Fat'] == best_fat) & (data_phase1['Velocity'] == best_velocity)].head(90)
optimized_samples.loc[:, 'Sample Mean'] = optimized_samples[[1, 2, 3, 4]].mean(axis=1)
optimized_samples.loc[:, 'Sample Range'] = optimized_samples[[1, 2, 3, 4]].max(axis=1) - optimized_samples[[1, 2, 3, 4]].min(axis=1)

# Recalculate averages
X_bar_optimized = optimized_samples['Sample Mean'].mean()
R_bar_optimized = optimized_samples['Sample Range'].mean()

# Recalculate control limits for optimized conditions
UCL_X_optimized = X_bar_optimized + A2 * R_bar_optimized
LCL_X_optimized = X_bar_optimized - A2 * R_bar_optimized
UCL_R_optimized = D4 * R_bar_optimized
LCL_R_optimized = D3 * R_bar_optimized

print(f"Optimized Control Limits for X̄: UCL = {UCL_X_optimized:.2f}, LCL = {LCL_X_optimized:.2f}, Mean = {X_bar_optimized:.2f}")
print(f"Optimized Control Limits for R: UCL = {UCL_R_optimized:.2f}, LCL = {LCL_R_optimized:.2f}, Mean Range = {R_bar_optimized:.2f}")

# Step 6: Calculate material loss under optimized conditions
threshold = 70
material_loss = optimized_samples.loc[optimized_samples[[1, 2, 3, 4]].min(axis=1) < threshold, [1, 2, 3, 4]].sum().sum()
material_loss_kg = material_loss / 1000  # Convert to kilograms
cost_per_kg = 30  # Assuming cost of raw material is 30 per kg
total_material_loss_cost = material_loss_kg * cost_per_kg
print(f"Total Material Loss (kg): {material_loss_kg:.2f}")
print(f"Total Material Loss Cost: {total_material_loss_cost:.2f} Shekels")

# Step 7: Analyze Phase 2 Data with Optimized Parameters
data_phase2 = data[data['Phase'] == 'Phase 2']  # Filter Phase 2 data
optimized_phase2 = data_phase2[(data_phase2['% Fat'] == best_fat) & (data_phase2['Velocity'] == best_velocity)]

# Calculate sample averages and ranges for Phase 2
optimized_phase2 = optimized_phase2.assign(
    Sample_Mean=optimized_phase2[[1, 2, 3, 4]].mean(axis=1),
    Sample_Range=optimized_phase2[[1, 2, 3, 4]].max(axis=1) - optimized_phase2[[1, 2, 3, 4]].min(axis=1)
)

# Recalculate averages for Phase 2
X_bar_phase2 = optimized_phase2['Sample_Mean'].mean()
R_bar_phase2 = optimized_phase2['Sample_Range'].mean()

# Recalculate control limits for Phase 2
UCL_X_phase2 = X_bar_phase2 + A2 * R_bar_phase2
LCL_X_phase2 = X_bar_phase2 - A2 * R_bar_phase2
UCL_R_phase2 = R_bar_phase2 * D4
LCL_R_phase2 = R_bar_phase2 * D3

# Print Phase 2 control limits
print(f"Phase 2 Control Limits for X̄: UCL = {UCL_X_phase2:.2f}, LCL = {LCL_X_phase2:.2f}, Mean = {X_bar_phase2:.2f}")
print(f"Phase 2 Control Limits for R: UCL = {UCL_R_phase2:.2f}, LCL = {LCL_R_phase2:.2f}, Mean Range = {R_bar_phase2:.2f}")
