import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Load the data from the file
file_path = 'proj_step_3.xlsx'  # Replace with your file path
data = pd.read_excel(file_path)

# ----------- Phase 1 Analysis ------------
# Filter Phase 1 data and select samples
data_phase1 = data[data['Phase'] == 'Phase 1']  # Filter Phase 1 data
samples_phase1 = data_phase1[(data_phase1['Hour'] >= 8) & (data_phase1['Minute'] <= 5)]  # First 5 minutes of each hour
samples_phase1 = samples_phase1.head(90)  # Limit to 90 samples

# Step 1: Calculate sample averages and ranges for Phase 1
samples_phase1.loc[:, 'Sample Mean'] = samples_phase1[[1, 2, 3, 4]].mean(axis=1)
samples_phase1.loc[:, 'Sample Range'] = samples_phase1[[1, 2, 3, 4]].max(axis=1) - samples_phase1[[1, 2, 3, 4]].min(axis=1)

# Step 2: Calculate control limits for Phase 1
A2 = 0.729  # SPC constant for n=4
D3 = 0      # SPC constant for lower range limit (n=4)
D4 = 2.282  # SPC constant for upper range limit (n=4)

X_bar = samples_phase1['Sample Mean'].mean()
R_bar = samples_phase1['Sample Range'].mean()

UCL_X = X_bar + A2 * R_bar
LCL_X = X_bar - A2 * R_bar
UCL_R = R_bar * D4
LCL_R = R_bar * D3

print(f"\nPhase 1 Control Limits for X̄: UCL = {UCL_X:.2f}, LCL = {LCL_X:.2f}, Mean = {X_bar:.2f}")
print(f"Phase 1 Control Limits for R: UCL = {UCL_R:.2f}, LCL = {LCL_R:.2f}, Mean Range = {R_bar:.2f}")

# Step 3: Plot control charts for Phase 1
# X̄ Chart
plt.figure(figsize=(12, 6))
plt.plot(samples_phase1.index, samples_phase1['Sample Mean'], marker='o', label='Sample Means')
plt.axhline(y=UCL_X, color='r', linestyle='--', label='UCL (Mean)')
plt.axhline(y=LCL_X, color='r', linestyle='--', label='LCL (Mean)')
plt.axhline(y=X_bar, color='g', linestyle='-', label='Mean')
plt.title('X̄ Chart for Phase 1')
plt.xlabel('Sample Number')
plt.ylabel('Sample Mean')
plt.legend()
plt.savefig('xbar_chart_phase1.png')
plt.show()

# R Chart
plt.figure(figsize=(12, 6))
plt.plot(samples_phase1.index, samples_phase1['Sample Range'], marker='o', label='Sample Ranges')
plt.axhline(y=UCL_R, color='r', linestyle='--', label='UCL (Range)')
plt.axhline(y=LCL_R, color='r', linestyle='--', label='LCL (Range)')
plt.axhline(y=R_bar, color='g', linestyle='-', label='Mean (Range)')
plt.title('R Chart for Phase 1')
plt.xlabel('Sample Number')
plt.ylabel('Sample Range')
plt.legend()
plt.savefig('r_chart_phase1.png')
plt.show()

# Step 4: Material loss for Phase 1
threshold = 70

# Corrected calculation for loss_below_phase1
loss_below_phase1 = samples_phase1[[1, 2, 3, 4]].apply(
    lambda col: col.apply(lambda x: x if x < threshold else 0)
).sum().sum()

# Corrected calculation for loss_above_phase1
loss_above_phase1 = samples_phase1[[1, 2, 3, 4]].apply(
    lambda col: col.apply(lambda x: max(x - threshold, 0))
).sum().sum()

# Total material loss for Phase 1
total_material_loss_phase1 = loss_below_phase1 + loss_above_phase1
total_material_loss_phase1_kg = total_material_loss_phase1 / 1000  # Convert to kilograms
cost_per_kg = 30  # Assuming cost of raw material is 30 per kg
total_material_loss_cost_phase1 = total_material_loss_phase1_kg * cost_per_kg

print(f"Phase 1 Material Loss Below Threshold (kg): {loss_below_phase1 / 1000:.2f}")
print(f"Phase 1 Material Loss Above Threshold (kg): {loss_above_phase1 / 1000:.2f}")
print(f"Phase 1 Total Material Loss (kg): {total_material_loss_phase1_kg:.2f}")
print(f"Phase 1 Total Material Loss Cost: {total_material_loss_cost_phase1:.2f} Shekels")
# Step 5: Correlation analysis for Phase 1
correlations = samples_phase1[['% Fat', 'Velocity', 'Sample Mean']].corr()
print("\nPhase 1 Correlation Matrix:")
print(correlations)

# Check if relationships are strong
strong_relationships = (correlations['Sample Mean']['% Fat'] > 0.7 or correlations['Sample Mean']['% Fat'] < -0.7) and \
                       (correlations['Sample Mean']['Velocity'] > 0.7 or correlations['Sample Mean']['Velocity'] < -0.7)

if strong_relationships:
    print("\nStrong relationships found between predictors and Sample Mean. Proceeding with optimization.")

    # Regression with Confidence Intervals
    X = samples_phase1[['% Fat', 'Velocity']]
    y = samples_phase1['Sample Mean']

    # Add constant for statsmodels
    X_with_const = sm.add_constant(X)
    ols_model = sm.OLS(y, X_with_const).fit()

    # Display regression summary
    print("\nRegression Summary:")
    print(ols_model.summary())

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

    model = LinearRegression()
    model.fit(X, y)
    best_fat, best_velocity, min_mean = optimize_parameters()

    print(f"\nOptimized % Fat: {best_fat:.2f}, Optimized Velocity: {best_velocity:.2f}, Minimum Predicted Mean: {min_mean:.2f}")

# ----------- Phase 2 Analysis ------------
# ----------- Phase 2 Analysis with Dynamic Fat Levels ------------

# Filter Phase 2 data
# ----------- Phase 2 Analysis ------------

# Filter Phase 2 data and select samples
data_phase2 = data[data['Phase'] == 'Phase 2']  # Filter Phase 2 data
samples_phase2 = data_phase2[(data_phase2['Hour'] >= 8) & (data_phase2['Minute'] <= 5)]  # First 5 minutes of each hour
samples_phase2 = samples_phase2.head(90)  # Limit to 90 samples

# Step 1: Calculate sample averages and ranges for Phase 2
samples_phase2.loc[:, 'Sample Mean'] = samples_phase2[[1, 2, 3, 4]].mean(axis=1)
samples_phase2.loc[:, 'Sample Range'] = samples_phase2[[1, 2, 3, 4]].max(axis=1) - samples_phase2[[1, 2, 3, 4]].min(axis=1)

# Step 2: Calculate control limits for Phase 2
A2 = 0.729  # SPC constant for n=4
D3 = 0      # SPC constant for lower range limit (n=4)
D4 = 2.282  # SPC constant for upper range limit (n=4)

X_bar_phase2 = samples_phase2['Sample Mean'].mean()
R_bar_phase2 = samples_phase2['Sample Range'].mean()

UCL_X_phase2 = X_bar_phase2 + A2 * R_bar_phase2
LCL_X_phase2 = X_bar_phase2 - A2 * R_bar_phase2
UCL_R_phase2 = R_bar_phase2 * D4
LCL_R_phase2 = R_bar_phase2 * D3

print(f"\nPhase 2 Control Limits for X̄: UCL = {UCL_X_phase2:.2f}, LCL = {LCL_X_phase2:.2f}, Mean = {X_bar_phase2:.2f}")
print(f"Phase 2 Control Limits for R: UCL = {UCL_R_phase2:.2f}, LCL = {LCL_R_phase2:.2f}, Mean Range = {R_bar_phase2:.2f}")

# Plot control charts for Phase 2
# X̄ Chart
plt.figure(figsize=(12, 6))
plt.plot(samples_phase2.index, samples_phase2['Sample Mean'], marker='o', label='Sample Means')
plt.axhline(y=UCL_X_phase2, color='r', linestyle='--', label='UCL (Mean)')
plt.axhline(y=LCL_X_phase2, color='r', linestyle='--', label='LCL (Mean)')
plt.axhline(y=X_bar_phase2, color='g', linestyle='-', label='Mean')
plt.title('X̄ Chart for Phase 2')
plt.xlabel('Sample Number')
plt.ylabel('Sample Mean')
plt.legend()
plt.savefig('xbar_chart_phase2.png')
plt.show()

# R Chart
plt.figure(figsize=(12, 6))
plt.plot(samples_phase2.index, samples_phase2['Sample Range'], marker='o', label='Sample Ranges')
plt.axhline(y=UCL_R_phase2, color='r', linestyle='--', label='UCL (Range)')
plt.axhline(y=LCL_R_phase2, color='r', linestyle='--', label='LCL (Range)')
plt.axhline(y=R_bar_phase2, color='g', linestyle='-', label='Mean (Range)')
plt.title('R Chart for Phase 2')
plt.xlabel('Sample Number')
plt.ylabel('Sample Range')
plt.legend()
plt.savefig('r_chart_phase2.png')
plt.show()
# Step 6: Material loss for Phase 2

# Corrected calculation for loss_below_phase2
loss_below_phase2 = samples_phase2[[1, 2, 3, 4]].apply(
    lambda col: col.apply(lambda x: x if x < threshold else 0)
).sum().sum()

# Corrected calculation for loss_above_phase2
loss_above_phase2 = samples_phase2[[1, 2, 3, 4]].apply(
    lambda col: col.apply(lambda x: max(x - threshold, 0))
).sum().sum()

# Total material loss for Phase 2
total_material_loss_phase2 = loss_below_phase2 + loss_above_phase2
total_material_loss_phase2_kg = total_material_loss_phase2 / 1000  # Convert to kilograms
total_material_loss_cost_phase2 = total_material_loss_phase2_kg * cost_per_kg

print(f"\nPhase 2 Material Loss Below Threshold (kg): {loss_below_phase2 / 1000:.2f}")
print(f"Phase 2 Material Loss Above Threshold (kg): {loss_above_phase2 / 1000:.2f}")
print(f"Phase 2 Total Material Loss (kg): {total_material_loss_phase2_kg:.2f}")
print(f"Phase 2 Total Material Loss Cost: {total_material_loss_cost_phase2:.2f} Shekels")

# Savings calculations
savings = total_material_loss_cost_phase1 - total_material_loss_cost_phase2
percentage_savings = (savings / total_material_loss_cost_phase1) * 100  # Percentage savings

print(f"\nSavings in Material Loss Cost: {savings:.2f} Shekels")
print(f"Percentage Savings: {percentage_savings:.2f}%")