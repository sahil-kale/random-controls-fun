import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('system_id/data/system_id_data.csv')
df.columns = df.columns.str.strip()

# Set up a figure with 2 subplots (stacked vertically)
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot heater control actions
axs[0].plot(df['Elapsed Time'], df['Heater 1'], label='Heater 1')
axs[0].plot(df['Elapsed Time'], df['Heater 2'], label='Heater 2')
axs[0].set_ylabel('Heater Output (%)')
axs[0].set_title('Heater Control Actions')
axs[0].legend()
axs[0].grid(True)

# Plot temperature readings
axs[1].plot(df['Elapsed Time'], df['Temp 1 (degC)'], label='Temperature 1')
axs[1].plot(df['Elapsed Time'], df['Temp 2 (degC)'], label='Temperature 2')
axs[1].set_xlabel('Elapsed Time (s)')
axs[1].set_ylabel('Temperature (°C)')
axs[1].set_title('Temperature Measurements')
axs[1].legend()
axs[1].grid(True)

# Improve layout
plt.tight_layout()
# save the figure
plt.savefig('system_id/data/system_id_data_plot.png', dpi=300)
plt.show()
