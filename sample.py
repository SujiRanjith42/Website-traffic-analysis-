import pandas as pd
import matplotlib.pyplot as plt

# Sample data (you can replace this with your actual data)
data = {
    "Row": [1, 2, 3, 4, 5],
    "Day": ["Mon", "Tue", "Wed", "Thu", "Fri"],
    "Day.Of.Week": [1, 2, 3, 4, 5],
    "Date": ["2023-10-01", "2023-10-02", "2023-10-03", "2023-10-04", "2023-10-05"],
    "Page.Loads": [100, 150, 130, 200, 180],
    "Unique.Visits": [80, 120, 100, 150, 140],
    "First.Time.Visits": [30, 40, 35, 50, 45],
    "Returning.Visits": [50, 80, 65, 100, 95]
}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Plotting
plt.figure(figsize=(10, 6))

# You can customize the graph based on your preference
plt.plot(df["Date"], df["Page.Loads"], marker='o', label='Page Loads')
plt.plot(df["Date"], df["Unique.Visits"], marker='o', label='Unique Visits')
plt.plot(df["Date"], df["First.Time.Visits"], marker='o', label='First Time Visits')
plt.plot(df["Date"], df["Returning.Visits"], marker='o', label='Returning Visits')

plt.xlabel('Date')
plt.ylabel('Count')
plt.title('Website Traffic Analysis')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.legend()
plt.tight_layout()

# Save the plot to a file (optional)
plt.savefig('website_traffic.png')

# Show the plot
plt.show()
