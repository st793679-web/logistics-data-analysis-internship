# Logistics Data Analysis - Week 1

import pandas as pd
import matplotlib.pyplot as plt

# Sample logistics data
data = {
    "Order_ID": [101, 102, 103, 104, 105, 106, 107, 108],
    "Delivery_Days": [3, 5, 2, 7, 4, 6, 3, 5],
    "Distance_KM": [25, 60, 15, 100, 40, 80, 30, 55],
    "Transport_Cost": [200, 450, 150, 800, 300, 650, 250, 400],
    "Delivery_Status": [
        "On Time", "Delayed", "On Time", "Delayed",
        "On Time", "Delayed", "On Time", "On Time"
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Display the dataset
print("LOGISTICS DATA")
print(df)

# Basic information
print("\nDATA INFORMATION")
print(df.info())

# Statistical summary
print("\nSTATISTICAL SUMMARY")
print(df.describe())

# Average delivery time
average_delivery = df["Delivery_Days"].mean()
print("\nAverage Delivery Time:", average_delivery, "days")

# Average transport cost
average_cost = df["Transport_Cost"].mean()
print("Average Transport Cost: ₹", average_cost)

# Count delivery status
print("\nDELIVERY STATUS")
print(df["Delivery_Status"].value_counts())

# Find delayed orders
delayed_orders = df[df["Delivery_Status"] == "Delayed"]

print("\nDELAYED ORDERS")
print(delayed_orders)

# Visualization: Delivery days
plt.figure(figsize=(8, 5))
plt.bar(df["Order_ID"].astype(str), df["Delivery_Days"])
plt.xlabel("Order ID")
plt.ylabel("Delivery Days")
plt.title("Delivery Time by Order")
plt.show()

# Visualization: Transport cost
plt.figure(figsize=(8, 5))
plt.bar(df["Order_ID"].astype(str), df["Transport_Cost"])
plt.xlabel("Order ID")
plt.ylabel("Transport Cost (₹)")
plt.title("Transport Cost by Order")
plt.show()