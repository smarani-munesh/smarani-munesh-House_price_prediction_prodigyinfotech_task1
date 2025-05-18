# House Price Prediction with GUI using Tkinter (Advanced UI)
# Author: [yamana smarani]
# Internship Project: Predict house prices using a GUI interface and ML

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tkinter as tk
from tkinter import messagebox, ttk

# Step 1: Load Datasets
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")

# Step 2: Feature Engineering
train_df['TotalBathrooms'] = train_df['FullBath'] + 0.5 * train_df['HalfBath']
test_df['TotalBathrooms'] = test_df['FullBath'] + 0.5 * test_df['HalfBath']

# Step 3: Select Features
selected_features = ['GrLivArea', 'BedroomAbvGr', 'TotalBathrooms', 'OverallQual', 'GarageCars', 'TotalBsmtSF']
X_train = train_df[selected_features]
y_train = train_df['SalePrice']
X_test = test_df[selected_features].fillna(X_train.mean())

# Step 4: Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Save Submission File
y_pred = model.predict(X_test)
submission = test_df[['Id']].copy()
submission['SalePrice'] = y_pred
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv submitted successfully to output folder!")

# Step 6: Evaluation and Graph
y_train_pred = model.predict(X_train)
rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
mae = mean_absolute_error(y_train, y_train_pred)
print(f"📊 Training RMSE: {rmse:.2f}")
print(f"📊 Training MAE: {mae:.2f}")

# Save the graph for later viewing
def show_graph():
    plt.figure(figsize=(8, 5))
    plt.scatter(y_train, y_train_pred, alpha=0.6, color='blue')
    plt.xlabel("Actual SalePrice")
    plt.ylabel("Predicted SalePrice")
    plt.title("Actual vs Predicted House Prices")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Step 7: Create GUI for User Input
def predict_price():
    try:
        sqft = float(entry_sqft.get())
        beds = int(entry_beds.get())
        baths = float(entry_baths.get())
        qual = int(entry_qual.get())
        garage = int(entry_garage.get())
        basement = float(entry_basement.get())

        user_input = np.array([[sqft, beds, baths, qual, garage, basement]])
        predicted = model.predict(user_input)[0]
        messagebox.showinfo("Predicted Price", f"Estimated House Price: ${predicted:,.2f}")

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers in all fields.")

# GUI window
root = tk.Tk()
root.title(" House Price Predictor ")
root.geometry("500x500")
root.configure(bg="#f0f4f8")

# Styling labels and entries
def create_input(label_text):
    tk.Label(root, text=label_text, font=("Arial", 12, "bold"), bg="#f0f4f8").pack(pady=(10, 0))
    entry = tk.Entry(root, font=("Arial", 12))
    entry.pack(pady=(0, 5))
    return entry

header = tk.Label(root, text="Enter House Details", font=("Helvetica", 16, "bold"), bg="#f0f4f8", fg="#333")
header.pack(pady=10)

entry_sqft = create_input("Square Footage (LivingArea):")
entry_beds = create_input("Number of Bedrooms:")
entry_baths = create_input("Total Bathrooms (e.g., 1.5):")
entry_qual = create_input("Overall Quality (1-10):")
entry_garage = create_input("Garage Capacity (Cars):")
entry_basement = create_input("Basement Area (sq ft):")

# Optional: Dropdown for Neighborhood
tk.Label(root, text="Neighborhood:", font=("Arial", 12, "bold"), bg="#f0f4f8").pack(pady=(10, 0))
neighborhoods = sorted(train_df['Neighborhood'].dropna().unique())
neighborhood_dropdown = ttk.Combobox(root, values=neighborhoods, font=("Arial", 12))
neighborhood_dropdown.pack(pady=(0, 10))
neighborhood_dropdown.set("Select Neighborhood")

# Predict Button
tk.Button(root, text=" Predict Price", command=predict_price, font=("Arial", 12, "bold"), bg="#007acc", fg="white").pack(pady=10)

# Show Graph Button
tk.Button(root, text=" Show Model Accuracy Graph", command=show_graph, font=("Arial", 11), bg="#4caf50", fg="white").pack(pady=5)

root.mainloop()