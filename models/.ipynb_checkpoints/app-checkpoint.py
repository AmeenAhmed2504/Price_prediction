import tkinter as tk
from tkinter import ttk
import numpy as np
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the ML model
model = joblib.load('title_cat123.pkl')

vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Function to get the selected values and pass them to the ML model
def predict_price():
    title = title_entry.get()
    category_1_value = category_values[category1_var.get()]
    category_2_value = category_values[category2_var.get()]
    category_3_value = category_values[category3_var.get()]
    
    # Prepare data for ML model
    input_data = [{
        'title': title,
        'category_1': category_1_value,
        'category_2': category_2_value,
        'category_3': category_3_value
    }]
    
    # Extract text data from input_data
    text_data = [data['title'] for data in input_data]
    
    # Transform text data using TF-IDF vectorizer
    X_text_input_transformed = vectorizer.transform(text_data)
    
    # Make predictions
    predicted_prices = model.predict(X_text_input_transformed)
    
    # Print predicted prices
    for i, price in enumerate(predicted_prices):
        print(f"Predicted Price for input {i+1}:", price)

# Create the main window
root = tk.Tk()
root.title("Title and Category Prediction")

# Define category values
category_values = {
    '-3': -3,
    '-57': -57,
    '-226': -226
}

# Create a label for the title
title_label = ttk.Label(root, text="Title:")
title_label.grid(row=0, column=0, padx=5, pady=5)

# Create an entry for entering the title
title_entry = ttk.Entry(root)
title_entry.grid(row=0, column=1, padx=5, pady=5)

# Create dropdowns for selecting categories
category1_label = ttk.Label(root, text="Category 1:")
category1_label.grid(row=1, column=0, padx=5, pady=5)

category1_var = tk.StringVar()
category1_dropdown = ttk.Combobox(root, textvariable=category1_var, values=list(category_values.keys()))
category1_dropdown.grid(row=1, column=1, padx=5, pady=5)
category1_dropdown.current(0)

category2_label = ttk.Label(root, text="Category 2:")
category2_label.grid(row=2, column=0, padx=5, pady=5)

category2_var = tk.StringVar()
category2_dropdown = ttk.Combobox(root, textvariable=category2_var, values=list(category_values.keys()))
category2_dropdown.grid(row=2, column=1, padx=5, pady=5)
category2_dropdown.current(0)

category3_label = ttk.Label(root, text="Category 3:")
category3_label.grid(row=3, column=0, padx=5, pady=5)

category3_var = tk.StringVar()
category3_dropdown = ttk.Combobox(root, textvariable=category3_var, values=list(category_values.keys()))
category3_dropdown.grid(row=3, column=1, padx=5, pady=5)
category3_dropdown.current(0)

# Create a button to trigger prediction
predict_button = ttk.Button(root, text="Predict", command=predict_price)
predict_button.grid(row=4, columnspan=2, padx=5, pady=5)

root.mainloop()
