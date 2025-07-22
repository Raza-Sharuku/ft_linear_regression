import matplotlib.pyplot as plt
import numpy as np
import json
import csv

def read_data(filename):
    """Read data from CSV file"""
    mileages = []
    prices = []
    
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            mileages.append(float(row['km']))
            prices.append(float(row['price']))
    return mileages, prices

def plot_data(mileages, prices, given_mileage, predicted_price):
    plt.scatter(mileages, prices, color='blue', marker='o', label='Training Data')
    plt.scatter(given_mileage, predicted_price, color='red', marker='x', label='Predicted Value')
    plt.xlabel('Mileage (km)')
    plt.ylabel('Price (yen)')
    plt.title('Car Price vs. Mileage')
    plt.legend()
    plt.show()

def main():
    mileages, prices = read_data('data.csv')
    plot_data(mileages, prices, 40000, 8000)

if __name__ == "__main__":
    main()