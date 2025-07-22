import json
import matplotlib.pyplot as plt
import csv
import numpy as np

def load_parameters():
    try:
        with open('model.json', 'r') as file:
            params = json.load(file)
            return params['theta0'], params['theta1']
    except FileNotFoundError:
        # If no trained model is found, set default values theta0=0 and theta1=0
        return 0.0, 0.0

def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)

def plot_data(mileages, prices, given_mileage, predicted_price, theta0, theta1):
    plt.scatter(mileages, prices, color='blue', marker='o', label='Training Data')
    plt.scatter(given_mileage, predicted_price, color='red', marker='x', label='Predicted Value')
    x_line = np.linspace(min(mileages), max(mileages), 100)
    y_line = theta0 + theta1 * x_line
    plt.plot(x_line, y_line, color='green', label='Regression Line')
    plt.xlabel('Mileage (km)')
    plt.ylabel('Price (yen)')
    plt.title('Car Price vs. Mileage')
    plt.legend()
    plt.show()

def read_data(filename):
    mileages = []
    prices = []
    
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            mileages.append(float(row['km']))
            prices.append(float(row['price']))
    return mileages, prices

def main():
    theta0, theta1 = load_parameters()
    mileages, prices = read_data('data.csv')
    
    print("=== Car Price Prediction Program ===")
    print("Enter mileage to predict price.")
    print("To quit, enter 'quit'.\n")
    
    while True:
        try:
            user_input = input("Enter mileage (km): ")
            
            # Finish condition
            if user_input.lower() == 'quit':
                print("Program terminated.")
                break
            
            mileage = float(user_input)
            
            # Negative value check
            if mileage < 0:
                print("Mileage must be 0 or greater.")
                continue
            
            # Price prediction
            predicted_price = estimate_price(mileage, theta0, theta1)
            
            # Display result
            print(f"Predicted price: {predicted_price:.2f}\n")

            # Plot data
            plot_data(mileages, prices, mileage, predicted_price, theta0, theta1)
            
        except ValueError:
            print("Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nProgram terminated.")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()