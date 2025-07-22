import json
import matplotlib.pyplot as plt
import csv

def load_parameters():
    """Load trained parameters"""
    try:
        with open('model.json', 'r') as file:
            params = json.load(file)
            return params['theta0'], params['theta1']
    except FileNotFoundError:
        # If no trained model is found, use theta0=0 and theta1=0
        print("Warning: No trained model found. Using theta0=0, theta1=0.")
        return 0.0, 0.0

def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)

def plot_data(mileages, prices, given_mileage, predicted_price, theta0, theta1):
    import numpy as np
    plt.scatter(mileages, prices, color='blue', marker='o', label='Training Data')
    plt.scatter(given_mileage, predicted_price, color='red', marker='x', label='Predicted Value')
    # 回帰直線の描画
    x_line = np.linspace(min(mileages), max(mileages), 100)
    y_line = theta0 + theta1 * x_line
    plt.plot(x_line, y_line, color='green', label='Regression Line')
    plt.xlabel('Mileage (km)')
    plt.ylabel('Price (yen)')
    plt.title('Car Price vs. Mileage')
    plt.legend()
    plt.show()

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

def main():
    # Load trained parameters
    theta0, theta1 = load_parameters()
    mileages, prices = read_data('data.csv')
    
    print("=== Car Price Prediction Program ===")
    print("Enter mileage to predict price.")
    print("To quit, enter 'quit'.")
    print()
    
    while True:
        try:
            user_input = input("Enter mileage (km): ")
            
            # End condition
            if user_input.lower() == 'quit':
                print("Program terminated.")
                break
            
            # Number conversion
            mileage = float(user_input)
            
            # Negative value check
            if mileage < 0:
                print("Mileage must be 0 or greater.")
                continue
            
            # Price prediction
            predicted_price = estimate_price(mileage, theta0, theta1)
            
            # Result display
            print(f"Predicted price: {predicted_price:.2f}")
            print()

            # Plot data (regression line also displayed)
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