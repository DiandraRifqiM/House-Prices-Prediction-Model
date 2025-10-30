from model import linreg, preprocessor
import pandas as pd

# Input Params
area = int(input("House Area (m²): "))
bedrooms = int(input("Number of Bedrooms: "))
bathrooms = int(input("Number of Bathrooms: "))
stories = int(input("Number of Stories: "))
parking = int(input("Number of Vehicle Parking Area: "))
mainroad = input("Mainroad (yes/no) ? ")
guestroom = input("Guestroom (yes/no) ? ")
basement = input("Basement (yes/no) ? ")
hotwaterheating = input("Hot Water Heater (yes/no) ? ")
airconditioning = input("Airconditioning (yes/no) ? ")
prefarea = input("Prefarea (yes/no) ? ")
furnishingstatus = input("Furnishing Status (furnished/semi-furnished/unfurnished) ? ")

# Input Type Datas
inputs = {
    "area":[area],
    "bedrooms":[bedrooms],
    "bathrooms":[bathrooms],
    "stories":[stories],
    "mainroad":[mainroad],
    "guestroom":[guestroom],
    "basement":[basement],
    "hotwaterheating":[hotwaterheating],
    "airconditioning":[airconditioning],
    "parking":[parking],
    "prefarea":[prefarea],
    "furnishingstatus":[furnishingstatus]
}

df = pd.DataFrame(inputs)


# Processeing
X_processed = preprocessor.transform(df)

# House Price Predict Result
pricePred = linreg.predict(X_processed)

print(f"Best Price We Offer: ${pricePred[0]:,.2f}")

