import pandas as pd

cars_sales = pd.read_csv("data/car_sales_data.csv")

def select_cars_simple(type, name):
    if name == None or name == '':    return "[]"
    filtered_col = cars_sales[type].str.strip().str.lower()
    selected_cars = cars_sales[filtered_col == name.lower()]
    return selected_cars.to_json(orient="records")

def select_cars(type1, name1, type2, name2):
    if name1 == None or name1 == '':
        return select_cars_simple(type2, name2)
    elif name2 == None or name2 == '':
        return select_cars_simple(type1, name1)
    filtered_col = cars_sales[type1].str.strip().str.lower()
    selected_cars = cars_sales[filtered_col == name1.lower()]
    filtered_col2 = selected_cars[type2].str.strip().str.lower()
    selected_cars2 = selected_cars[filtered_col2 == name2.lower()]
    return selected_cars2.to_json(orient="records")