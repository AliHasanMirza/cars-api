import json

import pandas as pd
import numpy as np
from flask import Flask, request
import os
from loan_model import load_data, evaluate_creditworthiness

inventory = dict()

credit_model = None

def load_credit_model():
    global credit_model
    credit_model = load_data()


def load_inventory(inventory_df:pd.DataFrame):
    global inventory
    for an_item in inventory_df.to_dict(orient='records'):
        make = an_item['make']
        model = an_item['model']
        year = an_item['year']
        if (make_entry:=inventory.get(make)) is None:
            make_entry = dict()
            inventory[make] = make_entry
        if (model_entry:=inventory[make].get(model)) is None:
            model_entry = dict()
            make_entry[model] = model_entry
        if (year_entry:=inventory[make][model]) is None:
            year_entry = dict()
            model_entry[year] = year_entry
        inventory[make][model][year] = an_item


app = Flask(__name__)
print('#'*30 + 'loading credit model' + '#' * 30)
load_credit_model()
print('#'*30+ 'credit model loaded' + '#' *32)


data_file_path = os.path.join(os.path.dirname(__file__), 'data.csv')
df = pd.read_csv(data_file_path)
#df = pd.read_csv('honest-tom/data.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_')

string_columns = list(df.dtypes[df.dtypes == 'object'].index)

for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')

log_price = np.log1p(df.msrp)

# Validation Framework
np.random.seed(2)

n = len(df)
load_inventory(df)
n_val = int(0.2 * n)
n_test = int(0.2 * n)
n_train = n - (n_val + n_test)

idx = np.arange(n)
np.random.shuffle(idx)

df_shuffled = df.iloc[idx]

df_train = df_shuffled.iloc[:n_train].copy()
df_val = df_shuffled.iloc[n_train:n_train + n_val].copy()
df_test = df_shuffled.iloc[n_train + n_val:].copy()

y_train_orig = df_train.msrp.values
y_val_orig = df_val.msrp.values
y_test_orig = df_test.msrp.values

y_train = np.log1p(df_train.msrp.values)
y_val = np.log1p(df_val.msrp.values)
y_test = np.log1p(df_test.msrp.values)

del df_train['msrp']
del df_val['msrp']
del df_test['msrp']


# Linear Regression
def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)

    return w[0], w[1:]


base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']

df['engine_fuel_type'].value_counts()


# %%
def prepare_X(df):
    df = df.copy()
    features = base.copy()

    df['age'] = 2017 - df.year
    features.append('age')

    for v in [2, 3, 4]:
        feature = 'num_doors_%s' % v
        df[feature] = (df['number_of_doors'] == v).astype(int)
        features.append(feature)

    for v in ['chevrolet', 'ford', 'volkswagen', 'toyota', 'dodge']:
        feature = 'is_make_%s' % v
        df[feature] = (df['make'] == v).astype(int)
        features.append(feature)

    for v in ['regular_unleaded', 'premium_unleaded_(required)',
              'premium_unleaded_(recommended)', 'flex-fuel_(unleaded/e85)']:
        feature = 'is_type_%s' % v
        df[feature] = (df['engine_fuel_type'] == v).astype(int)
        features.append(feature)

    for v in ['automatic', 'manual', 'automated_manual']:
        feature = 'is_transmission_%s' % v
        df[feature] = (df['transmission_type'] == v).astype(int)
        features.append(feature)

    for v in ['front_wheel_drive', 'rear_wheel_drive', 'all_wheel_drive', 'four_wheel_drive']:
        feature = 'is_driven_wheens_%s' % v
        df[feature] = (df['driven_wheels'] == v).astype(int)
        features.append(feature)

    for v in ['crossover', 'flex_fuel', 'luxury', 'luxury,performance', 'hatchback']:
        feature = 'is_mc_%s' % v
        df[feature] = (df['market_category'] == v).astype(int)
        features.append(feature)

    for v in ['compact', 'midsize', 'large']:
        feature = 'is_size_%s' % v
        df[feature] = (df['vehicle_size'] == v).astype(int)
        features.append(feature)

    for v in ['sedan', '4dr_suv', 'coupe', 'convertible', '4dr_hatchback']:
        feature = 'is_style_%s' % v
        df[feature] = (df['vehicle_style'] == v).astype(int)
        features.append(feature)

    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X

def rmse(y, y_pred):
    error = y_pred - y
    mse = (error ** 2).mean()
    return np.sqrt(mse)


X_train = prepare_X(df_train)
w_0, w = train_linear_regression(X_train, y_train)

y_pred = w_0 + X_train.dot(w)

X_val = prepare_X(df_val)
y_pred = w_0 + X_val.dot(w)


def get_car()->dict:
    return {'make': 'volvo', 'model': '940', 'year': 1994, 'engine_fuel_type': 'regular_unleaded',
            'engine_hp': 114.0, 'engine_cylinders': 4.0, 'transmission_type': 'automatic',
            # 'driven_wheels': None, 'number_of_doors': None,
            # 'market_category': 'luxury', 'vehicle_size': 'midsize', 'vehicle_style': 'sedan',
            # 'highway_mpg': 24, 'city_mpg': 17, 'popularity': 870, 'engine fuel_type': None}
            'driven_wheels': 'rear_wheel_drive', 'number_of_doors': 4.0,
            'market_category': 'luxury', 'vehicle_size': 'midsize', 'vehicle_style': 'sedan',
            'highway_mpg': 24, 'city_mpg': 17, 'popularity': 870}

def train_linear_regression_reg(X, y, r=0.0):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    reg = r * np.eye(XTX.shape[0])
    XTX = XTX + reg

    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)

    return w[0], w[1:]

X_train = prepare_X(df_train)
w_0, w = train_linear_regression_reg(X_train, y_train, r=0.01)

X_val = prepare_X(df_val)
y_pred = w_0 + X_val.dot(w)


X_test = prepare_X(df_test)
y_pred = w_0 + X_test.dot(w)


@app.route('/price', methods=['POST'])
def price_a_car():
    ad = get_car()
    X_test = prepare_X(pd.DataFrame([ad]))[0]
    y_pred = w_0 + X_test.dot(w)
    suggestion = np.expm1(y_pred)
    return str(round(suggestion,2)),200

@app.route('/inventory/<make>')
def get_inventory_by_make(make):
    the_make = inventory.get(make)
    if the_make is None:
        return f'no such make: {make}', 404
    return the_make, 200

@app.route('/inventory/<make>/<model>')
def get_inventory_by_make_and_model(make, model):
    the_make = inventory.get(make)
    if the_make is None:
        return f'no such make: {make}', 404
    the_model = the_make.get(model)
    if the_model is None:
        return f'no such model: {model}', 404
    return the_model, 200

@app.route('/inventory/<make>/<model>/<year>')
def get_inventory_by_make_model_and_year(make, model, year):
    the_make = inventory.get(make)
    if the_make is None:
        return f'no such make: {make}', 404
    the_model = the_make.get(model)
    if the_model is None:
        return f'no such model: {model}', 404
    try:
        year_num = int(year)
        the_year = the_model.get(year_num)
        if the_year is None:
            return f'no such year:  {year}', 404
    except TypeError:
        return f'invalid year: {year}', 404
    return the_year

@app.route('/price/<make>/<model>/<year>', methods=['GET'])
def price_for_car_in_inventory(make, model, year):
    the_make = inventory.get(make)
    if the_make is None:
        return f'no such make: {make}', 404
    the_model = the_make.get(model)
    if the_model is None:
        return f'no such model: {model}', 404
    try:
        year_num = int(year)
        the_year = the_model.get(year_num)
        if the_year is None:
            return f'no such year:  {year}', 404
    except TypeError:
        return f'invalid year: {year}', 404

    characteristics = the_year.copy()
    characteristics.pop('msrp')
    X_test = prepare_X(pd.DataFrame([characteristics]))[0]
    y_pred = w_0 + X_test.dot(w)
    suggestion = np.expm1(y_pred)
    return str(round(suggestion,2)),200

@app.route('/buy/<make>/<model>/<year>', methods=['POST'])
def apply_for_credit(make, model, year):
    car_price = price_for_car_in_inventory(make, model, year)
    credit_info = request.get_json()
    # credit_info['price'] = float(car_price[0])/10
    # credit_info['amount'] = float(car_price[0])*.075
    print(f'car price/10: {credit_info["price"]}, loan amount/10: {credit_info["amount"]}')
    one_record = [credit_info]
    if credit_info is None:
        return f'you must supply your credit application as json',400
    print(type(credit_info))
    default_likelihood = evaluate_creditworthiness(credit_model, one_record)
    print(default_likelihood)
    if default_likelihood < .10:
        return f'you are qualified, congratulations!', 200
    else:
        return f'you are not qualified, sorry', 403

if __name__ == '__main__':
    app.run(debug=True)


# for idx in range(1,2):
#     i = randint(0,len(df_test))
#     volvo_940 = df_test[(df['make'] == 'volvo') & (df['model'] == '940')]
#     # ad = df_test.iloc[i].to_dict()
#     ad = get_car()
#     print(ad)
#     car_desc = f'{ad["year"]} {ad["make"]} {ad["model"]}'
#
#     X_test = prepare_X(pd.DataFrame([ad]))[0]
#     y_pred = w_0 + X_test.dot(w)
#     suggestion = np.expm1(y_pred)
#     print(f'Estimated price for a {car_desc} is ${round(suggestion,2)}')