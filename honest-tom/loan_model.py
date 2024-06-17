import numpy as np
import pandas as pd
import xgboost as xgb
import sklearn


df = None
df_train_full = None
df_test = None
df_train = None
df_val = None
actual_results_train = None
actual_results_val = None
dict_train = None
dict_val = None
dv = None

def load_data():
    global df, df_train_full, df_test,df_train, df_val, actual_results_train, actual_results_val
    global dict_train, dict_val, dv
    # open the input file of historical data
    df = pd.read_csv('loan_outcomes_historical_v2.csv')

    ### Rename Column Names

    # convert the column names to lower case and print again

    df.columns = df.columns.str.lower()

    ### Perform Mapping for All Enumerated Columns"""

    # perform mapping for colum and print again.  column:  status
    status_values = {
        1: 'ok',
        2: 'default',
        0: 'unk'
    }
    df.status = df.status.map(status_values)

    # perform mapping for colum and print again.  column:  home
    home_values = {
        1: 'rent',
        2: 'owner',
        3: 'private',
        4: 'ignore',
        5: 'parents',
        6: 'other',
        0: 'unk'
    }
    df.home = df.home.map(home_values)

    # perform mapping for colum and print again.  column:  marital
    marital_values = {
        1: 'single',
        2: 'married',
        3: 'widow',
        4: 'separated',
        5: 'divorced',
        0: 'unk'
    }
    df.marital = df.marital.map(marital_values)

    # perform mapping for colum and print again.  column:  records
    records_values = {
        1: 'no',
        2: 'yes',
        0: 'unk'
    }
    df.records = df.records.map(records_values)

    # perform mapping for colum and print again.  column:  job
    job_values = {
        1: 'fixed',
        2: 'partime',
        3: 'freelance',
        4: 'others',
        0: 'unk'
    }
    df.job = df.job.map(job_values)

    """### Data Cleansing"""


    # this dataset had some values as 999s, which meant values were missing
    # clean up by changing 999s values to NaN ("not a number", null in a sense)
    for c in ['income', 'assets', 'debt']:
        df[c] = df[c].replace(to_replace=99999999, value=np.nan)
    df.describe().round()

    # 'let's look at our target column, which is 'status'
    # 'status' has 3 possible values:  ok, default, and unk
    # 'ok' = customer paid back the loan
    # 'default' = customer defaulted on the loan
    # 'unk' = unknown
    # print the counts for each value
    df.status.value_counts()

    # note from the output above, there is one that is 'unk'
    # let's delete it because we don't know the outcome for this loan
    df = df[df.status != 'unk']

    """## Split the Historical Dataset into 3 smaller sets:  Train, Validation, Test"""

    # now we are ready to split the historical data into 3 datasets

    # import train_test_split from sklearn package
    from sklearn.model_selection import train_test_split

    # split the historical dataset into subsets

    # 1 - df_test.  20% of original historical data.
    # set aside these rows for much later, to have some testing data for later
    # it is choosing rows randomly
    df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=11)

    # 2 - df_val.   25% of 80% of historical (20%).
    # will be used to validate our model
    # it is choosing rows randomly
    df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=11)

    # 3 - df_train.   60% of historical data
    # the bulk of our data, used to train our model

    # establish our 'target' for the ML model
    # our 'target' is what our ML program is trying to predict
    # we are trying to predict which customers are going to default on the loan
    # we shall make a note of all historical data loans that defaulted
    # in ML, the 'y' is the target that are trying to predict

    # training data
    # actual results from the training dataset
    actual_results_train = (df_train.status == 'default').values

    # validation data
    # actual results from the validation dataset
    actual_results_val = (df_val.status == 'default').values

    # now that we have set the 'y', we can delete the status column from
    # our training and validation datasets
    # we won't use status to help us predict the status!
    del df_train['status']
    del df_val['status']

    # recall that we have 3 datasets we are working with:
    # df_train - used to train the model (60% of the historical data)
    # df_val - used to validate the model (20% of the historical data)
    # df_test - for testing much later once we have established our model (20%)

    """## Prepare our Training and Validation Datasets"""

    """### More Data Cleansing"""

    # for any cells that were null, simply fill them with a 0
    # this will allow our model to train more accurately
    dict_train = df_train.fillna(0).to_dict(orient='records')
    dict_val = df_val.fillna(0).to_dict(orient='records')

    """### Prepare Structured Data for Later Viewing"""

    # notice we have now created dict_train and dict_val variables
    # these are lists of dictionaries
    # recall that a dictionary is a mapping of key / value pairs


    """## Train our Machine Learning Model"""

    # import various python modules that are needed for machine learning

    from sklearn.metrics import roc_auc_score
    from sklearn.feature_extraction import DictVectorizer

    # prepare with our training dataset (60%)
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dict_train)
    dtrain = xgb.DMatrix(X_train, label=actual_results_train, feature_names=dv.feature_names_)

    # prepare with our validation dataset (20%)
    X_val = dv.transform(dict_val)
    dval = xgb.DMatrix(X_val, label=actual_results_val, feature_names=dv.feature_names_)

    # prior to this, we did much 'tuning' to figure out the exact parameters
    # that would yield the best results
    best_parameters = {
        'eta': 0.3,
        'max_depth': 6,
        'min_child_weight': 1,
        'objective': 'binary:logistic',
        'nthread': 8,
        'seed': 1
    }

    # now train the model using our training dataset (60%)
    model = xgb.train(best_parameters, dtrain, num_boost_round=10)

    # now create a list of predictions for the validation dataset
    y_predictions = model.predict(dval)


    # just for fun, print the accuracy of our predictive model.
    # compare the model's predictions (y_predictions)
    # this particular evaluation is based on two terms:
    # ROC = 'receiver operating characteristic'
    # ROC interprets how we have more True Positives than False Positives
    # AUC = 'area under the ROC curve', we want as close to 1.0 as possible

    """## Final Training of the Model with 80% of Dataset"""

    # now that we have validated our model
    # start all over again, do the same steps again,
    # this time with a LARGER training data set (80% of historical data)

    # and we don't need the 20% validation dataset anymore,
    # because we already validated the model and are satisfied
    # now we can combine the 60% (train) and 20% (validation) together to get 80%
    actual_results_train_full = (df_train_full.status == 'default').values
    actual_results_test = (df_test.status == 'default').values

    # same as before, delete the status from the dataset of input data
    # we don't want an incorrect circular dependency!
    del df_train_full['status']
    del df_test['status']

    # same as before, prepare the data
    dict_train_full = df_train_full.fillna(0).to_dict(orient='records')
    dict_test = df_test.fillna(0).to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train_full = dv.fit_transform(dict_train_full)
    X_test = dv.transform(dict_test)
    dtrain_full = xgb.DMatrix(X_train_full, label=actual_results_train_full, feature_names=dv.feature_names_)
    dtest = xgb.DMatrix(X_test, label=actual_results_test, feature_names=dv.feature_names_)

    # same as before, use the same parameters that were tuned
    best_parameters = {
        'eta': 0.1,
        'max_depth': 3,
        'min_child_weight': 1,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'nthread': 8,
        'seed': 1,
    }
    num_trees = 160

    # train the model
    model = xgb.train(best_parameters, dtrain_full, num_boost_round=num_trees)

    return model


def evaluate_creditworthiness(model, one_record:[dict]):
    """## Now run a few records against the model"""

    # now let's run a few records into our model

    # first try opening up a new file for new july loan requests
    df_few_records = pd.read_csv('loan_requests_july.csv')

    # prepare our dataset
    actual_results_few_records = (df_few_records.status == 'default').values
    del df_few_records['status']


    # prepare the dataset just like we prepared our other datasets
    #one_record = df_few_records.fillna(0).to_dict(orient='records')
    X_few_records = dv.transform(one_record)
    few_records_dmatrix = xgb.DMatrix(X_few_records, label=actual_results_few_records, feature_names=dv.feature_names_)

    # now predict using the model and get back some predictions
    predictions = model.predict(few_records_dmatrix)
    return predictions[0]
    # TODO - copy over your traditional algorithm function here

    # loop through each loan request
    # count = 0
    # keys = ['seniority', 'home', 'age', 'marital', 'job', 'assets', 'debt']
    # for loan_request in one_record:
    #   subset = dict()
    #   for k in keys:
    #     subset[k] = loan_request[k]
    #
    #   # print the loan request
    #   print('loan request is:', subset)
    #
    #   # first, print the ML model determination
    #   print('Our ML model predicts the chance that this customer will default:', round(predictions[count], 3))
    #   print('Does our ML model grant the loan?', predictions[count] < .1)
    #
    #
    #   count += 1
    #   print()

    """## Just for fun, attempt to understand the inner-workings of the model"""

    # just for fun, print the inner-workings of the model
    # it creates a table based on what it learns
    # import sys
    # model.dump_model(sys.stdout)

if __name__ == '__main__':
    credit_model = load_data()
    one_record = [{'age': 33, 'amount': 1200, 'assets': 0, 'debt': 0, 'expenses': 35, 'home': 'parents', 'income': 200, 'job': 'fixed', 'marital': 'married', 'price': 1580, 'records': 'no', 'seniority': 1, 'time': 36}]

    decision = evaluate_creditworthiness(credit_model, one_record)
    print(decision)