#Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


# Load Data Set

train = pd.read_csv("train_HFxi8kT/train.csv")
campaign_data = pd.read_csv("train_HFxi8kT/campaign_data.csv")
test = pd.read_csv("test_BDIfz5B.csv/test_BDIfz5B.csv")

# Merge Data sets for feature engineering

df_train = pd.merge(train, campaign_data, how='left', on='campaign_id', left_on=None, right_on=None,
                          left_index=False, right_index=False, sort=True,
                          suffixes=('_x', '_y'), copy=True, indicator=False)


df_test = pd.merge(test, campaign_data, how='left', on='campaign_id', left_on=None, right_on=None,
                          left_index=False, right_index=False, sort=True,
                          suffixes=('_x', '_y'), copy=True, indicator=False)

# convert catogorical variable

le = LabelEncoder()

le.fit(np.hstack([df_train.communication_type, df_test.communication_type]))
df_train.communication_type = le.transform(df_train.communication_type)
df_test.communication_type = le.transform(df_test.communication_type)
del le

# Select Predictors and prdiction columns

predictors = df_train[["user_id", "campaign_id", "communication_type", "total_links","no_of_internal_links", "no_of_images",
                   "no_of_sections"]]

y = df_train["is_click"]


# Creat Model

model = RandomForestClassifier()

# Cross Validation

scores = cross_val_score(model, predictors, y, scoring='roc_auc', cv=5)
print(scores)

# Select predictors from test columns

test_predictors = df_test[["user_id", "campaign_id", "communication_type", "total_links","no_of_internal_links", "no_of_images",
                   "no_of_sections"]]

# Fit Model

model.fit(predictors, y)

# prediction

prediction = model.predict(test_predictors)

# Creat submission file

my_submission = pd.DataFrame({'id': test.id, 'is_click': prediction})
my_submission.to_csv('RF_FE_submission.csv', index=False)
