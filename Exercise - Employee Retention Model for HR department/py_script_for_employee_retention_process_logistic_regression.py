# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# <h2><b>Employee Retention prediction model | Logistic regression | Binary classification</b></h2>

# %%
# Modules to be imported
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# %% [markdown]
# <h2>Load the HR data set and view first 5 tuples using 'head()' method</h2>

# %%
df = pd.read_csv('HR_comma_sep.csv')
df.head()

# %% [markdown]
# <h2>Data Processing and Visualization</h2>
# <p>Inorder to build a model for this problem we need to find out the
# 'Independent variable ' and 'Dependent variable'. From the problem statement It is clear that the attribute(feature) 'left' is the Dependent variable. So we need to find the Independent variable which affects the employee retention process</p>
# %% [markdown]
# <h2>1. Find out Number of employees get 'retain' and Number of employees get 'fired'</h2>

# %%
# left (or) fired
left = df[df.left == 1]
left.shape


# %%
# Retained
retain = df[df.left == 0]
retain.shape

# %% [markdown]
# <h2>2. Find out what are all the features plays major role in Employee retention process</h2>
# <p>This is done by using 'groupby()' and 'mean()' method.<b> This is only
# applicable for numeric datas not for categorical data. so we need take
# a special care about them</b></p>

# %%
overall_view = df.groupby(df.left).mean()
overall_view

# %% [markdown]
# <h3>From the above results..</h3>
# <ol>
#     <li><b>Satisfication_level:</b> Employees having more satisfaction level were retained</li>
#     <li><b>time_spend_company:</b> Employees those had spent more times in company
#     gets fired</li>
#     <li><b>promotion_last_5years:</b> Employees those got promotion in the last 5 years were retained</li>
# </ol><br><br><br>
# <h1><b>So It is clear that the features such as 'satisfaction_level', 'time_spend_company', 'promotion_last_5years' are going to taken as 'Independent variables' for this problem statment.</b></h1>
# %% [markdown]
# <h2>3. Now find out the impact of categorical datas such as 'salary' and 'department' in employee retention process.</h2>
# <p>Inorder to do that we need to cross tabulate the 'salary' and 'department
# ' features against 'left' feature.</p>

# %%
# salary
pd.crosstab(df.salary, df.left).plot(kind="bar")

# %% [markdown]
# <h3><b>From the above results, Employees with high salary rate were less fired.</b></h3>

# %%
# Department
pd.crosstab(df.Department, df.left).plot(kind="bar")

# %% [markdown]
# <h3><b>From the above result we can conclude that Department of the employee doesn't plays a major role in employee retention process. so Assume 'salary' as independent variable.</b></h3>
# %% [markdown]
# <h2>4. Build the final data frame</h2>

# %%
sub_df = df[['satisfaction_level', 'time_spend_company',
             'promotion_last_5years', 'salary']]
sub_df.head()

# %% [markdown]
# <h3><b>convert the categorical ordinal data into numerical by 'one hot encoding' method.</b></h3>

# %%
# convert the dummy datas into numerical by 'one hot encoding ' method.
sub_df_dummies = pd.get_dummies(sub_df.salary, prefix="salary")
sub_df_dummies.head()


# %%
# concatenate the 'sub_df_dummies' with 'sub_df' and drop the 'salary' column
final_df = pd.concat([sub_df, sub_df_dummies], axis=1)
final_df.drop(labels=['salary'], axis=1, inplace=True)
final_df.head()

# %% [markdown]
# <h2>5. Split the data set into training and testing data set</h2>

# %%
x = final_df
x.head()


# %%
y = df.left
y.head()


# %%
(x_train, x_test, y_train, y_test) = train_test_split(x, y, train_size=0.7)

# %% [markdown]
# <h2>6. Create the logistic regression model and train it.</h2>

# %%
log_reg_model = LogisticRegression()
log_reg_model.fit(x_train, y_train)


# %%
log_reg_model.score(x_test, y_test)


# %%
log_reg_model.predict(x_test)
