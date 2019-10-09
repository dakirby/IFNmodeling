import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Import data set and split into predictor variables (receptors, JAKs, SOCS, etc.) and response variables (STATs)
ImmGen_df = pd.read_excel('ImmGen_signaling_with_protein_response.xlsx', sheet_name='Sheet1', axis=1)
cell_type_to_index = pd.Series(ImmGen_df.index, index=ImmGen_df.Cell_type.values).to_dict()

#create df of results and features from full dataframe
response_variable_names = ['pSTAT1', 'pSTAT3']
response_variables = ImmGen_df[response_variable_names]

predictor_variable_names = [c for c in ImmGen_df.columns.values if c not in response_variable_names + ['Cell_type']]
predictor_variables = ImmGen_df[predictor_variable_names]

#center the variables
x_centered = scale(predictor_variables, with_mean='True', with_std='False')
centered_predictor_variables = pd.DataFrame(x_centered, columns=predictor_variables.columns)

# --------------------
# Quadratic regression
# --------------------
poly = PolynomialFeatures(degree=2, interaction_only=True)
poly_variables = poly.fit_transform(centered_predictor_variables)

poly_var_train, poly_var_test, poly_res_train, poly_res_test = train_test_split(poly_variables, response_variables.values,
                                                                      test_size=0.25, random_state=4)
# Create linear regression object
regr = LinearRegression()
# Train the model using the training sets
regr.fit(poly_var_train, poly_res_train)
score = regr.score(poly_var_test, poly_res_test)
# Make predictions using the testing set
STAT_pred = regr.predict(poly_var_test)
# The coefficients
#print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Quadratic regression mean squared error: %.2f"
      % mean_squared_error(poly_res_test, STAT_pred))
# Explained variance score: 1 is perfect prediction
print('Quadratic regression variance score: %.2f' % r2_score(poly_res_test, STAT_pred))

# --------------------------
# Lasso quadratic regression
# --------------------------
alpha_tests = np.logspace(-3, 2)
r2_array = []
for a in alpha_tests:
    clf = Lasso(alpha=a)
    clf.fit(poly_var_train, poly_res_train)
    STAT_pred = clf.predict(poly_var_test)
    s = r2_score(poly_res_test, STAT_pred)
    r2_array.append(s)
best_alpha = alpha_tests[np.where(r2_array==max(r2_array))]
print("Lasso quadratic regression best alpha value: {}".format(best_alpha))
print("Lasso quadratic regression best r2 score: {}".format(max(r2_array)))
plt.figure()
ax=plt.gca()
ax.set_xscale('log')
plt.plot(alpha_tests, r2_array)
plt.show()

# --------------------------
# Lasso linear regression
# --------------------------
linear_var_train, linear_var_test, linear_res_train, linear_res_test = train_test_split(centered_predictor_variables.values, response_variables.values,
                                                                      test_size=0.25, random_state=4)
alpha_tests = np.logspace(-3, 2)
r2_array = []
for a in alpha_tests:
    clf = Lasso(alpha=a)
    clf.fit(linear_var_train, linear_res_train)
    STAT_pred = clf.predict(linear_var_test)
    s = r2_score(linear_res_test, STAT_pred)
    r2_array.append(s)
best_alpha = alpha_tests[np.where(r2_array==max(r2_array))]
print("Lasso linear regression best alpha value: {}".format(best_alpha))
print("Lasso linear regression best r2 score: {}".format(max(r2_array)))
plt.figure()
ax=plt.gca()
ax.set_xscale('log')
plt.plot(alpha_tests, r2_array)
plt.show()