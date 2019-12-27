import numpy as np
import xlrd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold

'''Introducing the data & basic set up'''
data01 = xlrd.open_workbook(r"E:\PHBS\yr2-module2\Big Data\2019.M2.BigData\climate_change_1.xlsx")
data02 = xlrd.open_workbook(r"E:\PHBS\yr2-module2\Big Data\2019.M2.BigData\climate_change_2.xlsx")
table01 = data01.sheet_by_index(0)
table02 = data02.sheet_by_index(0)


def introducing_data(table, beg_row, end_row, beg_col, end_col=None, with_constant=True):
    sample_list = []
    for i in range(beg_row, end_row):
        sample_i = table.row_values(i, beg_col, end_col)
        if with_constant:
            sample_i.append(1)
        sample_list.append(sample_i)
    return sample_list


def getting_R_square(x, y, theta):
    dif_of_y_est = x.dot(theta) - y
    rss = np.dot(dif_of_y_est.T, dif_of_y_est)
    dif_of_y = y - np.mean(y)
    tss = np.dot(dif_of_y.T, dif_of_y)
    return 1 - rss / tss


def getting_t_statistic(x, y, theta):
    eps = (y - x.dot(theta))
    sigma_square = np.dot(eps.T, eps) / (x.shape[0] - x.shape[1])
    xx = np.dot(x.T, x)
    xxi_di = xx.I.diagonal()
    t_statistic = []
    se = np.sqrt(sigma_square[0, 0] * xxi_di)
    for i in range(9):
        t_sta_i = theta[i, 0] / se[0, i]
        t_statistic.append(t_sta_i)
    return t_statistic


def print_if_significant(factors, t_stat, num):
    for i in range(num):
        if abs(t_stat[i]) > 1.969:
            print(f"The theta of {factors[i]} is significant in the 95% confident level.")
        else:
            print(f"The theta of {factors[i]} is insignificant in the 95% confident level.")


# getting data from climate1
x01_training = np.mat(introducing_data(table01, 1, 285, 2, 10))
x01_testing = np.mat(introducing_data(table01, 285, 309, 2, 10))
y01_training = np.mat(introducing_data(table01, 1, 285, -1, with_constant=False))
y01_testing = np.mat(introducing_data(table01, 285, 309, -1, with_constant=False))
# getting data from climate2
x02_training = np.mat(introducing_data(table02, 1, 285, 2, 11))
x02_testing = np.mat(introducing_data(table02, 285, 309, 2, 11))
y02_training = np.mat(introducing_data(table02, 1, 285, -1, with_constant=False))
y02_testing = np.mat(introducing_data(table02, 285, 309, -1, with_constant=False))

"""Question1"""


# (1)
def closed_form_1(x, y):
    x_t = x.T
    xx = np.dot(x_t, x)
    xxi = xx.I
    xy = np.dot(x_t, y)
    return np.dot(xxi, xy)


# (2)
# calculating theta
theta01_normal = closed_form_1(x01_training, y01_training)
print(theta01_normal)
# calculating R square
R_square_training_norm = getting_R_square(x01_training, y01_training, theta01_normal)
print(R_square_training_norm)
R_square_testing_norm = getting_R_square(x01_testing, y01_testing, theta01_normal)
print(R_square_testing_norm)

# (3)
t_sta01_norm_train = getting_t_statistic(x01_training, y01_training, theta01_normal)
print(t_sta01_norm_train)
var_labels = ["MEI", "CO2", "CH4", "N2O", "CFC-11", "CFC-12", "TSI", "Aerosols", "Constant"]
print_if_significant(var_labels, t_sta01_norm_train, 9)

# (4)
# written part is attached in the markdown file
theta02 = closed_form_1(x02_training, y02_training)
t_sta02_norm_train = getting_t_statistic(x02_training, y02_training, theta02)
var_labels02 = ["MEI", "CO2", "CH4", "N2O", "CFC-11", "CFC-12", "TSI", "Aerosols", "NO", "Constant"]
print_if_significant(var_labels02, theta02, 10)

'''Question2'''


# (1) is written in the markdown file

# (2)
def closed_form_2(x, y, lam):
    x_t = x.T
    xx = np.dot(x_t, x)
    identity = np.identity(x.shape[1])
    xy = np.dot(x_t, y)
    return np.dot((xx + lam * identity).I, xy)


# (3)
# getting the robustness of regularized model
theta01_reg = closed_form_2(x01_training, y01_training, 1)
R_square_testing_reg = getting_R_square(x01_testing, y01_testing, theta01_reg)
print(theta01_reg)

# (4)
lam_list = [10, 1, 0.1, 0.01, 0.001]
R_square_training_list = []
R_square_testing_list = []
for i in lam_list:
    theta_reg = closed_form_2(x01_training, y01_training, i)
    R_square_training_list.append(getting_R_square(x01_training, y01_training, theta_reg))
    R_square_testing_list.append(getting_R_square(x01_testing, y01_testing, theta_reg))
print(R_square_training_list, R_square_testing_list)

KF_model = KFold(n_splits=10, shuffle=True)
R_square_list = {}
for i in lam_list:
    R_square_sum = 0
    for train_index, valid_index in KF_model.split(x01_training):
        x_train_set, x_valid_set = x01_training[train_index], x01_training[valid_index]
        y_train_set, y_valid_set = y01_training[train_index], y01_training[valid_index]
        theta = closed_form_2(x_train_set, y_train_set, i)
        R_square_sum = R_square_sum + getting_R_square(x_valid_set, y_valid_set, theta)
    R_square = R_square_sum / 10
    R_square_list[i]=R_square
print(R_square_list)

'''Question3'''
# (1)
# dropping insignificant features
significant_x01_training = np.column_stack((x01_training[:, 0:2], x01_training[:, 4:8]))


def VIF(x, features):
    vif_dict = {}
    for i in range(len(features)):
        x_test = x[:, i]
        x_rest = np.column_stack((x[:, 0:i], x[:, i + 1:]))
        theta = closed_form_1(x_rest, x_test)
        vif = 1 / (1 - getting_R_square(x_rest, x_test, theta))
        vif_dict[features[i]] = vif
    print(vif_dict)


# first time of calculating VIF
sig_var_labels = ["MEI", "CO2", "CFC-11", "CFC-12", "TSI", "Aerosols"]
VIF(significant_x01_training, sig_var_labels)
# second time of calculating VIF
sig_var_labels02 = ["MEI", "CO2", "CFC-11", "TSI", "Aerosols"]
efficient_x01_training = np.column_stack((significant_x01_training[:, 0:3], significant_x01_training[:, 4:]))
VIF(efficient_x01_training, sig_var_labels02)

# (2)
# first dropping related features
x01_without_cons = x01_training[:, :8]
VIF(x01_without_cons, var_labels[:8])
# after dropping "CFC-12"
x01_training01 = np.column_stack((x01_without_cons[:, :5], x01_without_cons[:, 6:]))
feature_list01 = ["MEI", "CO2", "CH4", "N2O", "CFC-11", "TSI", "Aerosols"]
VIF(x01_training01, feature_list01)
# after dropping "N2O"
x01_training02 = np.column_stack((x01_training01[:, :3], x01_training01[:, 4:]))
feature_list02 = ["MEI", "CO2", "CH4", "CFC-11", "TSI", "Aerosols"]
VIF(x01_training02, feature_list02)
# after dropping "CH4"
x01_training03 = np.column_stack((x01_training02[:, :2], x01_training02[:, 3:]))
feature_list03 = ["MEI", "CO2", "CFC-11", "TSI", "Aerosols"]
VIF(x01_training03, feature_list03)

# then check the validity of the rest features
efficient_x01_reg_train = np.column_stack((x01_training[:, :2], x01_training[:, 4], x01_training[:, 6:]))
efficient_x01_reg_test = np.column_stack((x01_testing[:, :2], x01_testing[:, 4], x01_testing[:, 6:]))

model_lasso = LassoCV()
model_lasso.fit(efficient_x01_reg_train, y01_training)
print(model_lasso.coef_)

'''Question4'''
# transform data x
x01_testing_array = np.array(introducing_data(table01, 1, 285, 2, 10))


# function of gradient descent for ridge regression


def GradientDescent(x_array, y, alpha, lam, tol):
    normalized_x = np.ones(x_array.shape[0]).reshape((x_array.shape[0], 1))
    for i in range(x_array.shape[1] - 1):
        xi = x_array[:, i]
        xi_mat = xi.reshape(xi.shape[0], 1)
        normalized_xi = (xi_mat - np.mean(xi_mat)) / np.std(xi_mat)
        normalized_xi = np.mat(normalized_xi)
        normalized_x = np.column_stack((normalized_x, normalized_xi))
    k = normalized_x.shape[1]
    n = normalized_x.shape[0]
    theta = np.mat(np.zeros((k, 1)))
    discount = 1 - alpha * lam / n
    count = 0
    while True:
        count = count + 1
        diff = normalized_x.dot(theta) - y
        multiply = np.dot(normalized_x.T, diff)
        subtract_part = alpha / n * multiply
        next_theta = theta.dot(discount) - subtract_part
        if count > 5000 or np.dot(diff.T, diff) / (2 * n) <= tol:
            break
        theta = next_theta
    return theta


theta_gs = GradientDescent(x01_testing_array, y01_training, 0.1, 0.01, 0.001)
print(theta_gs)
print(theta01_reg)
