#### Question1
###### (1)
The code part of function is as below:
```
def closed_form_1(x, y):
    x_t = x.T
    xx = x_t * x
    xxi = xx.I
    xy = x_t * y
    return xxi * xy
```
###### (2)
According to the theta calculated in the python file, the formula can be written as:
```math
Temp_i=-124.5943+0.0641MEI_i+0.0065CO2_i+0.0001CH4_i-0.0165N2O_i-0.0066CFC11_i+0.0038CFC12_i+0.0931TSI_i-1.5376Aerosols_i
```
 R square of training data is 0.7509, of testing data is 0.2251.
###### (3)
 According to the t-statistic calculated in the python file, "MEI", "CO2", "CFC-11", "CFC-12", "TSI", "Aerosols", "Constant" is significant in the model.
###### (4)
According to the result calculated in python file, only theta of "NO" and "Constant" are significant, which means the application of model to climate_change_2 file is not robust.

This is because the second file has one more dimension, which will cause the data become more sparese and harder to fit in the model.

#### Question2
###### (1)
loss function with L1:
```math
J(θ)=\frac{1}{2N}[∑_{i=1}^N(h(x_i )-y_i )^2 +λ∑_{j=1}^k\left|θ_j\right| ] \quad where\quad h(x)=x*θ
```
with L2:
```math
J(θ)=\frac{1}{2N}[∑_{i=1}^N(h(x_i )-y_i )^2 +λ∑_{j=1}^kθ_j^2 ] \quad where\quad h(x)=x*θ
```
###### (2)
The code part of function is as below:
```
def closed_form_2(x, y, lam):
    x_t = x.T
    xx = x_t * x
    identity = np.identity(9)
    xy = x_t * y
    return (xx + lam * identity).I * xy
```
###### (3)
According to the calculation in python file(setting the λ=1), the R square of testing set in the model with L2 regularization is 0.8468, which is much higher than the R square of testing set in the normal model (recall that previous R square = 0.2251)

Linear model with L2 regularization is more robust because it avoid the overfitting by reducting the magnitude of each parameters θ.
###### (4)

λ | training set| testing set
---|---|---
10    | 0.6746|0.0969
1     | 0.6795|0.1085
0.1   | 0.6945|0.1266
0.01  | 0.7117|0.1384
0.001 | 0.7148|0.1410

From the table above we can say that λ=0.001 is the best parameter since it reduce the problem of overfitting the most.

#### Question3
###### (1)
###### Step1:
According to question 1, "CH4" and "N2O" are not significantly related to temperature, so we can drop this two features.
###### Step2：
Calculating the VIF of each feature and drop the highly correlated features. First calculation of VIF tells that the features 'CO2', 'CFC-11' and 'CFC-12' has a large VIF, which means that they are highly correlated with other features. We dropped the one with the largest VIF (i.e.'CFC-12') and calculated the VIFs again. We can see from the second result that features are not closely correlated anymore.

The VIF of each significant feature is as follow:

features | VIFs-1 | VIFs-2
---|---|---
'MEI'     | 1.1489  |1.1435
'CO2'     | 16.2373 |1.6058
'CFC-11'  | 18.0205 |1.4011
'CFC-12'  | 48.1977 |dropped
'TSI'     | -0.0012 |-0.0001
'Aerosols'| 1.3342  |1.3342

P.S.The function of calculating VIF is:

```
def VIF(x, features):
    vif_dict = {}
    for i in range(len(features)):
        x_test = x[:, i]
        x_rest = np.column_stack((x[:, 0:i], x[:, i + 1:]))
        theta = closed_form_1(x_rest, x_test)
        vif = 1 / (1 - getting_R_square(x_rest, x_test, theta))
        vif_dict[features[i]] = vif
    print(vif_dict)
```
###### Step3:
Train the model again using the rest features:"MEI", "CO2", "CFC-11", "TSI", "Aerosols".

###### (2)
To train the second model, first we want to drop out mutually related features, after calculating the VIFs as above, we dropped "CFC-12", "N2O", "CH4".

After that, we check the validity of the rest features by checking the result of testing set after dropping each feature.
The R square of testing set without dropping any features is: 0.4683.
The R squre of dropping each feature is as following:

features | R square after dropping
---|---
'MEI' | 0.0489
'CO2' | 0.1364
'CFC-11'|0.1746
'TSI'|0.1735
'Aerosols'|0.1364

From the data above we can know that none of them can be deleted.

#### Question4
The code part of function is as below:

```
def GradientDescent(x, y, alpha, lam, tol):
    k = x.shape[1]
    n = x.shape[0]
    theta = np.mat(np.zeros((k, 1)))
    discount = 1 - alpha * lam / n
    count = 0
    while True:
        count = count + 1
        diff = x * theta - y
        multiply = x.T * diff
        subtract_part = alpha / n * multiply
        next_theta = theta * discount - subtract_part
        if count > 1000 or diff.T * diff / (2 * n) <= tol:
            break
        theta = next_theta
    return theta
```
