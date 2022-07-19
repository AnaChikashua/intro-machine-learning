from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

boston_data = load_boston()

X_train, X_test, y_train, y_test = train_test_split(boston_data.data, boston_data.target, random_state=0)

lr = LinearRegression()
lr.fit(X_train, y_train)
print("Linear Regression train evaluation: ", lr.score(X_train, y_train))
print("Linear Regression test evaluation: ", lr.score(X_test, y_test))
ridge = Ridge()
ridge.fit(X_train, y_train)
print("Ridge Regression train evaluation: ", ridge.score(X_train, y_train))
print("Ridge Regression test evaluation: ", ridge.score(X_test, y_test))

ridge = Ridge(alpha=10)
ridge.fit(X_train, y_train)
print("Ridge Regression(alpha=10) train evaluation: ", ridge.score(X_train, y_train))
print("Ridge Regression(alpha=10) test evaluation: ", ridge.score(X_test, y_test))

ridge = Ridge(alpha=0.1)
ridge.fit(X_train, y_train)
print("Ridge Regression(alpha=0.1) train evaluation: ", ridge.score(X_train, y_train))
print("Ridge Regression(alpha=0.1) test evaluation: ", ridge.score(X_test, y_test))

lasso = Lasso()
lasso.fit(X_train, y_train)
print("Lasso Regression train evaluation: ", lasso.score(X_train, y_train))
print("Lasso Regression test evaluation: ", lasso.score(X_test, y_test))

lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)
print("Lasso Regression(alpha=0.01) train evaluation: ", lasso.score(X_train, y_train))
print("Lasso Regression(alpha=0.01) test evaluation: ", lasso.score(X_test, y_test))

print("number of nonzero coef: ", sum(lasso.coef_ != 0))
print("number of zero coef: ", sum(lasso.coef_ == 0))
