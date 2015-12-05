from sklearn import linear_model
from sklearn.linear_model.base import LinearRegression
from scipy.constants.constants import alpha
from sklearn.linear_model.ridge import Ridge

clf = linear_model.LinearRegression()
clf.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
print clf.coef_

clf = linear_model.Ridge (alpha=.5)
clf.fit ([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
Ridge(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None, normalize=False, random_state=None, solver='auto', tol=0.001)
print clf.coef_
print clf.intercept_
