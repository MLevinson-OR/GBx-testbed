import numpy as np

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import GaussianNB


class SGD(object):
	def __init__(self):
		self.sgd = SGDClassifier(loss='modified_huber', alpha = .00001, penalty='elasticnet',shuffle=True, n_jobs=-1,random_state = 2014)
	def predict(self, X):
		return self.sgd.predict_proba(X)[:,1][:,np.newaxis]
	def fit(self, X, y):
		self.sgd.fit(X,y)
		
class GNB(object):
	def __init__(self):
		self.gnb = GaussianNB()
	def predict(self, X):
		return self.gnb.predict_proba(X)[:,1][:,np.newaxis]
	def fit(self, X, y):
		self.gnb.fit(X,y)
		
class RF(object):
	def __init__(self):
		self.rf = RandomForestClassifier(n_estimators=100,n_jobs=-1)
	def predict(self,X):
		return self.rf.predict_proba(X)[:,1][:,np.newaxis]
	def fit(self, X, y):
		self.rf.fit(X,y)
		
class Logistic(object):
	def __init__(self):
		self.logistic = LogisticRegression()
	def predict(self,X):
		return self.logistic.predict_proba(X)[:,1][:,np.newaxis]
	def fit(self, X, y):
		self.logistic.fit(X,y)
		
class GBC(object):
	def __init__(self):
		self.gbc = GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,subsample=0.5,max_depth=6,verbose=1)
	def predict(self,X):
		return self.gbc.predict_proba(X)[:,1][:,np.newaxis]
	def fit(self, X, y):
		self.gbc.fit(X,y)
		
		
		