import numpy as np
import itertools
import random
from response import Response, Question
from scipy.optimize import minimize
from helper import attributesToFeatures, weightedSum

def costNTopK(n, k):
  if n == 10 and k == 1:
    return 1.384957*k+32.24613
  elif n == 2 and k == 1:
    return 5.333818*n
  elif n == 10 and k == 9:
    return 5.333818*n
  else:
    print 'cost ???'

class preferenceElicitation(object):
  """docstring for preferenceElicitation"""
  def __init__(self, dcsn):
    self.decision = dcsn
    self.data = [] # a list of Responses
    self.asked = [] # a list of Questions
    self.mBeta = None
  
  def clearData(self):
    del self.data[:]
    del self.asked[:]
    self.mBeta = None

  def addData(self, d): # d = Response(Alternative, (a,b), set(a,b,c))
    assert isinstance(d, Response)
    self.data.append(d)
    q = Question(d.agent, d.proposed, len(d.ranked))
    assert q not in self.asked, '%s' % (q)
    self.asked.append(q)

  def MAP(self, itr=2000):
    bnds = [(-1,1)]*self.decision.dimension
    res = minimize(self.neg_f, 
            np.zeros(self.decision.dimension), 
            method="SLSQP", 
            bounds=bnds,
            options={
              'maxiter': itr
            }
          )
    self.mBeta = tuple(res.x)
    return self.mBeta

  def f(self, beta):
    ll = 0.0
    for response in self.data:
      ll += response.getLogProbability(beta)
    return ll

  def neg_f(self, beta):
    return -self.f(beta)

  def nextQuestion(self, criterion, num, top=None):
    assert num >= 2
    if not top: top = num-1
    R = np.zeros((self.decision.dimension, self.decision.dimension), float)
    for r in self.data:
      R -= self.secondDerivativeOfPlackettLuce(r, self.mBeta) 
    if criterion == 'proposed':
      prior_info = self.minCertainty(self.mBeta, np.linalg.inv(R))
    elif criterion == 'd-opt':
      prior_info = np.linalg.det(R)
    elif criterion == 'e-opt':
      prior_info = min(np.linalg.eig(R)[0])
    elif criterion == 't-opt':
      prior_info = np.trace(R)
    else:
      prior_info = None


    if num == len(self.decision.alternatives) and (top == num-1 or top == num): # rank all
      quality = {}
      quality_over_cost = {}
      alts = tuple(self.decision.alternatives)
      hs = [Question(agt, set(alts), top) for agt in self.decision.crowds if Question(agt, set(alts), top) not in self.asked]
      if len(hs) == 0: return False, False
      if criterion == 'random':
        return random.choice(hs), np.random.uniform()

      for h in hs:
        rs = [self.decision.getResponse(agt, beta=self.mBeta) for _ in xrange(50)]
        quality[h] = 0.0
        for r in rs:
          quality[h] += self.G_star(self.mBeta, r, criterion, R, prior_info)*(1.0/50)
        quality_over_cost[h] = quality[h]/(agt.cost*costNTopK(num, top))
      nextQ = max(quality_over_cost, key=quality_over_cost.get)
      # print num, 'choose', top, '- criterion:', criterion, '; Q_Quality:', quality[nextQ], '; agt_cost:', nextQ.agent.cost, '; Q_cost:', costNTopK(num, top)
      return nextQ, quality_over_cost[nextQ]


    hs = [Question(agt, set(alts), top) for agt in self.decision.crowds for alts in list(itertools.combinations(self.decision.alternatives, num)) if Question(agt, set(alts), top) not in self.asked] # all possible questions
    if len(hs) == 0: return False, False
    if criterion == 'random':
      return random.choice(hs), np.random.uniform()
 
    quality = {}
    quality_over_cost = {}
    for h in hs:
      rs = [Response(agt, p, set(alts)) for c in list(itertools.combinations(alts, top)) for p in list(itertools.permutations(c))] # all possible answers
      quality[h] = 0.0
      for r in rs:
        quality[h] += self.G_star(self.mBeta, r, criterion, R, prior_info)*r.getProbability(self.mBeta)
      quality_over_cost[h] = quality[h]/(agt.cost*costNTopK(num, top))
    
    nextQ = max(quality_over_cost, key=quality_over_cost.get)
    # print num, 'choose', top, '- criterion:', criterion, '; Q_Quality:', quality[nextQ], '; agt_cost:', nextQ.agent.cost, '; Q_cost:', costNTopK(num, top)
    return nextQ, quality_over_cost[nextQ]

  def G_star(self, theta, response, criterion, R, prior_info):
    I_h = -(self.secondDerivativeOfPlackettLuce(response, theta))
    if criterion == 'proposed':
      return self.minCertainty(theta, np.linalg.inv(R+I_h)) - prior_info
    elif criterion == 'd-opt':
      return np.linalg.det(R+I_h) - prior_info
    elif criterion == 'e-opt':
      return min(np.linalg.eig(R+I_h)[0]) - prior_info
    elif criterion == 't-opt':
      return np.trace(R+I_h) - prior_info
    else:
      print '???'

  def secondDerivativeOfPlackettLuce(self, response, theta):
    expScores = {alt: np.exp(self.decision.getUtility(response.agent, alt, theta)) for alt in response.proposed}
    expArray = np.matrix([expScores[alt] for alt in response.proposed])
    localsum = np.sum(expArray[0])
    alts = list(response.proposed)
    features = []
    for alt in alts:
      features.append(attributesToFeatures(response.agent.attributeValues, alt.attributeValues))
    features = np.matrix(features)
    m = np.zeros((self.decision.dimension, self.decision.dimension), float)
    idx = 0
    if len(response.proposed) > len(response.ranked):
      for choice in response.ranked:
        for i in xrange(self.decision.dimension):
          p1 = np.dot(expArray, features[:,i])
          p2 = np.dot(expArray, np.square(features[:,i]))
          m[i, i] += (p1/localsum) ** 2 - p2/localsum
          for j in xrange(i+1, self.decision.dimension):
            p3 = np.dot(expArray, features[:,j])
            p4 = np.dot(expArray, np.multiply(features[:, i], features[:, j]))
            m[i,j] += p1 * p3 / (localsum ** 2) - p4/localsum
            m[j, i] = m[i, j]
        idx = alts.index(choice)
        del alts[idx]
        localsum -= expScores[choice]
        expArray = np.delete(expArray, idx, 1)
        features = np.delete(features, idx, 0)
    else:
      for choice in response.ranked[:-1]:
        for i in xrange(self.decision.dimension):
          p1 = np.dot(expArray, features[:,i])
          p2 = np.dot(expArray, np.square(features[:,i]))
          m[i, i] += (p1/localsum) ** 2 - p2/localsum
          for j in xrange(i+1, self.decision.dimension):
            p3 = np.dot(expArray, features[:,j])
            p4 = np.dot(expArray, np.multiply(features[:, i], features[:, j]))
            m[i,j] += p1 * p3 / (localsum ** 2) - p4/localsum
            m[j, i] = m[i, j]
        idx = alts.index(choice)
        del alts[idx]
        localsum -= expScores[choice]
        expArray = np.delete(expArray, idx, 1)
        features = np.delete(features, idx, 0)
    return np.matrix(m)

  def minCertainty(self, theta, covariance_matrix):
    # find the least certain pairwise compairison between top and all other alts, among all agents
    # returns a scaler (minimum certainty)
    pairs = list(itertools.combinations(self.decision.alternatives, 2))
    for agt in self.decision.agents:
      # ranking = self.decision.getRanking(agt, beta=theta)
      # pairs = [(ranking[0], ranking[i]) for i in range(1,len(ranking))]
      certainty = {}
      for pair in pairs:
        alt1, alt2 = pair
        feature1 = attributesToFeatures(agt.attributeValues, alt1.attributeValues)
        feature2 = attributesToFeatures(agt.attributeValues, alt2.attributeValues)
        diff_feature = [feature1[i]-feature2[i] for i in xrange(self.decision.dimension)]
        std_tmp = 0.0
        for i in xrange(self.decision.dimension):
          for j in xrange(self.decision.dimension):
            std_tmp += diff_feature[i]*diff_feature[j]*covariance_matrix[i,j]
        std = np.sqrt(std_tmp)
        mean = abs(weightedSum(diff_feature, theta))
        certainty[(agt,pair)] = mean/std
    leastCertainPair = min(certainty, key=certainty.get)
    return certainty[leastCertainPair]
