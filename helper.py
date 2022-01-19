import numpy as np
import random
from numpy import array, asarray, ma, zeros

def attributesToFeatures(agentAttributes, alternativeAttributes):
  # a cross product feature transform
  return [x*y for x in agentAttributes for y in alternativeAttributes]

def prod( iterable ):
  p = 1.0
  for n in iterable:
    p *= n
  return p

def weightedSum(beta, featureValues):
  assert len(beta) == len(featureValues)
  return sum([beta[i]*featureValues[i] for i in range(len(beta))])

# def plackettLuce(ranking, exp_scores, alts=None):
#   if not alts: alts = ranking
#   exp_scores = {k:exp_scores[k] for k in exp_scores if k in alts}
#   # ranking is a list of Object; exp_scores is a dict of Object : int
#   assert len(alts) == len(exp_scores)
#   product = 1.0
#   for alt in ranking:
#     product *= (exp_scores[alt] / sum(exp_scores.values()))
#     del exp_scores[alt]
#   assert len(exp_scores) == (len(alts)-len(ranking))
#   return product

# def bradleyTerry(ranking, exp_scores):
#   assert len(ranking) == 2
#   return plackettLuce(ranking, exp_scores)

def randomByDict(dct):
  rand_val = random.random()
  total = 0
  for k, v in dct.items():
    total += v
    if rand_val <= total:
      return k
  assert False, 'unreachable'

# def addNoise(beta, noiseLevel):
#   noise = numpy.random.normal(scale=noiseLevel, size=len(beta))
#   # if beta[i] + noise <= 1.0 and beta[i] + noise >= 0.0:
#   return tuple(beta+noise)

def discreteDirichlet(n,s,d):
  return [tuple(float(j)/100 for j in i) for i in list(discreteDirichletHelper(n, int(s*100), int(d*100)))]

def discreteDirichletHelper(n,s,d):
  if n == 1:
    yield (s,)
  else:
    for i in xrange(0, s+1, d):
      for j in discreteDirichletHelper(n - 1,s - i,d):
        yield (i,) + j

def distance(a, b):
  assert len(a) == len(b)
  return np.linalg.norm(np.array(a)-np.array(b))

def topKOverlap(l1, l2, k1=3, k2=3):
  assert len(l1) >= k1
  assert len(l2) >= k2
  count = 0
  for item in l1[:k1]:
    if item in l2[:k2]:
      count += 1
  return count

def rank_distance(x, y, weights=None, method='spearman'):
  """
  Distance measure between rankings.
  Parameters
  ----------
  x, y: array-like
    1-D permutations of [1..N] vector
  weights: array-like, optional
    1-D array of weights. Default None equals to unit weights.
  method: {'spearman'm 'kendalltau'}, optional
    Defines the method to find distance:
    'spearman' - sum of absolute distance between same elements
    in x and y.
    'kendalltau' - number of inverions needed to bring x to y.
    Default is 'spearman'.
  Return
  ------
  distance: float
    Distance between x and y.
  Example
  -------
  >>> from scipy.stats import rank_distance
  >>> rank_distance([1,3,4,2],[2,3,1,4])
  6.0
  >>> rank_distance([1,3,4,2],[2,3,1,4], method='kendalltau')
  4.0
  """
  x = np.asarray(x)
  y = np.asarray(y)
  if np.unique(x).size != x.size or np.unique(y).size != y.size:
    raise ValueError("x and y must have only unique elements")
  if x.size != y.size:
    raise ValueError("x and y have different size")
  if weights is None:
    weights = np.ones(x.size - 1)
  else:
    weights = np.asarray(weights)
    if weights.size < (x.size - 1):
      raise ValueError("weights vector have a small size")
  if method == 'spearman':
    return _spearman_footrule(x, y, weights)
  elif method == 'kendalltau':
    return _kendalltau_distance(x, y, weights)
  else:
    raise ValueError("unknown value for method parameter.")

def _spearman_footrule(x, y, weights):
  distance = 0
  for i in xrange(x.size):
    x_index = np.where(x == x[i])[0][0]
    y_index = np.where(y == x[i])[0][0]
    pair = np.abs(x_index - y_index)
    min_index = np.minimum(x_index, y_index)
    for j in xrange(pair):
      distance += weights[min_index + j]
  return distance

def _kendalltau_distance(x, y, weights):
  distance = 0
  n = x.size - 1
  for i in xrange(n - 1, -1, -1):
    key = x[i]
    j = i + 1
    while j <= n and np.where(y == key)[0] > np.where(y == x[j])[0]:
      x[j - 1] = x[j]
      distance += weights[j - 1]
      j += 1
    x[j - 1] = key
  return distance