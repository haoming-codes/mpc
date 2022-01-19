import numpy as np
import itertools
import random 
from helper import weightedSum, attributesToFeatures

def costNTopK(n, k):
  if n == 10 and k == 1:
    return 1.384957*k+32.24613
  elif n == 2 and k == 1:
    return 5.333818*n
  elif n == 10 and k == 9:
    return 5.333818*n
  else:
    print '???'

class Response(object):
  def __init__(self, agent, ranked, proposed):
    assert isinstance(ranked, tuple)
    assert isinstance(proposed, set)
    self.agent = agent
    self.ranked = ranked
    self.proposed = proposed

  def __repr__(self):
    return 'Response(%s, %s, %s)' % (self.agent, self.ranked, self.proposed)

  def getProbability(self, beta):
    # assert isinstance(beta, tuple)
    exp_scores = {alt: np.exp(weightedSum(beta, attributesToFeatures(self.agent.attributeValues, alt.attributeValues))) for alt in self.proposed}
    product = 1.0
    for alt in self.ranked:
      product *= (exp_scores[alt] / sum(exp_scores.values()))
      del exp_scores[alt]
    assert len(exp_scores) == (len(self.proposed)-len(self.ranked))
    return product

  def getLogProbability(self, beta):
    exp_scores = {alt: np.exp(weightedSum(beta, attributesToFeatures(self.agent.attributeValues, alt.attributeValues))) for alt in self.proposed}
    scores = {alt: weightedSum(beta, attributesToFeatures(self.agent.attributeValues, alt.attributeValues)) for alt in self.proposed}
    s = 0.0
    for alt in self.ranked:
      s += (scores[alt] - np.log(sum(exp_scores.values())))
      del exp_scores[alt]
    assert len(exp_scores) == (len(self.proposed)-len(self.ranked))
    return s

  def getPartialResponse(self, alts, top=None):
    if not top: top = len(alts)
    assert set(alts) <= set(self.ranked)
    rank_dict = {alt: self.ranked.index(alt) for alt in alts}
    partial_rank = sorted(rank_dict, key=rank_dict.get)
    return Response(self.agent, tuple(partial_rank[:top]), set(alts))

class Question(object):
  def __init__(self, agent, proposed, top):
    assert isinstance(proposed, set)
    assert isinstance(top, int)
    assert top <= len(proposed)
    self.agent = agent
    self.proposed = proposed
    self.top = top

  def __repr__(self):
    return 'Question(%s, %s, %s)' % (self.agent, self.proposed, self.top)

  def __hash__(self):
    return hash(self.__repr__())

  def __eq__(self, other):
    return self.agent == other.agent and self.proposed == other.proposed and self.top == other.top

  def __ne__(self, other):
    return not self == other

  def qType(self):
    return (len(self.proposed), self.top)

  def agentType(self):
    return self.agent.cost

  def qCost(self):
    return self.agent.cost*costNTopK(len(self.proposed), self.top)