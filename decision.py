import numpy as np
import random
import helper
import itertools
from response import Response
from helper import attributesToFeatures

def randomAgent(numAttributes, c=1.0):
  attributeValues = np.random.normal(loc=0, scale=1, size=numAttributes)
  return Agent(tuple(attributeValues), cost=c)

def randomAlternative(numAttributes):
  attributeValues = np.random.normal(loc=0, scale=1, size=numAttributes)
  return Alternative(tuple(attributeValues))

def randomBeta(dimension):
  beta = np.random.dirichlet(np.ones(dimension))
  return tuple(beta)

class Decision(object):
  def __init__(self, agents, crowds, alternatives, beta):
    self.agents = agents
    self.crowds = crowds
    self.alternatives = alternatives
    self.dimension = len(attributesToFeatures(agents[0].attributeValues, alternatives[0].attributeValues))
    self.trueBeta = beta

  def __repr__(self):
    return 'Decision(%s, %s, %s, %s)' % (self.agents, self.crowds, self.alternatives, self.trueBeta)

  def getUtility(self, agt, alt, beta=None):
    if not beta: beta = self.trueBeta
    # returns a utility value = weighted sum
    featureValues = attributesToFeatures(agt.attributeValues, alt.attributeValues)
    return helper.weightedSum(beta, featureValues)

  def setTrueBeta(self, beta):
    assert isinstance(beta, tuple)
    assert len(beta) == self.dimension
    self.trueBeta = beta

  def getResponse(self, agt, alts=None, beta=None, top=None):
    if not alts: alts = self.alternatives
    if not beta: beta = self.trueBeta
    if not top: top = len(alts)

    all_alts = set(alts)
    exp_scores = np.array([np.exp(self.getUtility(agt, alt, beta)) for alt in alts])
    vote = []
    for _ in xrange(top):
      exp_scores /= np.sum(exp_scores)
      exp_scores_intervals = np.copy(exp_scores)
      prev = 0.0
      for k in xrange(len(exp_scores_intervals)):
        exp_scores_intervals[k] += prev
        prev = exp_scores_intervals[k]
      selection = np.random.random()
      for l in xrange(len(exp_scores_intervals)): # determine position
        if selection <= exp_scores_intervals[l]:
          vote.append(alts[l])
          exp_scores = np.delete(exp_scores, l) # remove that gamma
          alts = np.delete(alts, l) # remove the alternative
          break
    response = Response(agt, tuple(vote), all_alts)
    return response

  def getRanking(self, agt, alts=None, beta=None):
    if not alts: alts = self.alternatives
    if not beta: beta = self.trueBeta
    # returns a list of Alternative, deterministically
    currentUtils = {} # dictionary of Alternative : avg utility value
    for alt in alts:
      currentUtils[alt] = self.getUtility(agt, alt, beta)
    return sorted(currentUtils, key=currentUtils.get, reverse=True)

  def totalVariationDistance(self, mBeta, dis_type):
    trueBeta = self.trueBeta
    agents = self.agents
    alts = self.alternatives
    
    current_top_score = np.zeros(len(alts))
    true_top_score = np.zeros(len(alts))
    if dis_type == 'plurality':
      for agt in agents:
        current_top_score += np.array([Response(agt, (alt,), set(alts)).getProbability(mBeta) for alt in alts])
        true_top_score += np.array([Response(agt, (alt,), set(alts)).getProbability(trueBeta) for alt in alts])
    elif dis_type == 'borda':
      for agt in agents:
        current_top_score += np.array([sum([Response(agt, (alt1,), set([alt1, alt2])).getProbability(mBeta) for alt2 in alts if alt1 != alt2]) for alt1 in alts])
        true_top_score += np.array([sum([Response(agt, (alt1,), set([alt1, alt2])).getProbability(trueBeta) for alt2 in alts if alt1 != alt2]) for alt1 in alts])

    current_top_score = current_top_score/sum(current_top_score)
    true_top_score = true_top_score/sum(true_top_score)
    distance = np.sum(np.absolute(current_top_score - true_top_score))/len(agents)/2
    return distance

  # def randomPairwiseComparisons(self, alts=None, size=None):
  #   assert self.trueBeta
  #   if not alts: alts = self.alternatives
  #   if not size:
  #     result = self.getResponse(agt=random.choice(self.agents),alts=random.sample(alts, 2))
  #   else:
  #     result = [self.getResponse(agt=random.choice(self.agents),alts=random.sample(alts, 2)) for _ in xrange(size)]
  #   return result

class Agent(object):
  attributes = None
  # numAttributes = None

  def __init__(self, attributeValues, name=None, cost=1.0):
    self.name = str(id(attributeValues)) if name is None else name
    self.attributeValues = attributeValues
    self.cost = cost
    # assert len(attributeValues) == self.numAttributes
    assert isinstance(self.name, str)
    assert isinstance(self.attributeValues, tuple)

  def __repr__(self):
    return 'Agent(%s, \'%s\', %s)' % (self.attributeValues, self.name, self.cost)

  def __hash__(self):
    return hash(self.__repr__())

  def __eq__(self, other):
    return self.name == other.name

  def __ne__(self, other):
        return not self == other

class Alternative(object):
  attributes = None
  # numAttributes = None

  def __init__(self, attributeValues, name=None):
    self.name = str(id(attributeValues)) if name is None else name
    self.attributeValues = attributeValues
    # assert len(attributeValues) == self.numAttributes
    assert isinstance(self.name, str)
    assert isinstance(self.attributeValues, tuple)

  def __repr__(self):
    return 'Alternative(%s, \'%s\')' % (self.attributeValues, self.name)

  def __hash__(self):
    return hash(self.__repr__())

  def __eq__(self, other):
    return self.name == other.name

  def __ne__(self, other):
    return not self == other

# if __name__ == '__main__':

#   l = discreteDirichlet(10,1,0.1)
#   print l

  # # Define the problem
  # Agent.numAttributes = 3
  # Agent.attributes = ['agent_cons', 'age', 'income']
  # Alternative.numAttributes = 3
  # Alternative.attributes = ['alternative_cons', 'revenue', 'employee']
  

  # # Define the agents
  # agent =           Agent((1, 1, 1), 'Haoming')
  # alternatives =   [Alternative((1, 0.80, 0.20), 'A'),
  #                   Alternative((1, 0.75, 0.25), 'B'),
  #                   Alternative((1, 0.25, 0.75), 'C'),
  #                   Alternative((1, 0.20, 0.80), 'D'),
  #                   Alternative((1, 0.50, 0.50), 'E')]
  # dcsn = Decision(agent, alternatives)
  # print dcsn.getRanking([1,2,3,4,5,6,7,8,9])
  # pf = ParticleFilter(NUM_PARTICLES)
  # pf.initializeUniformly(Agent.numAttributes*Alternative.numAttributes)

  # question = 'Tell me a comparison (e.g. A > B): '
  # while True:
  #   userInput = raw_input(question)
  #   partialRanking = userInput.split(" > ") # ranking is a list of 
  #   pf.observe(partialRanking, agent, alternatives)
  #   currentRanking = pf.currentRanking(agent, alternatives)
  #   print 'Current ranking: %s' % ' > '.join(map(str, currentRanking))
  #   nextQuestion = pf.leastCertainPair(agent, alternatives)
  #   question = '\nCompare %s and %s: ' % (nextQuestion)