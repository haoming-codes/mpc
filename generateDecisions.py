from preferenceElicitation import preferenceElicitation
from decision import Agent, Alternative, Decision, randomAgent, randomAlternative, randomBeta
from response import Question
import random
NUM_DCSN = 500
NUM_PRE_DATA = 50

if __name__ == '__main__':
  agentNumAttributes = 3
  alternativeNumAttributes = 3
  dimension = 9
  numAlternatives = 10
  numAgents = 5
  numCrowds = 20
  costAgents = 2.0
  costCrowds = 1.0
  fname = 'generated_decisions_dirich_numAgent5c.txt'

  for i in xrange(NUM_DCSN):

    agents = [randomAgent(agentNumAttributes, costAgents) for _ in xrange(numAgents)]
    crowds = [randomAgent(agentNumAttributes, costCrowds) for _ in xrange(numCrowds)]
    alternatives = [randomAlternative(alternativeNumAttributes) for _ in xrange(numAlternatives)]
    dcsn = Decision(agents, crowds, alternatives, randomBeta(dimension))
    # responses = {agt: dcsn.getResponse(agt) for agt in dcsn.agents+dcsn.crowds}
    # assert len(responses) == numAgents+numCrowds
    pre_asked = []
    pre_data = []
    while len(pre_data) < NUM_PRE_DATA:
      agt = random.choice(crowds)
      alts = random.sample(alternatives, 2)
      q = Question(agt, set(alts), 1)
      if q in pre_asked:
        continue
      r = dcsn.getResponse(agt, alts, top=1)
      pre_data.append(r)
      pre_asked.append(q)
    
    to_write = "{'decision_id': %s, 'decision': %s, 'pre_data': %s}\n" % (i, dcsn, pre_data) 
    f = open(fname, 'a')
    f.write(to_write)
    f.close()