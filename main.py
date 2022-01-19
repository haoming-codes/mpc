import sys
import helper
from preferenceElicitation import preferenceElicitation
from response import Response, Question
from decision import Agent, Alternative, Decision
from datetime import datetime

fname_data = "generated_decisions_dirich_numAgent5c.txt"
NUM_Q = 50
MAX_COST = 1000
NUM_EXP = 300
NUM_PRE_DATA = 50

assert len(sys.argv) >= 2
criteria = sys.argv[1:]
fname = str(datetime.now())
fname = "-".join(criteria) + '-' + fname.replace(' ', '-').replace(':', '-')
fname_elicitations = fname + 'elicitations_'+__file__+'.txt'

with open(fname_data) as f:
  data = f.readlines()
data = [eval(x) for x in data]
assert len(data) >= NUM_EXP
assert len(data[0]['pre_data']) >= NUM_PRE_DATA

for i in xrange(NUM_EXP):
  assert data[i]['decision_id'] == i
  dcsn = data[i]['decision']
  pre_data = data[i]['pre_data'][:NUM_PRE_DATA]
  elicitaions = {c: preferenceElicitation(dcsn) for c in criteria}
  for elicitaion in elicitaions.values():
    for d in pre_data:
      elicitaion.addData(d)

  for criterion, elicitaion in elicitaions.iteritems():
    accum_cost = 0.0
    j = 0
    while accum_cost < MAX_COST:
      print "NUM_EXP", i, "ITR", j, "cri", criterion, "accum_cost", accum_cost
      mBeta = elicitaion.MAP()
      err = helper.distance(dcsn.trueBeta, mBeta)
      assert len(elicitaion.data) == NUM_PRE_DATA+j

      Qs = {}
      nextQ, quality_over_cost = elicitaion.nextQuestion(criterion, 10, 1) # (agt, frozenset([alt]))
      if nextQ: Qs[nextQ] = quality_over_cost
      nextQ, quality_over_cost = elicitaion.nextQuestion(criterion, 2, 1) # (agt, frozenset([alt]))
      if nextQ: Qs[nextQ] = quality_over_cost
      nextQ, quality_over_cost = elicitaion.nextQuestion(criterion, 10, 9) # (agt, frozenset([alt]))
      if nextQ: Qs[nextQ] = quality_over_cost
      assert len(Qs) > 0

      nextQ = max(Qs, key=Qs.get)
      accum_cost += nextQ.qCost()
      next_agent, next_alternatives, next_top = nextQ.agent, list(nextQ.proposed), nextQ.top
      print criterion+" asked:", len(next_alternatives), 'choose', next_top, "; err =", err
      response = dcsn.getResponse(next_agent, alts=next_alternatives, top=next_top)
      # assert response.agent == nextQ.agent and response.proposed == nextQ.proposed and len(response.ranked) == nextQ.top and set(response.ranked) <= response.proposed
      elicitaion.addData(response)

      to_write = "{'decision_id': %d, 'criterion': \'%s\', 'itr': %d, 'mBeta': %s, 'error': %.18f, 'best_questions': %s, 'asked_question': %s, 'response': %s, 'accum_cost': %s}\n" % (i, criterion, j, mBeta, err, Qs, nextQ, response, accum_cost)
      f = open(fname_elicitations, 'a')
      f.write(to_write)
      f.close()
      j += 1
