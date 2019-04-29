#test parameter
import mypackage as la
from itertools import product


data = la.getHourlyData()
train = data['1/7/2017':'30/6/2018']

ps = range(0, 4)
d=0
qs = range(0, 4)
Ps = range(0, 0)
D=1
Qs = range(0, 4)

s = 24
parameters = product(ps, qs, Ps, Qs)

result = la.optimizeARIMA(data,parameters,0,1,24)


#test precision


