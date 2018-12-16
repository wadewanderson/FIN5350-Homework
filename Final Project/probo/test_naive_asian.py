from probo.marketdata import MarketData
from probo.payoff import ExoticPayoff, arithmeticAsianCallPayoff, arithmeticAsianPutPayoff, GeometricAsianCallPayoff, GeometricAsianPutPayoff
from probo.engine import MonteCarloEngine, GeometricBlackScholesPricingEngine, PathwiseNaiveMonteCarloPricer, \
GeoAsianCallBSMPricer, GeoAsianPutBSMPricer
from probo.facade import OptionFacade
## Set up the market data
spot = 100
rate = 0.06
volatility = 0.20
dividend = 0.03
thedata = MarketData(rate, spot, volatility, dividend)
## Set up the option
expiry = 1.0
strike = 100.0
thecall = ExoticPayoff(expiry, strike, arithmeticAsianCallPayoff)
theput = ExoticPayoff(expiry, strike, arithmeticAsianPutPayoff)


## Set up Simple Monte Carlo
nreps = 100000
steps = 10
pricer = PathwiseNaiveMonteCarloPricer
mcengine = MonteCarloEngine(nreps, steps, pricer)

calloption = OptionFacade(thecall, mcengine, thedata)
call_price = calloption.price()
print("The call price via Naive Monte Carlo is: {0:.3f}".format(call_price))

putoption = OptionFacade(theput, mcengine, thedata)
put_price = putoption.price()
print(put_price)

## control variate







