#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from probo.marketdata import MarketData
from probo.payoff import VanillaPayoff, call_payoff, put_payoff
from probo.engine import MonteCarloEngine, PathwiseControlVariatePricer, PathwiseNaiveMonteCarloPricer
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
thecall = VanillaPayoff(expiry, strike, call_payoff)
theput = VanillaPayoff(expiry, strike, put_payoff)


## Set up Simple Monte Carlo
nreps = 10000
steps = 10
pricer = PathwiseNaiveMonteCarloPricer
call_mcengine = MonteCarloEngine(nreps, steps, pricer, "call")
put_mcengine = MonteCarloEngine(nreps, steps, pricer, "put")

calloption = OptionFacade(thecall, call_mcengine, thedata)
call_price = calloption.price()
print("\nThe call price and standard error via Naive Monte Carlo are: " + str(call_price))

putoption = OptionFacade(theput, put_mcengine, thedata)
put_price = putoption.price()
print("\nThe put price and standard error via Naive Monte Carlo are: " + str(put_price))

## control variate
pricer = PathwiseControlVariatePricer

print("\nThe call price and standard error via Control Variate Monte Carlo are: " + str(call_price))

print("\nThe put price and standard error via Control Variate Monte Carlo are: " + str(put_price))
