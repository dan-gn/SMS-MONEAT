#install.packages("irace", version="3.4.1")

library(irace)

setwd("C:/Users/23252359/Documents/SMS-MONEAT/iRace/sfe")

scenario <- readScenario(filename = "scenario.txt", scenario = defaultScenario())

checkIraceScenario(scenario = scenario)

irace.main(scenario = scenario)