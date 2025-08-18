this repository containts my effort on creating an enviroment that preforms the following:
1. i have an agent which plays the game rock-sample
2. i have an oracle which tries to:
    guess which state and belief state the agent is at
    estimate what the agent will do and send him information when you decide it would benefit the agent

i am having some issues with my code and i want to to preform the following:
1. first we want to fix my issues:
pomcp agent does not act "buy information" a special action we insert in simulation mode only
we need him to do it from time to time as we need to use it in order to understand when the oracle should interveene

so this is our biggest issue, if we dont solve it our research is doomed! can you help?