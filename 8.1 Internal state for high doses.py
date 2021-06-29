'''
-----------------------------------------------------------------------------------------------------------------------------------
--                                                                                                                               --
--                                                                                                                               --
--                                    Escalation of Cocaine-Seeking                                                              --
--                                              in the                                                                           -- 
--                             Homeostatic Reinforcement Learning Framewrok                                                      --  
--                                                                                                                               --
--                                                                                                                               --
--                                                                                                                               --
--      Programmed in : Python 2.6                                                                                               --
--      By            : Mehdi Keramati                                                                                           -- 
--      Date          : March 2013                                                                                            --
--                                                                                                                               --
-----------------------------------------------------------------------------------------------------------------------------------
'''

import scipy
import numpy
import pylab
import cmath

'''
###################################################################################################################################
##################################################################################################################################
#                                                         Functions                                                               #
###################################################################################################################################
###################################################################################################################################
'''
'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Setting the transition function of the MDP   --------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''
def setTransition(state,action,nextState,transitionProbability):
    transition [state][action][nextState] = transitionProbability
    return 

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Setting the outcome function of the MDP   -----------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''
def setOutcome(state,action,nextState,out):
    outcome [state][action][nextState] = out
    return 

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Setting the non-homeostatic reward function of the MDP   --------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''
def setNonHomeostaticReward(state,action,nextState,rew):
    nonHomeostaticReward [state][action][nextState] = rew
    return 

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Return the probability of the transitions s-a->s'  -------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def getTransition(s,a,nextS):
    return transition[s][a][nextS]

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Return the next state that the animal fell into  ---------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def getRealizedTransition(state,action):
           
    index = numpy.random.uniform(0,1)
    probSum = 0
    for nextS in range(0,statesNum):
        probSum = probSum + getTransition(state,action,nextS)
        if index <= probSum:
            return nextS    

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Obtained outcome   ---------------------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def getOutcome(state,action,nextState):
    return outcome[state,action,nextState]

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Obtained non-homeostatic reward    ------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''
def getNonHomeostaticReward(state,action,nextState):
    return nonHomeostaticReward [state][action][nextState] 

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Homeostatically-regulated Reward   ------------------------------------------------------------ 
--------------------------------------------------------------------------------------------------------------------------------'''
def driveReductionReward(inState,setpointS,outcome):
    d1 = numpy.power(numpy.absolute(numpy.power(setpointS-inState,n*1.0)),(1.0/m))
    d2 = numpy.power(numpy.absolute(numpy.power(setpointS-inState-outcome,n*1.0)),(1.0/m))
    return d1-d2

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Create a new animal   ------------------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def initializeAnimal():    
            
    state[0] = initialExState
    state[1] = initialInState
    state[2] = initialSetpoint 
    state[3] = 0 
        
    for i in range(0,statesNum):
        for j in range(0,actionsNum):
            for k in range(0,statesNum):
                estimatedTransition            [i][j][k] = 0.0
                estimatedOutcome               [i][j][k] = 0.0
                estimatedNonHomeostaticReward  [i][j][k] = 0.0

    estimatedTransition    [0][0][0] = 1
    estimatedTransition    [0][1][1] = 1
    estimatedTransition    [0][2][1] = 1
    for i in range(1,statesNum-1):
        estimatedTransition    [i][0][i+1] = 1
        estimatedTransition    [i][1][i+1] = 1
        estimatedTransition    [i][2][i+1] = 1
    estimatedTransition    [statesNum-1][0][0] = 1
    estimatedTransition    [statesNum-1][1][0] = 1
    estimatedTransition    [statesNum-1][2][0] = 1
    
#    Assuming that the animals know the energy cost (fatigue) of pressing a lever 
    for i in range(0,statesNum):
        for j in range(0,statesNum):
            estimatedNonHomeostaticReward       [i][1][j] = -leverPressCost
            estimatedNonHomeostaticReward       [i][2][j] = -leverPressCost

    estimatedOutcome                [0][1][1]     = cocaine
    estimatedNonHomeostaticReward   [0][2][1]     = -leverPressCost+foodReward

    
    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Is action a available is state s?   ----------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def isActionAvailable(state,action):
    probSum = 0 ;
    for i in range(0,statesNum):
        probSum = probSum + getTransition(state,action,i)
    if probSum == 1:
        return 1
    elif probSum == 0:
        return 0
    else:
        print "Error: There seems to be a problem in defining the transition function of the environment"        
        return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Goal-directed Value estimation   -------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def valueEstimation(state,inState,setpointS,depthLeft):

    values = numpy.zeros ( [actionsNum] , float )

    # If this is the last depth that should be searched :
    if depthLeft==1:
        for action in range(0,actionsNum):
            for nextState in range(0,statesNum):
                homeoReward    = driveReductionReward(inState,setpointS,estimatedOutcome[state][action][nextState])
                nonHomeoReward = estimatedNonHomeostaticReward[state][action][nextState]
                transitionProb = estimatedTransition[state][action][nextState]
                values[action] = values[action] +  transitionProb * ( homeoReward + nonHomeoReward )
        return values
    
    # Otherwise :
    for action in range(0,actionsNum):
        for nextState in range(0,statesNum):
            if estimatedTransition[state][action][nextState] < pruningThreshold :
                VNextStateBest = 0
            else:    
                VNextState = valueEstimation(nextState,setpointS,inState,depthLeft-1)
                VNextStateBest = maxValue (VNextState)
            homeoReward    = driveReductionReward(inState,setpointS,estimatedOutcome[state][action][nextState])
            nonHomeoReward = estimatedNonHomeostaticReward[state][action][nextState]
            transitionProb = estimatedTransition[state][action][nextState]
            values[action] = values[action] + transitionProb * ( homeoReward + nonHomeoReward + gamma*VNextStateBest ) 
            
    return values
    
'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Max ( Value[nextState,a] ) : for all a  ------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def maxValue(V):
    maxV = V[0]
    for action in range(0,actionsNum):
        if V[action]>maxV:
            maxV = V[action]    
    return maxV
    
'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Action Selection : Softmax   ------------------------------------------------------------------ 
--------------------------------------------------------------------------------------------------------------------------------'''
def actionSelectionSoftmax(state,V):

    # Normalizing values, in order to be overflow due to very high values
    maxV = V[0]
    if maxV==0:
        maxV=1        
    for action in range(0,actionsNum):
        if maxV < V[action]:
            maxV = V[action]
    for action in range(0,actionsNum):
        V[action] = V[action]/maxV


    sumEV = 0
    for action in range(0,actionsNum):
        sumEV = sumEV + abs(cmath.exp( V[action] / beta ))

    index = numpy.random.uniform(0,sumEV)

    probSum=0
    for action in range(0,actionsNum):
            probSum = probSum + abs(cmath.exp( V[action] / beta ))
            if probSum >= index:
                return action

    print "Error: An unexpected (strange) problem has occured in action selection..."
    return 0
    
'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Update internal state upon consumption   ------------------------------------------------------ 
--------------------------------------------------------------------------------------------------------------------------------'''
def updateInState(inState,outcome):
    interS = inState + outcome - cocaineDegradationRate*(inState-inStateLowerBound)
    if interS<inStateLowerBound:
        interS=inStateLowerBound
    return interS

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Update the homeostatic setpoint (Allostatic mechanism)   -------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def updateSetpoint(optimalInState,out):
    
    optInS = optimalInState + out*setpointShiftRate - setpointRecoveryRate

    if optInS<optimalInStateLowerBound:
        optInS=optimalInStateLowerBound

    if optInS>optimalInStateUpperBound:
        optInS=optimalInStateUpperBound

    return optInS

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Update the expected-outcome function  --------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def updateOutcomeFunction(state,action,nextState,out):
    estimatedOutcome[state][action][nextState] = (1.0-updateOutcomeRate)*estimatedOutcome[state][action][nextState] + updateOutcomeRate*out
    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Update the expected-non-homeostatic-reward function  ------------------------------------------ 
--------------------------------------------------------------------------------------------------------------------------------'''
def updateNonHomeostaticRewardFunction(state,action,nextState,rew):
    estimatedNonHomeostaticReward[state][action][nextState] = (1.0-updateRewardRate)*estimatedNonHomeostaticReward[state][action][nextState] + updateRewardRate*rew
    return


'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Update the expected-transition function  ------------------------------------------------------ 
--------------------------------------------------------------------------------------------------------------------------------'''
def updateTransitionFunction(state,action,nextState):

    #---- First inhibit all associations
    for i in range(0,statesNum):
        estimatedTransition[state][action][i] = (1.0-updateTransitionRate)*estimatedTransition[state][action][i]
    
    #---- Then potentiate the experiences association
    estimatedTransition[state][action][nextState] = estimatedTransition[state][action][nextState] + updateTransitionRate
            
    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Cocaine Seeking Sessions  --------------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def cocaineSeeking  (sessionNum , ratType):

    exState     = state[0]
    inState     = state[1]
    setpointS   = state[2]
    trialCount  = state[3]
    cocBuffer   = 0
    
    if ratType=='ShA':  
        trialsNum = seekingTrialsNumShA    
    if ratType=='LgA':  
        trialsNum = seekingTrialsNumLgA    
    
    for trial in range(trialCount,trialCount+trialsNum):

        estimatedActionValues   = valueEstimation(exState,inState,setpointS,searchDepth)
        action                  = actionSelectionSoftmax(exState,estimatedActionValues)
        nextState               = getRealizedTransition(exState,action)
        out                     = getOutcome(exState,action,nextState)
        nonHomeoRew             = getNonHomeostaticReward(exState,action,nextState)
        HomeoRew                = driveReductionReward(inState,setpointS,out)

        if ratType=='ShA':  
            loggingShA(trial,action,inState,setpointS,out)    
            print "ShA rat number: %d / %d     Session Number: %d / %d     trial: %d / %d      animal seeking cocaine" %(animal+1,animalsNum,sessionNum+1,sessionsNum,trial-trialCount+1,trialsNum)
        if ratType=='LgA':  
            loggingLgA(trial,action,inState,setpointS,out)    
            print "LgA rat number: %d / %d     Session Number: %d / %d     trial: %d / %d      animal seeking cocaine" %(animal+1,animalsNum,sessionNum+1,sessionsNum,trial-trialCount+1,trialsNum)

        updateOutcomeFunction(exState,action,nextState,out)
        updateNonHomeostaticRewardFunction(exState,action,nextState,nonHomeoRew)
        updateTransitionFunction(exState,action,nextState)            
        
        cocBuffer = cocBuffer + out                
        
        inState     = updateInState(inState,cocBuffer*cocAbsorptionRatio)
        setpointS   = updateSetpoint(setpointS,out)

        cocBuffer = cocBuffer*(1-cocAbsorptionRatio)

        exState   = nextState

    state[0]    = exState
    state[1]    = inState
    state[2]    = setpointS
    state[3]    = trialCount+trialsNum

    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Home-cage Sessions  --------------------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def homeCage (sessionNum, ratType):

    exState     = state[0]
    inState     = state[1]
    setpointS   = state[2]
    trialCount  = state[3]
 
    if ratType=='ShA':  
        trialsNum = restTrialsNumShA    
        print "ShA rat number: %d / %d     Session Number: %d / %d                          animal rests in home cage" %(animal+1,animalsNum,sessionNum+1,sessionsNum)
    elif ratType=='LgA':  
        trialsNum = restTrialsNumLgA
        print "LgA rat number: %d / %d     Session Number: %d / %d                          animal rests in home cage" %(animal+1,animalsNum,sessionNum+1,sessionsNum)
    elif ratType=='afterPretrainingShA':  
        trialsNum = restAfterPretrainingTrialsNum    
        print "ShA rat number: %d / %d     After pretraining                                animal rests in home cage" %(animal+1,animalsNum)
    elif ratType=='afterPretrainingLgA':  
        trialsNum = restAfterPretrainingTrialsNum    
        print "LgA rat number: %d / %d     After pretraining                                animal rests in home cage" %(animal+1,animalsNum)
     
    for trial in range(trialCount,trialCount+trialsNum):

        inState     = updateInState(inState,0)
        setpointS   = updateSetpoint(setpointS,0)

        if ratType=='ShA':  
            loggingShA(trial,0,inState,setpointS,0)    
        elif ratType=='LgA':  
            loggingLgA(trial,0,inState,setpointS,0)    
        elif ratType=='afterPretrainingShA':  
            loggingShA(trial,0,inState,setpointS,0)    
        elif ratType=='afterPretrainingLgA':  
            loggingLgA(trial,0,inState,setpointS,0)    

    state[0]    = exState
    state[1]    = inState
    state[2]    = setpointS
    state[3]    = trialCount+trialsNum

    return

    
'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Logging the current information for the Short-access group  ----------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def loggingShA(trial,action,inState,setpointS,coca):
   
    if action==0: 
        nulDoingShA[trial]             = nulDoingShA[trial] + 1
    elif action==1: 
        inactiveLeverPressShA[trial]   = inactiveLeverPressShA[trial] + 1
    elif action==2: 
        activeLeverPressShA[trial]     = activeLeverPressShA[trial] + 1
    internalStateShA[trial]    = internalStateShA[trial] + inState
    setpointShA[trial]         = setpointShA[trial] + setpointS    
    if coca==cocaine:
        infusionShA[trial]     = infusionShA[trial] + 1
    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Logging the current information for the Long-access group  ------------------------------------ 
--------------------------------------------------------------------------------------------------------------------------------'''
def loggingLgA(trial,action,inState,setpointS,coca):
   
    if action==0: 
        nulDoingLgA[trial]             = nulDoingLgA[trial] + 1
    elif action==1: 
        inactiveLeverPressLgA[trial]   = inactiveLeverPressLgA[trial] + 1
    elif action==2: 
        activeLeverPressLgA[trial]     = activeLeverPressLgA[trial] + 1
    internalStateLgA[trial]    = internalStateLgA[trial] + inState
    setpointLgA[trial]         = setpointLgA[trial] + setpointS    
    if coca==cocaine:
        infusionLgA[trial]     = infusionLgA[trial] + 1
    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Wrap up all the logged data   ----------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def loggingFinalization():
    
    for trial in range(0,totalTrialsNum):
        nulDoingShA[trial]             = nulDoingShA[trial]/animalsNum
        inactiveLeverPressShA[trial]   = inactiveLeverPressShA[trial]/animalsNum
        activeLeverPressShA[trial]     = activeLeverPressShA[trial]/animalsNum
        internalStateShA[trial]        = internalStateShA[trial]/animalsNum
        setpointShA[trial]             = setpointShA[trial]/animalsNum  
        infusionShA[trial]             = infusionShA[trial]/animalsNum 

        nulDoingLgA[trial]             = nulDoingLgA[trial]/animalsNum
        inactiveLeverPressLgA[trial]   = inactiveLeverPressLgA[trial]/animalsNum
        activeLeverPressLgA[trial]     = activeLeverPressLgA[trial]/animalsNum
        internalStateLgA[trial]        = internalStateLgA[trial]/animalsNum
        setpointLgA[trial]             = setpointLgA[trial]/animalsNum  
        infusionLgA[trial]             = infusionLgA[trial]/animalsNum 

    return


'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plot the internal state of the last session  -------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotInternalStateLastSession():

    font = {'family' : 'normal', 'size'   : 16}
    pylab.rc('font', **font)
    pylab.rcParams.update({'legend.fontsize': 16})
          
    fig1 = pylab.figure( figsize=(8,3.5) )
    fig1.subplots_adjust(top=0.85)
    fig1.subplots_adjust(left=0.16)
    fig1.subplots_adjust(bottom=0.2)

    ax1 = fig1.add_subplot(111)
    S1 = ax1.plot(internalStateLgA [ 0 : 900 ] , linewidth = 2.5 , color='black' )
  
    ax1.axhline(100,  color='0.25',ls='--', lw=1 )

    pylab.ylim((-10 , 210))
    pylab.xlim((-20,920))
 
    tick_lcs = []
    tick_lbs = []
    for i in range ( 0 , 13 ):
        tick_lcs.append( i*75 ) 
        tick_lbs.append(i*5)
    pylab.xticks(tick_lcs, tick_lbs)

    for line in ax1.get_yticklines() + ax1.get_xticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)

 
    ax1.set_ylabel('Internal State')
    ax1.set_xlabel('Time (min)')
    ax1.set_title('Cocaine dose (mg/infusion): %.2f ' % (cocaine*0.005))
    fig1.savefig('Internal state'+ '%.2f' % (cocaine*0.005) +'.eps', format='eps')

    return


'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plot the infusions for the last session ------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotInfusionLastSession():

#---------------------------------ShA rats
        
    fig1 = pylab.figure( figsize=(8,2) )
    fig1.subplots_adjust(top=0.8)
    fig1.subplots_adjust(bottom=0.3)
    fig1.subplots_adjust(left=0.16)

    ax1 = fig1.add_subplot(111)
    S0 = ax1.plot(infusionLgA [0:900], linewidth = 2 , color='black' )
    
#    pylab.yticks(pylab.arange(0, 1.01, 0.2))
    pylab.ylim((0,2.5))

    pylab.xlim((0,901))
 
    tick_lcs = []
    tick_lbs = []
    pylab.yticks(tick_lcs, tick_lbs)
    for i in range ( 0 , 13 ):
        tick_lcs.append(i*75 ) 
        tick_lbs.append(i*5)
    pylab.xticks(tick_lcs, tick_lbs)

    for line in ax1.get_yticklines() + ax1.get_xticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)

    ax1.set_ylabel('Infusion')
    ax1.set_xlabel('Time (min)')
    ax1.set_title('Cocaine dose (mg/infusion): %.2f ' % (cocaine*0.005))
    fig1.savefig('infusions' + '%.2f' % (cocaine*0.005) + '.eps', format='eps')

    return


'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plot the infusions for the last session ------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def printInfusionRate():
    infRate = 0
    for j in range(0,seekingTrialsNumLgA):
        infRate = infRate + infusionLgA[j]
        
    infRate = infRate/seekingHoursLgA
    
    print infRate
    
    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plot all the results  ------------------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotting():

    loggingFinalization()

    plotInternalStateLastSession()
    plotInfusionLastSession()
    printInfusionRate()
    
    pylab.show()   
    
    return

'''
###################################################################################################################################
###################################################################################################################################
#                                                             Main                                                                #
###################################################################################################################################
###################################################################################################################################
'''
'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Definition of the Markov Decison Process FR1 - Timeout 20sec  ---------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''

cocaine         = 13 + (1.24)**13 # Dose of self-administered drug
leverPressCost  = 150             # Energy cost for pressing the lever
foodReward      = 150

statesNum       = 35              # number of stater 
actionsNum      = 3              # number of action   action 0 = Null     action 1 = Inactive Lever Press    action 2 = Active Lever Press
initialExState  = 0

transition = numpy.zeros( [statesNum , actionsNum, statesNum] , float)
setTransition(0,0,0,1)          # From state s, and by taking a, we go to state s', with probability p
setTransition(0,1,1,1)
setTransition(0,2,1,1)
for i in range(1,statesNum-1):
    setTransition(i,0,i+1,1)
    setTransition(i,1,i+1,1)
    setTransition(i,2,i+1,1)
setTransition(statesNum-1,0,0,1)
setTransition(statesNum-1,1,0,1)
setTransition(statesNum-1,2,0,1)

outcome = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )
setOutcome(0,1,1,cocaine)       # At state s, by doing action a and going to state s', we receive the outcome 

nonHomeostaticReward = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )
setNonHomeostaticReward(0,1,1,-leverPressCost)
setNonHomeostaticReward(0,2,1,-leverPressCost+foodReward)
for i in range(1,statesNum-1):
    setNonHomeostaticReward(i,1,i+1,-leverPressCost)
    setNonHomeostaticReward(i,2,i+1,-leverPressCost)
setNonHomeostaticReward(15,1,0,-leverPressCost)
setNonHomeostaticReward(15,2,0,-leverPressCost)


'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Definition of the Animal   --------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''

#------------ Homeostatic System
initialInState          = 0
#initialSetpoint         = 200
initialSetpoint         = 125
inStateLowerBound       = 0
cocaineDegradationRate  = 0.007    # Dose of cocaine that the animal loses in every time-step
cocAbsorptionRatio      = 0.12      # Proportion of the injected cocaine that affects the brain right after infusion 

#------------ Allostatic System
setpointShiftRate       = 0.0018
setpointRecoveryRate    = 0.00016
optimalInStateLowerBound= 100
optimalInStateUpperBound= 200

#------------ Drive Function
m                       = 3     # Parameter of the drive function : m-th root
n                       = 4     # Parameter of the drive function : n-th pawer

#------------ Goal-directed system
updateOutcomeRate       = 0.2  # Learning rate for updating the outcome function
updateTransitionRate    = 0.2  # Learning rate for updating the transition function
updateRewardRate        = 0.2  # Learning rate for updating the non-homeostatic reward function
gamma                   = 1     # Discount factor
beta                    = 10     # Rate of exploration
searchDepth             = 1     # Depth of going into the decision tree for goal-directed valuation of choices
pruningThreshold        = 0.1   # If the probability of a transition like (s,a,s') is less than "pruningThreshold", cut it from the decision tree 

estimatedTransition              = numpy.zeros( [statesNum , actionsNum, statesNum] , float)
estimatedOutcome                 = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )
estimatedNonHomeostaticReward    = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )

state                            = numpy.zeros ( [4] , float )     # a vector of the external state, internal state, setpoint, and trial

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Simulation Parameters   -----------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''

animalsNum          = 1                                  # Number of animals

pretrainingHours    = 0
sessionsNum         = 1                                  # Number of sessions of cocain seeking, followed by rest in home-cage
seekingHoursShA     = 1            
seekingHoursLgA     = 1            
extinctionHours     = 0

trialsPerHour       = 60*60/4                            # Number of trials during one hour (as each trial is supposed to be 4 seconds)
trialsPerDay        = 24*trialsPerHour
pretrainingTrialsNum= pretrainingHours* trialsPerHour
restAfterPretrainingTrialsNum = (24 - pretrainingHours) *trialsPerHour

seekingTrialsNumShA = seekingHoursShA * trialsPerHour    # Number of trials for each cocaine seeking session
restingHoursShA     = 24 - seekingHoursShA
restTrialsNumShA    = restingHoursShA * trialsPerHour    # Number of trials for each session of the animal being in the home cage
extinctionTrialsNum = extinctionHours*trialsPerHour      # Number of trials for each extinction session

seekingTrialsNumLgA = seekingHoursLgA * trialsPerHour    # Number of trials for each cocaine seeking session
restingHoursLgA     = 24 - seekingHoursLgA
restTrialsNumLgA    = restingHoursLgA * trialsPerHour    # Number of trials for each session of the animal being in the home cage
extinctionTrialsNum = extinctionHours*trialsPerHour      # Number of trials for each extinction session

totalTrialsNum      =  sessionsNum * (trialsPerDay)  #+ extinctionTrialsNum*2 + 1

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plotting Parameters   -------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''

trialsPerBlock = 10*60/4            # Each BLOCK is 10 minutes - Each minute 60 second - Each trial takes 4 seconds

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Logging Parameters   --------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''

nulDoingShA            = numpy.zeros( [totalTrialsNum] , float)
inactiveLeverPressShA  = numpy.zeros( [totalTrialsNum] , float)
activeLeverPressShA    = numpy.zeros( [totalTrialsNum] , float)
internalStateShA       = numpy.zeros( [totalTrialsNum] , float)
setpointShA            = numpy.zeros( [totalTrialsNum] , float)
infusionShA            = numpy.zeros( [totalTrialsNum] , float)

nulDoingLgA            = numpy.zeros( [totalTrialsNum] , float)
inactiveLeverPressLgA  = numpy.zeros( [totalTrialsNum] , float)
activeLeverPressLgA    = numpy.zeros( [totalTrialsNum] , float)
internalStateLgA       = numpy.zeros( [totalTrialsNum] , float)
setpointLgA            = numpy.zeros( [totalTrialsNum] , float)
infusionLgA            = numpy.zeros( [totalTrialsNum] , float)

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Simulation   ----------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''

animal=0

initializeAnimal        (   )

for session in range(0,sessionsNum):
    cocaineSeeking        (  session , 'LgA'        )
    homeCage              (  session , 'LgA'        ) 


plotting()

