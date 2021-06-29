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
from scipy.optimize import curve_fit


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
    estimatedTransition    [1][0][2] = 1
    estimatedTransition    [1][1][2] = 1
    estimatedTransition    [2][0][3] = 1
    estimatedTransition    [2][1][3] = 1
    estimatedTransition    [3][0][4] = 1
    estimatedTransition    [3][1][4] = 1
    estimatedTransition    [4][0][0] = 1
    estimatedTransition    [4][1][0] = 1
    
#    Assuming that the animals know the energy cost (fatigue) of pressing a lever 
    for i in range(0,statesNum):
        for j in range(0,statesNum):
            estimatedNonHomeostaticReward         [i][1][j] = -leverPressCost

    estimatedOutcome [0][1][1]     = cocaine
    
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
#            print "LgA rat number: %d / %d     Session Number: %d / %d     trial: %d / %d      animal seeking cocaine" %(animal+1,animalsNum,sessionNum+1,sessionsNum,trial-trialCount+1,trialsNum)

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
#        print "LgA rat number: %d / %d     Session Number: %d / %d                          animal rests in home cage" %(animal+1,animalsNum,sessionNum+1,sessionsNum)
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
---------------------------------   Reset Logging information of LgA   ------------------------------------------------------------ 
--------------------------------------------------------------------------------------------------------------------------------'''
def resetLogInfoLgA():    

    for i in range(0,totalTrialsNum):
        nulDoingLgA            [i] = 0
        inactiveLeverPressLgA  [i] = 0
        activeLeverPressLgA    [i] = 0
        internalStateLgA       [i] = 0
        setpointLgA            [i] = 0
        infusionLgA            [i] = 0    

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

cocaine         = 30             # Dose of self-administered drug
leverPressCost  = 150             # Energy cost for pressing the lever

statesNum       = 6              # number of stater 
actionsNum      = 2              # number of action   action 0 = Null     action 1 = Inactive Lever Press    action 2 = Active Lever Press
initialExState  = 0

transition = numpy.zeros( [statesNum , actionsNum, statesNum] , float)
setTransition(0,0,0,1)          # From state s, and by taking a, we go to state s', with probability p
setTransition(0,1,1,1)
setTransition(1,0,2,1)
setTransition(1,1,2,1)
setTransition(2,0,3,1)
setTransition(2,1,3,1)
setTransition(3,0,4,1)
setTransition(3,1,4,1)
setTransition(4,0,0,1)
setTransition(4,1,0,1)

outcome = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )
setOutcome(0,1,1,cocaine)       # At state s, by doing action a and going to state s', we receive the outcome 

nonHomeostaticReward = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )
setNonHomeostaticReward(0,1,1,-leverPressCost)
setNonHomeostaticReward(1,1,2,-leverPressCost)
setNonHomeostaticReward(2,1,3,-leverPressCost)
setNonHomeostaticReward(3,1,4,-leverPressCost)
setNonHomeostaticReward(4,1,0,-leverPressCost)

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Definition of the Animal   --------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''

#------------ Homeostatic System
initialInState          = 0
initialSetpoint         = 200
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
beta                    = 0.01  # Rate of exploration
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
seekingHoursLgA     = 2            
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


dosesNum = 25
infusionPerDoseHigh = numpy.zeros([dosesNum],float)
infusionPerDoseLow  = numpy.zeros([dosesNum],float)

for i in range(0,dosesNum):
    
    print "Dose num: %d / %d     " %(i+1,dosesNum)
    
    initialSetpoint         = 125

    cocaine         = 11.5 + (1.3)**i             # Dose of self-administered drug
    setOutcome(0,1,1,cocaine)              # At state s, by doing action a and going to state s', we receive the outcome 
    initializeAnimal        (   )
    resetLogInfoLgA       (                         )

    cocaineSeeking        (  0 , 'LgA'        )
    homeCage              (  0 , 'LgA'        ) 

    loggingFinalization()

    infRate = 0
    for j in range(0,seekingTrialsNumLgA):
        infRate = infRate + infusionLgA[j]        
    infusionPerDoseHigh[i] = infRate/seekingHoursLgA
    

    
    
    
    
    initialSetpoint         = 200

    cocaine         = 11.5 + (1.3)**i             # Dose of self-administered drug
    setOutcome(0,1,1,cocaine)              # At state s, by doing action a and going to state s', we receive the outcome 
    initializeAnimal        (   )
    resetLogInfoLgA       (                         )

    cocaineSeeking        (  0 , 'LgA'        )
    homeCage              (  0 , 'LgA'        ) 

    loggingFinalization()

    infRate = 0
    for j in range(0,seekingTrialsNumLgA):
        infRate = infRate + infusionLgA[j]        
    infusionPerDoseLow[i] = infRate/seekingHoursLgA
    


'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plotting 1  ------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''    
 
font = {'family' : 'normal', 'size'   : 16}
pylab.rc('font', **font)
pylab.rcParams.update({'legend.fontsize': 16})

fig1 = pylab.figure( figsize=(5,3.5) )
fig1.subplots_adjust(left=0.16)
fig1.subplots_adjust(bottom=0.2)
ax1 = fig1.add_subplot(111)

infusionPerDoseHigh [10] = 0
infusionPerDoseLow [8] = 0
infusionPerDoseLow [9] = 0

S1 = ax1.plot(infusionPerDoseLow  , 'o', ms=8, markeredgewidth =1.5, alpha=1, mfc='white',linewidth = 2 , color='red' )
S0 = ax1.plot(infusionPerDoseHigh , 'o', ms=8, markeredgewidth =1.5, alpha=1, mfc='black',linewidth = 2 , color='red' )


def fitFuncGuassian(x, a0, a1, a2):
    y = a0 * numpy.exp( -a1* ( (a2-x)**2 ) )
    return y
 
x = numpy.arange(0, dosesNum, 1)
parameters, covariance = curve_fit(fitFuncGuassian, x, infusionPerDoseHigh)
fitted = fitFuncGuassian(x,parameters[0],parameters[1],parameters[2])
#S2 = ax1.plot(fitted , '-', ms=8, markeredgewidth =2, alpha=1, mfc='black',linewidth = 2 , color='black' )

parameters, covariance = curve_fit(fitFuncGuassian, x, infusionPerDoseLow)
fitted = fitFuncGuassian(x,parameters[0],parameters[1],parameters[2])
#S3 = ax1.plot(fitted , '-', ms=8, markeredgewidth =2, alpha=1, mfc='black',linewidth = 2 , color='black' )

leg=fig1.legend((S1, S0), ('D2R-Low','D2R-High'), loc = (0.51,0.68))
leg.draw_frame(False)

pylab.yticks(pylab.arange(0, 51, 10))
pylab.ylim((-5,55))

pylab.xlim((-1,26))
tick_lcs = [0,5,10,15,20,25]
tick_lbs = numpy.zeros( [6] , float)
for i in range(0,6): 
    tick_lbs[i] = '%.2f' %( ( 11.5 + 1.3**(i*5) ) * 0.005 )
pylab.xticks(tick_lcs, tick_lbs)


for line in ax1.get_xticklines() + ax1.get_yticklines():
    line.set_markeredgewidth(2)
    line.set_markersize(5)

ax1.set_ylabel('Responses / h')
ax1.set_xlabel('Cocaine dose (mg/infusion)')
#ax1.set_title('Post-escelation')
fig1.savefig('DoseResponseCurve.eps', format='eps')

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plotting 2  ------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''    

consumptionDoseHigh = numpy.zeros([dosesNum],float)
consumptionDoseLow  = numpy.zeros([dosesNum],float)

for j in range(0 , dosesNum):
    cocaine                 = ( 11.5 + (1.3)**j ) * 0.005            # Dose of self-administered drug
    consumptionDoseHigh [j] =  infusionPerDoseHigh[j] * cocaine
    consumptionDoseLow  [j] =  infusionPerDoseLow [j] * cocaine
 

fig1 = pylab.figure( figsize=(5,3.5) )
fig1.subplots_adjust(left=0.16)
fig1.subplots_adjust(bottom=0.2)
ax1 = fig1.add_subplot(111)

consumptionDoseHigh [10] = 0
consumptionDoseLow [8] = 0
consumptionDoseLow [9] = 0

S1 = ax1.plot(consumptionDoseLow  , 'o', ms=8, markeredgewidth =1.5, alpha=1, mfc='white',linewidth = 2 , color='red' )
S0 = ax1.plot(consumptionDoseHigh , 'o', ms=8, markeredgewidth =1.5, alpha=1, mfc='black',linewidth = 2 , color='red' )


def fitFuncSigmoid(x, a0, a1, a2):
    y = a0 / ( 1 +  numpy.exp( -a1*(x-a2)) )
    return y
 
x = numpy.arange(0, dosesNum, 1)
parameters, covariance = curve_fit(fitFuncSigmoid, x[0:18], consumptionDoseHigh[0:18])
fitted = fitFuncSigmoid(x[0:18],parameters[0],parameters[1],parameters[2])
#S2 = ax1.plot(x[0:18],fitted , '-', ms=8, markeredgewidth =2, alpha=1, mfc='black',linewidth = 2 , color='black' )

parameters, covariance = curve_fit(fitFuncSigmoid, x[0:18], consumptionDoseLow[0:18])
fitted = fitFuncSigmoid(x[0:18],parameters[0],parameters[1],parameters[2])
#S3 = ax1.plot(x[0:18],fitted , '-', ms=8, markeredgewidth =2, alpha=1, mfc='black',linewidth = 2 , color='black' )


leg=fig1.legend((S1, S0), ('D2R-Low','D2R-High'), loc = (0.16,0.68))
leg.draw_frame(False)

pylab.yticks(pylab.arange(0, 10, 2))
pylab.ylim((-1,10))


pylab.xlim((-1,26))
tick_lcs = [0,5,10,15,20,25]
tick_lbs = numpy.zeros( [6] , float)
for i in range(0,6): 
    tick_lbs[i] = '%.2f' %( ( 11.5 + 1.3**(i*5) ) * 0.005 )
pylab.xticks(tick_lcs, tick_lbs)


for line in ax1.get_xticklines() + ax1.get_yticklines():
    line.set_markeredgewidth(2)
    line.set_markersize(5)

ax1.set_ylabel('Total intake (mg/h)')
ax1.set_xlabel('Cocaine dose (mg/infusion)')
#ax1.set_title('Post-escelation')
fig1.savefig('DoseConsumptionCurve.eps', format='eps')



pylab.show()
