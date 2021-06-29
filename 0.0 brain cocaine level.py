'''

    Modification of parameters:
        - The priming dose is twice the dose used in the experiments. This is necessary to replicate the results.
        - The learning rate for updating the outcome and transition functions is used for the underNoCoca case.
          For the underCoca case, a much slower leaning rate is used. This additional parameter is necessary in order
          to replicate the "slow" rate of extinction of the priming-induced reinstatement.
        - leverPressCost  = 20   --> in order to avoid exploratory lever-pressing

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
###################################################################################################################################
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
                estimatedTransitionUnderCoca            [i][j][k] = 0.0
                estimatedOutcomeUnderCoca               [i][j][k] = 0.0
                estimatedNonHomeostaticRewardUnderCoca  [i][j][k] = 0.0
                estimatedTransitionNoCoca               [i][j][k] = 0.0
                estimatedOutcomeNoCoca                  [i][j][k] = 0.0
                estimatedNonHomeostaticRewardNoCoca     [i][j][k] = 0.0


    estimatedTransitionUnderCoca [0][0][0] = 1
    estimatedTransitionUnderCoca [0][1][1] = 1
    estimatedTransitionUnderCoca [1][0][2] = 1
    estimatedTransitionUnderCoca [1][1][2] = 1
    estimatedTransitionUnderCoca [2][0][3] = 1
    estimatedTransitionUnderCoca [2][1][3] = 1
    estimatedTransitionUnderCoca [3][0][4] = 1
    estimatedTransitionUnderCoca [3][1][4] = 1
    estimatedTransitionUnderCoca [4][0][0] = 1
    estimatedTransitionUnderCoca [4][1][0] = 1

    estimatedTransitionNoCoca    [0][0][0] = 1
    estimatedTransitionNoCoca    [0][1][1] = 1
    estimatedTransitionNoCoca    [1][0][2] = 1
    estimatedTransitionNoCoca    [1][1][2] = 1
    estimatedTransitionNoCoca    [2][0][3] = 1
    estimatedTransitionNoCoca    [2][1][3] = 1
    estimatedTransitionNoCoca    [3][0][4] = 1
    estimatedTransitionNoCoca    [3][1][4] = 1
    estimatedTransitionNoCoca    [4][0][0] = 1
    estimatedTransitionNoCoca    [4][1][0] = 1
    
#    Assuming that the animals know the energy cost (fatigue) of pressing a lever 
    for i in range(0,statesNum):
        for j in range(0,statesNum):
            estimatedNonHomeostaticRewardUnderCoca      [i][1][j] = -leverPressCost
            estimatedNonHomeostaticRewardNoCoca         [i][1][j] = -leverPressCost

    estimatedOutcomeUnderCoca [0][1][1]     = cocaine
    estimatedOutcomeNoCoca    [0][1][1]     = cocaine
    
    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   To what extend the animal is under the effect of cocaine   ------------------------------------ 
--------------------------------------------------------------------------------------------------------------------------------'''
def underCocaine(inS,setS):
        
    underCocaRate = (inS - inStateLowerBound) / ( setS - inStateLowerBound )
    if underCocaRate>1: 
        underCocaRate = 1
        
    return underCocaRate


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
---------------------------------   Goal-directed Value estimation, Assuming the animal is under Cocaine  ------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def valueEstimationUnderCoca(state,inState,setpointS,depthLeft):

    values = numpy.zeros ( [actionsNum] , float )

    # If this is the last depth that should be searched :
    if depthLeft==1:
        for action in range(0,actionsNum):
            for nextState in range(0,statesNum):
                homeoReward    = driveReductionReward(inState,setpointS,cocaine)*estimatedOutcomeUnderCoca[state][action][nextState]/cocaine
                nonHomeoReward = estimatedNonHomeostaticRewardUnderCoca[state][action][nextState]
                transitionProb = estimatedTransitionUnderCoca[state][action][nextState]
                values[action] = values[action] +  transitionProb * ( homeoReward + nonHomeoReward )
        return values
    
    # Otherwise :
    for action in range(0,actionsNum):
        for nextState in range(0,statesNum):
            if estimatedTransitionUnderCoca[state][action][nextState] < pruningThreshold :
                VNextStateBest = 0
            else:    
                VNextState = valueEstimationUnderCoca(nextState,setpointS,inState,depthLeft-1)
                VNextStateBest = maxValue (VNextState)
            homeoReward    = driveReductionReward(inState,setpointS,cocaine)*estimatedOutcomeUnderCoca[state][action][nextState]/cocaine
            nonHomeoReward = estimatedNonHomeostaticRewardUnderCoca[state][action][nextState]
            transitionProb = estimatedTransitionUnderCoca[state][action][nextState]
            values[action] = values[action] + transitionProb * ( homeoReward + nonHomeoReward + gamma*VNextStateBest ) 
            
    return values

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Goal-directed Value estimation, Assuming the animal is not under Cocaine  --------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def valueEstimationNoCoca(state,inState,setpointS,depthLeft):

    values = numpy.zeros ( [actionsNum] , float )

    # If this is the last depth that should be searched :
    if depthLeft==1:
        for action in range(0,actionsNum):
            for nextState in range(0,statesNum):
                homeoReward    = driveReductionReward(inState,setpointS,cocaine)*estimatedOutcomeNoCoca[state][action][nextState]/cocaine
                nonHomeoReward = estimatedNonHomeostaticRewardNoCoca[state][action][nextState]
                transitionProb = estimatedTransitionNoCoca[state][action][nextState]
                values[action] = values[action] +  transitionProb * ( homeoReward + nonHomeoReward )
        return values
    
    # Otherwise :
    for action in range(0,actionsNum):
        for nextState in range(0,statesNum):
            if estimatedTransitionNoCoca[state][action][nextState] < pruningThreshold :
                VNextStateBest = 0
            else:    
                VNextState = valueEstimationNoCoca(nextState,setpointS,inState,depthLeft-1)
                VNextStateBest = maxValue (VNextState)
            homeoReward    = driveReductionReward(inState,setpointS,cocaine)*estimatedOutcomeNoCoca[state][action][nextState]/cocaine
            nonHomeoReward = estimatedNonHomeostaticRewardNoCoca[state][action][nextState]
            transitionProb = estimatedTransitionNoCoca[state][action][nextState]
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
def updateOutcomeFunction(state,action,nextState,out,underCocaWeight):

    learningRateUnderCoca = updateOutcomeRate * underCocaWeight * cocaineInducedLearningRateDeficiency
    learningRateNoCoca    = updateOutcomeRate * (1 - underCocaWeight)

    estimatedOutcomeUnderCoca[state][action][nextState] = (1.0-learningRateUnderCoca)*estimatedOutcomeUnderCoca[state][action][nextState] +     learningRateUnderCoca*out
    estimatedOutcomeNoCoca   [state][action][nextState] = (1.0-learningRateNoCoca   )*estimatedOutcomeNoCoca   [state][action][nextState] +     learningRateNoCoca   *out
    
    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Update the expected-non-homeostatic-reward function  ------------------------------------------ 
--------------------------------------------------------------------------------------------------------------------------------'''
def updateNonHomeostaticRewardFunction(state,action,nextState,rew,underCocaWeight):
    
    learningRateUnderCoca = updateOutcomeRate * underCocaWeight
    learningRateNoCoca    = updateOutcomeRate * (1 - underCocaWeight)
    
    estimatedNonHomeostaticRewardUnderCoca[state][action][nextState] = (1.0-learningRateUnderCoca)*estimatedNonHomeostaticRewardUnderCoca[state][action][nextState] +     learningRateUnderCoca*rew
    estimatedNonHomeostaticRewardNoCoca   [state][action][nextState] = (1.0-learningRateNoCoca   )*estimatedNonHomeostaticRewardNoCoca   [state][action][nextState] +     learningRateNoCoca   *rew
    
    return


'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Update the expected-transition function  ------------------------------------------------------ 
--------------------------------------------------------------------------------------------------------------------------------'''
def updateTransitionFunction(state,action,nextState,underCocaWeight):

    learningRateUnderCoca = updateOutcomeRate * underCocaWeight
    learningRateNoCoca    = updateOutcomeRate * (1 - underCocaWeight)
  
    #---- First inhibit all associations
    for i in range(0,statesNum):
        estimatedTransitionUnderCoca[state][action][i] = (1.0-learningRateUnderCoca)*estimatedTransitionUnderCoca[state][action][i]
        estimatedTransitionNoCoca   [state][action][i] = (1.0-learningRateNoCoca   )*estimatedTransitionNoCoca   [state][action][i]
    
    #---- Then potentiate the experiences association
    estimatedTransitionUnderCoca[state][action][nextState] = estimatedTransitionUnderCoca[state][action][nextState] + learningRateUnderCoca
    estimatedTransitionNoCoca   [state][action][nextState] = estimatedTransitionNoCoca   [state][action][nextState] + learningRateNoCoca
                
    return
'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Pre-training Sessions  ------------------------------------------------------------------------ 
--------------------------------------------------------------------------------------------------------------------------------'''
def pretraining  (ratType):

    exState     = state[0]
    inState     = state[1]
    setpointS   = state[2]
    trialCount  = state[3]
    cocBuffer   = 0
    
    trialsNum = pretrainingTrialsNum
    
    for trial in range(0,trialsNum):

        estimatedActionValuesUnderCoca   = valueEstimationUnderCoca ( exState, inState, setpointS, searchDepth )
        estimatedActionValuesNoCoca      = valueEstimationNoCoca    ( exState, inState, setpointS, searchDepth )        
        underCocaineWeight               = underCocaine             ( inState , setpointS                      )
        estimatedActionValues            = estimatedActionValuesUnderCoca*underCocaineWeight + estimatedActionValuesNoCoca*(1-underCocaineWeight)         
        
        action                          = actionSelectionSoftmax    ( exState , estimatedActionValues           )
        nextState                       = getRealizedTransition     ( exState , action                          )
        out                             = getOutcome                ( exState , action    , nextState           )
        nonHomeoRew                     = getNonHomeostaticReward   ( exState , action    , nextState           )
        HomeoRew                        = driveReductionReward      ( inState , setpointS , out                 )

        if ratType=='ShA':  
            loggingShA (trial,action,inState,setpointS,out)    
            print "ShA rat number: %d / %d     Pre-training session     trial: %d / %d      animal seeking cocaine  %d   %d" %(animal+1,animalsNum,trial+1,trialsNum, estimatedOutcomeUnderCoca[0][1][1],estimatedOutcomeNoCoca[0][1][1])
        elif ratType=='LgA':  
            loggingLgA (trial,action,inState,setpointS,out)    
            print "LgA rat number: %d / %d     Pre-training session     trial: %d / %d      animal seeking cocaine" %(animal+1,animalsNum,trial+1,trialsNum)

        updateOutcomeFunction               ( exState , action , nextState , out ,underCocaineWeight            )
        updateNonHomeostaticRewardFunction  ( exState , action , nextState , nonHomeoRew ,underCocaineWeight    )
        updateTransitionFunction            ( exState , action , nextState , underCocaineWeight                 )            
        
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
    
    for trial in range(0,trialsPerDay):

#        estimatedActionValuesUnderCoca   = valueEstimationUnderCoca ( exState, inState, setpointS, searchDepth )
#        estimatedActionValuesNoCoca      = valueEstimationNoCoca    ( exState, inState, setpointS, searchDepth )        
#        underCocaineWeight               = underCocaine             ( inState , setpointS                      )
#        estimatedActionValues            = estimatedActionValuesUnderCoca*underCocaineWeight + estimatedActionValuesNoCoca*(1-underCocaineWeight)         

        action = 0
        nextState =0
        nextState = 0
        HomeoRew = 0
        if trial==0:
            out = cocaine
        else:
            out = 0
        
        if ratType=='LgA':  
            loggingLgA(trial,action,inState,setpointS,out)    
            print "LgA rat number: %d / %d     Session Number: %d / %d     trial: %d / %d      animal seeking cocaine" %(animal+1,animalsNum,sessionNum+1,sessionsNum,trial-trialCount+1,trialsNum)

#        updateOutcomeFunction               ( exState , action , nextState , out ,underCocaineWeight            )
#        updateNonHomeostaticRewardFunction  ( exState , action , nextState , nonHomeoRew ,underCocaineWeight    )
#        updateTransitionFunction            ( exState , action , nextState , underCocaineWeight                 )            
        
        cocBuffer = cocBuffer + out                
        
        inState     = updateInState(inState,cocBuffer*cocAbsorptionRatio)
#        setpointS   = updateSetpoint(setpointS,out)

        cocBuffer = cocBuffer*(1-cocAbsorptionRatio)

#        exState   = nextState

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
        activeLeverPressShA[trial]     = activeLeverPressShA[trial] + 1
    internalStateShA[trial]    = internalStateShA[trial] + inState
    setpointShA[trial]         = setpointShA[trial] + setpointS    
    if coca==cocaine:
        infusionShA[trial]     = infusionShA[trial] + 1

    estimatedOutcomeUnderCocaShA [trial] = estimatedOutcomeUnderCocaShA [trial] + estimatedOutcomeUnderCoca[0][1][1]
    estimatedOutcomeNoCocaShA    [trial] = estimatedOutcomeNoCocaShA    [trial] + estimatedOutcomeNoCoca   [0][1][1]

    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Logging the current information for the Long-access group  ------------------------------------ 
--------------------------------------------------------------------------------------------------------------------------------'''
def loggingLgA(trial,action,inState,setpointS,coca):
   
    internalStateLgA[trial]    = internalStateLgA[trial] + inState
    setpointLgA[trial]         = setpointLgA[trial] + setpointS    
    if coca==cocaine:
        infusionLgA[trial]     = infusionLgA[trial] + 1

    estimatedOutcomeUnderCocaLgA [trial] = estimatedOutcomeUnderCocaLgA [trial] + estimatedOutcomeUnderCoca[0][1][1]
    estimatedOutcomeNoCocaLgA    [trial] = estimatedOutcomeNoCocaLgA    [trial] + estimatedOutcomeNoCoca   [0][1][1]

    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Wrap up all the logged data   ----------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def loggingFinalization():
    
    for trial in range(0,totalTrialsNum):
        nulDoingShA[trial]             = nulDoingShA[trial]/animalsNum
        activeLeverPressShA[trial]     = activeLeverPressShA[trial]/animalsNum
        internalStateShA[trial]        = internalStateShA[trial]/animalsNum
        setpointShA[trial]             = setpointShA[trial]/animalsNum  
        infusionShA[trial]             = infusionShA[trial]/animalsNum 
        estimatedOutcomeUnderCocaShA [trial] = estimatedOutcomeUnderCocaShA [trial]/animalsNum
        estimatedOutcomeNoCocaShA    [trial] = estimatedOutcomeNoCocaShA    [trial]/animalsNum

        nulDoingLgA[trial]             = nulDoingLgA[trial]/animalsNum
        activeLeverPressLgA[trial]     = activeLeverPressLgA[trial]/animalsNum
        internalStateLgA[trial]        = internalStateLgA[trial]/animalsNum
        setpointLgA[trial]             = setpointLgA[trial]/animalsNum  
        infusionLgA[trial]             = infusionLgA[trial]/animalsNum 
        estimatedOutcomeUnderCocaLgA [trial] = estimatedOutcomeUnderCocaLgA [trial]/animalsNum
        estimatedOutcomeNoCocaLgA    [trial] = estimatedOutcomeNoCocaLgA    [trial]/animalsNum


    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plot the internal state  ---------------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotInternalState45():

    font = {'family' : 'normal', 'size'   : 16}
    pylab.rc('font', **font)
    pylab.rcParams.update({'legend.fontsize': 16})
        
    fig1 = pylab.figure( figsize=(5,3.5) )
    fig1.subplots_adjust(left=0.16)
    fig1.subplots_adjust(bottom=0.2)



    ax1 = fig1.add_subplot(111)
    S0 = ax1.plot(internalStateLgA[0:675] , linewidth = 2 , color='black' )

    ax1.axhline(0, color='0.25',ls='--', lw=1 )
  
    pylab.yticks(pylab.arange(0, 41, 10))
    pylab.ylim((-5,47))
    pylab.xlim( ( -45 , 675 + 45 ) )
    
    tick_lcs = []
    tick_lbs = []
    for i in range ( 0 , 5 ):
        tick_lcs.append( i*15*10 ) 
        tick_lbs.append(i*10)
    pylab.xticks(tick_lcs, tick_lbs)

#    for i in range ( 0 , 5 ):
#        if i%2==0:
#            p = pylab.axvspan( i*extinctionTrialsNum + i, (i+1)*extinctionTrialsNum + i , facecolor='0.75',edgecolor='none', alpha=0.5)        
    
    for line in ax1.get_xticklines() + ax1.get_yticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)

    ax1.set_ylabel('Internal State')
    ax1.set_xlabel('Time (min)')
    ax1.set_title('')
    fig1.savefig('internalState45min.eps', format='eps')

    return
'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plot the internal state  ---------------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotInternalState5():

    font = {'family' : 'normal', 'size'   : 16}
    pylab.rc('font', **font)
        
    fig1 = pylab.figure( figsize=(5,3.5) )
    fig1.subplots_adjust(left=0.16)
    fig1.subplots_adjust(bottom=0.2)


    ax1 = fig1.add_subplot(111)
    S0 = ax1.plot(internalStateLgA[0:60] , linewidth = 2 , color='black' )

    ax1.axhline(0, color='0.25',ls='--', lw=1 )
  
    pylab.yticks(pylab.arange(0, 41, 10))
    pylab.ylim((-5,47))
    pylab.xlim( ( -4 , 60 + 4 ) )
    
    tick_lcs = []
    tick_lbs = []
    for i in range ( 0 , 5 ):
        tick_lcs.append( i*15 ) 
        tick_lbs.append((i*240)/4)
    pylab.xticks(tick_lcs, tick_lbs)

#    for i in range ( 0 , 5 ):
#        if i%2==0:
#            p = pylab.axvspan( i*extinctionTrialsNum + i, (i+1)*extinctionTrialsNum + i , facecolor='0.75',edgecolor='none', alpha=0.5)        
    
    for line in ax1.get_xticklines() + ax1.get_yticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)

    ax1.set_ylabel('Internal State')
    ax1.set_xlabel('Time (sec)')
    ax1.set_title('')
    fig1.savefig('internalState4min.eps', format='eps')

    return
'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plot all the results  ------------------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotting():

    loggingFinalization()

    plotInternalState45() 
    plotInternalState5() 


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

cocaine         = 50             # Dose of self-administered drug
leverPressCost  = 20             # Energy cost for pressing the lever

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
updateOutcomeRate       = 0.025 # Learning rate for updating the outcome function
cocaineInducedLearningRateDeficiency = 0.15
updateTransitionRate    = 0.2   # Learning rate for updating the transition function
updateRewardRate        = 0.2   # Learning rate for updating the non-homeostatic reward function
gamma                   = 1     # Discount factor
beta                    = 0.25  # Rate of exploration
searchDepth             = 3     # Depth of going into the decision tree for goal-directed valuation of choices
pruningThreshold        = 0.1   # If the probability of a transition like (s,a,s') is less than "pruningThreshold", cut it from the decision tree 

estimatedTransitionUnderCoca             = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )
estimatedOutcomeUnderCoca                = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )
estimatedNonHomeostaticRewardUnderCoca   = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )
estimatedTransitionNoCoca                = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )
estimatedOutcomeNoCoca                   = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )
estimatedNonHomeostaticRewardNoCoca      = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )

state                            = numpy.zeros ( [4] , float )     # a vector of the external state, internal state, setpoint, and trial

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Simulation Parameters   -----------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''

animalsNum          = 1                                  # Number of animals

pretrainingHours    = 0
sessionsNum         = 1                            # Number of sessions of cocain seeking, followed by rest in home-cage
seekingHoursShA     = 1            
seekingHoursLgA     = 24            
extinctionHours     = 0.75

trialsPerHour       = 60*60/4                            # Number of trials during one hour (as each trial is supposed to be 4 seconds)
trialsPerDay        = 24*trialsPerHour
pretrainingTrialsNum= pretrainingHours* trialsPerHour
restAfterPretrainingTrialsNum = (24 - pretrainingHours) *trialsPerHour

seekingTrialsNumShA = seekingHoursShA * trialsPerHour    # Number of trials for each cocaine seeking session
restingHoursShA     = 24 - seekingHoursShA
restTrialsNumShA    = restingHoursShA * trialsPerHour    # Number of trials for each session of the animal being in the home cage

seekingTrialsNumLgA = seekingHoursLgA * trialsPerHour    # Number of trials for each cocaine seeking session
restingHoursLgA     = 24 - seekingHoursLgA
restTrialsNumLgA    = restingHoursLgA * trialsPerHour    # Number of trials for each session of the animal being in the home cage

extinctionTrialsNum = extinctionHours*trialsPerHour      # Number of trials for each extinction session

totalTrialsNum      = trialsPerDay

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plotting Parameters   -------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''

trialsPerBlock = 10*60/4            # Each BLOCK is 10 minutes - Each minute 60 second - Each trial takes 4 seconds

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Logging Parameters   --------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''

nulDoingShA            = numpy.zeros( [totalTrialsNum] , float)
activeLeverPressShA    = numpy.zeros( [totalTrialsNum] , float)
internalStateShA       = numpy.zeros( [totalTrialsNum] , float)
setpointShA            = numpy.zeros( [totalTrialsNum] , float)
infusionShA            = numpy.zeros( [totalTrialsNum] , float)
estimatedOutcomeUnderCocaShA  = numpy.zeros( [totalTrialsNum] , float)
estimatedOutcomeNoCocaShA     = numpy.zeros( [totalTrialsNum] , float)



nulDoingLgA            = numpy.zeros( [totalTrialsNum] , float)
activeLeverPressLgA    = numpy.zeros( [totalTrialsNum] , float)
internalStateLgA       = numpy.zeros( [totalTrialsNum] , float)
setpointLgA            = numpy.zeros( [totalTrialsNum] , float)
infusionLgA            = numpy.zeros( [totalTrialsNum] , float)
estimatedOutcomeUnderCocaLgA  = numpy.zeros( [totalTrialsNum] , float)
estimatedOutcomeNoCocaLgA     = numpy.zeros( [totalTrialsNum] , float)

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Simulation   ----------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''

for animal in range(0,animalsNum):

    initialSetpoint     = optimalInStateUpperBound
    initializeAnimal      (                               )

    cocaineSeeking  (0 ,'LgA')

plotting()
