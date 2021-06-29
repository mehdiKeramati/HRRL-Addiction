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
    
    for trial in range(trialCount,trialCount+trialsNum):

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
            loggingShA(trial,action,inState,setpointS,out)    
            print "ShA rat number: %d / %d     Session Number: %d / %d     trial: %d / %d      animal seeking cocaine" %(animal+1,animalsNum,sessionNum+1,sessionsNum,trial-trialCount+1,trialsNum)
        if ratType=='LgA':  
            loggingLgA(trial,action,inState,setpointS,out)    
            print "LgA rat number: %d / %d     Session Number: %d / %d     trial: %d / %d      animal seeking cocaine" %(animal+1,animalsNum,sessionNum+1,sessionsNum,trial-trialCount+1,trialsNum)

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
---------------------------------   Extinction Sessions  -------------------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def extinction  ( sessionNum,trialsNum , ratsType):

    exState     = state[0]
    inState     = state[1]
    setpointS   = state[2]
    trialCount  = state[3]
    cocBuffer   = 0
    
    for trial in range(trialCount,trialCount+trialsNum):
        
        estimatedActionValuesUnderCoca   = valueEstimationUnderCoca ( exState, inState, setpointS, searchDepth )
        estimatedActionValuesNoCoca      = valueEstimationNoCoca    ( exState, inState, setpointS, searchDepth )        
        underCocaineWeight               = underCocaine (inState,setpointS)
        estimatedActionValues            = estimatedActionValuesUnderCoca*underCocaineWeight + estimatedActionValuesNoCoca*(1-underCocaineWeight)         
        
        action                  = actionSelectionSoftmax        ( exState, estimatedActionValues    )
        nextState               = getRealizedTransition         ( exState, action                   )
        out                     = 0
        nonHomeoRew             = getNonHomeostaticReward       ( exState, action, nextState        )
        HomeoRew                = driveReductionReward          ( inState, setpointS, out           )

        
        if ratsType == 'ShA':
            loggingShA(trial,action,inState,setpointS,out)   
            print "ShA rat number: %d / %d     Extinction session %d / %d       trial: %d / %d      Extinction of cocaine seeking" %(animal+1,animalsNum,sessionNum,sessionsNum,trial-trialCount+1,trialsNum)
        if ratsType == 'LgA':
            loggingLgA(trial,action,inState,setpointS,out)    
            print "LgA rat number: %d / %d     Extinction session %d / %d       trial: %d / %d      Extinction of cocaine seeking" %(animal+1,animalsNum,sessionNum,sessionsNum*4 +1,trial-trialCount+1,trialsNum)

        updateOutcomeFunction               ( exState, action, nextState, out ,underCocaineWeight           )
        updateNonHomeostaticRewardFunction  ( exState, action, nextState, nonHomeoRew ,underCocaineWeight   )
        updateTransitionFunction            ( exState, action, nextState, underCocaineWeight                )            
        
        cocBuffer = cocBuffer + out                
        
        inState     = updateInState         ( inState, cocBuffer*cocAbsorptionRatio )
        setpointS   = updateSetpoint        ( setpointS, out )

        cocBuffer = cocBuffer * ( 1-cocAbsorptionRatio )

        exState   = nextState

    state[0]    = exState
    state[1]    = inState
    state[2]    = setpointS
    state[3]    = trialCount+trialsNum

    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Non-contingent cocaine infusion   ------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def noncontingentInfusion ( sessionNum, cocDose,ratType):

    exState     = state[0]
    inState     = state[1]
    setpointS   = state[2]
    trialCount  = state[3]

    inState     = updateInState(inState,cocDose)
    setpointS   = updateSetpoint(setpointS,cocDose)
    if ratType == 'ShA':
        loggingShA(trialCount,0,inState,setpointS,cocDose)    
        print "ShA rat number: %d / %d     Session Number: %d / %d                         animal receives non-contingent cocaine infusion" %(animal+1,animalsNum,sessionNum+1,sessionsNum)
    if ratType == 'LgA':
        loggingLgA(trialCount,0,inState,setpointS,cocDose)    
        print "LgA rat number: %d / %d     Session Number: %d / %d                         animal receives non-contingent cocaine infusion" %(animal+1,animalsNum,sessionNum+1,sessionsNum)
        

    state[0]    = exState
    state[1]    = inState
    state[2]    = setpointS
    state[3]    = trialCount + 1

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
   
    if action==0: 
        nulDoingLgA[trial]             = nulDoingLgA[trial] + 1
    elif action==1: 
        activeLeverPressLgA[trial]     = activeLeverPressLgA[trial] + 1
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
def plotInternalState():
 
    font = {'family' : 'normal', 'size'   : 16}
    pylab.rc('font', **font)
    pylab.rcParams.update({'legend.fontsize': 16})
        
    fig1 = pylab.figure( figsize=(8,3.5) )
    fig1.subplots_adjust(left=0.16)
    fig1.subplots_adjust(bottom=0.2)

    ax1 = fig1.add_subplot(111)
    S0 = ax1.plot(internalStateLgA [0 : extinctionTrialsNum*5 + 4] , linewidth = 2 , color='black' )

    ax1.axhline(200, color='0.25',ls='--', lw=1 )
  
    pylab.yticks(pylab.arange(0, 410, 100))
    pylab.ylim((-40,450))
    pylab.xlim( ( -150 , extinctionTrialsNum*5 + 4 + 150 ) )
    
    tick_lcs = []
    tick_lbs = []
    for i in range ( 0 , 6 ):
        tick_lcs.append( i*15*15*3 ) 
        tick_lbs.append(i*15*3)
    pylab.xticks(tick_lcs, tick_lbs)

    for i in range ( 0 , 5 ):
        if i%2==0:
            p = pylab.axvspan( i*extinctionTrialsNum + i, (i+1)*extinctionTrialsNum + i , facecolor='0.75',edgecolor='none', alpha=0.5)        
    
    for line in ax1.get_xticklines() + ax1.get_yticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)

    ax1.set_ylabel('Internal State')
    ax1.set_xlabel('Time (min)')
    ax1.set_title('')
    fig1.savefig('internalState.eps', format='eps')

    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   plot Etimated Cocaine Probability for ShA rats ------------------------------------------------ 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotEtimatedCocaineProbabilityFirstLgA():
        
    fig1 = pylab.figure( figsize=(8,3.5) )
    fig1.subplots_adjust(left=0.16)
    fig1.subplots_adjust(bottom=0.2)

    ax1 = fig1.add_subplot(111)
    S0 = ax1.plot(estimatedOutcomeUnderCocaLgA [ 0 : extinctionTrialsNum*5 + 4 ]/50 , linewidth = 2.5 , color='black' )
    S1 = ax1.plot(estimatedOutcomeNoCocaLgA    [ 0 : extinctionTrialsNum*5 + 4 ]/50 , linewidth = 1.5 , color='black' )


    ax1.axhline(1, color='0.25',ls='--', lw=1 )
    ax1.axhline(0, color='0.25',ls='--', lw=1 )
  
    pylab.yticks(pylab.arange(0, 1.1, 0.2))
    pylab.ylim((-0.2,1.2))
    pylab.xlim( ( -150 , extinctionTrialsNum*5 + 4 + 150) )

    tick_lcs = []
    tick_lbs = []
    for i in range ( 0 , 6 ):
        tick_lcs.append( i*15*15*3 ) 
        tick_lbs.append(i*15*3)
    pylab.xticks(tick_lcs, tick_lbs)
      
    leg = fig1.legend((S0, S1), ('Cocaine state','NoCocaine state'), loc = (0.56,0.42))
    leg.draw_frame(False)      
    
    for i in range ( 0 , 5 ):
        if i%2==0:
            p = pylab.axvspan( i*extinctionTrialsNum + i, (i+1)*extinctionTrialsNum + i , facecolor='0.75',edgecolor='none', alpha=0.5)        

    for line in ax1.get_xticklines() + ax1.get_yticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)


    ax1.set_ylabel('Subjective probability')
    ax1.set_xlabel('Time (min)')
    ax1.set_title('')
    fig1.savefig('etimatedCocaineProbabilityFirstLgA.eps', format='eps')

    return    


'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   plot Etimated Cocaine Probability for ShA rats ------------------------------------------------ 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotEtimatedCocaineProbabilityLgA():
     
    fig1 = pylab.figure( figsize=(5,3.5) )
    fig1.subplots_adjust(left=0.16)
    fig1.subplots_adjust(bottom=0.2)


    ax1 = fig1.add_subplot(111)
    S0 = ax1.plot(estimatedOutcomeUnderCocaLgA [ 0 : extinctionTrialsNum + (extinctionTrialsNum*4 + 4)*10 ]/50 , linewidth = 2.5 , color='black' )
    S1 = ax1.plot(estimatedOutcomeNoCocaLgA    [ 0 : extinctionTrialsNum + (extinctionTrialsNum*4 + 4)*10 ]/50 , linewidth = 1.5 , color='black' )


    ax1.axhline(1, color='0.25',ls='--', lw=1 )
    ax1.axhline(0, color='0.25',ls='--', lw=1 )
  
    pylab.yticks(pylab.arange(0, 1.1, 0.2))
    pylab.ylim((-0.2,1.2))
    pylab.xlim( ( -500 , extinctionTrialsNum + (extinctionTrialsNum*4 + 4)*10 + 500) )

    tick_lcs = []
    tick_lbs = []
    for i in range ( 0 , 10 ):
        if i%2==0:
            tick_lbs.append( i+1 ) 
        else:
            tick_lbs.append( i+1 ) 
        tick_lcs.append(extinctionTrialsNum + i*(extinctionTrialsNum*4 + 4) + (extinctionTrialsNum*2 + 2) )
    pylab.xticks(tick_lcs, tick_lbs)
      
    leg = fig1.legend((S0, S1), ('Cocaine state','NoCocaine state'), loc = (0.38,0.62))
    leg.draw_frame(False)      
    
    for i in range ( 0 , 10 ):
        if i%2==0:
            p = pylab.axvspan( extinctionTrialsNum + i*(extinctionTrialsNum*4 + 4), extinctionTrialsNum + (i+1)*(extinctionTrialsNum*4 + 4) , facecolor='0.75',edgecolor='none', alpha=0.5)        

    for line in ax1.get_xticklines() + ax1.get_yticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)


    ax1.set_ylabel('Subjective probability')
    ax1.set_xlabel('Extinction session')
    ax1.set_title('')
    fig1.savefig('etimatedCocaineProbabilityLgA.eps', format='eps')

    return    



'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plot the Lever-press per session  ------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotLeverPressPerSession():

    infusionPerSessionLgA = numpy.zeros( [sessionsNum] , float)
    x = numpy.arange(1, 10+1, 1)
    
    for i in range(0,sessionsNum):
        for j in range( extinctionTrialsNum + i*((extinctionTrialsNum+1)*4) + (extinctionTrialsNum+1)*3 , extinctionTrialsNum + i*((extinctionTrialsNum+1)*4) + (extinctionTrialsNum+1)*4  ):
            infusionPerSessionLgA[i] = infusionPerSessionLgA[i] + activeLeverPressLgA[j]

        
    fig1 = pylab.figure( figsize=(5,3.5) )
    fig1.subplots_adjust(left=0.16)
    fig1.subplots_adjust(bottom=0.2)
    ax1 = fig1.add_subplot(111)
    S1 = ax1.plot(x,infusionPerSessionLgA [0:10] , '-o', ms=8, markeredgewidth =2, alpha=1, mfc='white',linewidth = 2 , color='black' )
    
    pylab.yticks(pylab.arange(0, 81, 20))
    pylab.ylim((0,80))
    pylab.xlim((0,10+1))
    pylab.xticks(pylab.arange(1, 10+1, 1))
    
    for i in range ( 0 , 10 ):
        if i%2==0:
            p = pylab.axvspan( 0.5 + i , 0.5 + i+1 , facecolor='0.75',edgecolor='none', alpha=0.5)        

    for line in ax1.get_xticklines() + ax1.get_yticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)

    ax1.set_ylabel('Responses / 45min')
    ax1.set_xlabel('Extinction session')
    ax1.set_title(' ')
    fig1.savefig('ResponsesPerSession.eps', format='eps')

    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plot the Lever-press in the first and last sessions, for every priming dose  ------------------ 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotLeverPressPerDose():

    infusionPerSessionLgAFisrt = numpy.zeros( [4] , float)
    infusionPerSessionLgALast  = numpy.zeros( [4] , float)
    x = numpy.arange(0, 4, 1)
    x[3] = 4
    
    for i in range(0,4):
        for j in range( extinctionTrialsNum + i*(extinctionTrialsNum+1) , extinctionTrialsNum + i*(extinctionTrialsNum+1) + extinctionTrialsNum ):
            infusionPerSessionLgAFisrt[i] = infusionPerSessionLgAFisrt[i] + activeLeverPressLgA[j]

    for i in range(0,4):
        for j in range( extinctionTrialsNum + (sessionsNum-1)*((extinctionTrialsNum+1)*4) + i*(extinctionTrialsNum+1) , extinctionTrialsNum + (sessionsNum-1)*((extinctionTrialsNum+1)*4) + i*(extinctionTrialsNum+1) + extinctionTrialsNum ):
            infusionPerSessionLgALast [i] = infusionPerSessionLgALast [i] + activeLeverPressLgA[j]
        
    fig1 = pylab.figure( figsize=(5,3.5) )
    fig1.subplots_adjust(left=0.16)
    fig1.subplots_adjust(bottom=0.2)
    ax1 = fig1.add_subplot(111)
    
    S0 = ax1.plot(x,infusionPerSessionLgAFisrt , '-o', ms=8, markeredgewidth =2, alpha=1, mfc='white',linewidth = 2 , color='black' )
    S1 = ax1.plot(x,infusionPerSessionLgALast  , '-o', ms=8, markeredgewidth =2, alpha=1, mfc='black',linewidth = 2 , color='black' )

    leg=fig1.legend((S0, S1), ('Pr1','Pr10'), loc = (0.20,0.68))
    leg.draw_frame(False)
    
    pylab.yticks(pylab.arange(0, 101, 20))
    pylab.ylim((-10,110))
    pylab.xlim((-1,5))

    tick_lcs = [0,1,2,4]
    tick_lbs = ['0','.25','.5','1']
    pylab.xticks(tick_lcs, tick_lbs)
      
    for line in ax1.get_xticklines() + ax1.get_yticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)

    ax1.set_ylabel('Responses / 45min')
    ax1.set_xlabel('Priming dose (mg/infusion)')
    ax1.set_title(' ')
    fig1.savefig('ResponsesPerDose.eps', format='eps')

    return


'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plot the Lever-press per session  ------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotLeverPressPer5Min():

    infusionPerSessionFirst = numpy.zeros( [5*9] , float)
    infusionPerSessionLast  = numpy.zeros( [4*9] , float)
    x = numpy.arange(1, 5*9 +1, 1)
    
    
    for i in range(0,5):
        for k in range(0,9):
            for j in range(  i*(extinctionTrialsNum+1) + k*75 , i*(extinctionTrialsNum+1) + k*75 + 75 ):
                infusionPerSessionFirst[i*9 + k] = infusionPerSessionFirst[i*9 + k] + activeLeverPressLgA[j]

    for i in range(0,4):
        for k in range(0,9):
            for j in range( extinctionTrialsNum + ((extinctionTrialsNum+1)*4)*(sessionsNum-1) + i*(extinctionTrialsNum+1) + k*75 + 1 , extinctionTrialsNum + ((extinctionTrialsNum+1)*4)*(sessionsNum-1) + i*(extinctionTrialsNum+1) + k*75 + 1 + 75 ):
                infusionPerSessionLast [i*9 + k] = infusionPerSessionLast [i*9 + k] + activeLeverPressLgA[j]

        
    fig1 = pylab.figure( figsize=(8,3.5) )
    fig1.subplots_adjust(left=0.16)
    fig1.subplots_adjust(bottom=0.2)
    ax1 = fig1.add_subplot(111)

    S1 = ax1.plot(x[9:18]  ,infusionPerSessionLast[0:9]    , '-o', ms=5, markeredgewidth =2, alpha=1, mfc='black',linewidth = 2 , color='black' )
    S1 = ax1.plot(x[18:27] ,infusionPerSessionLast[9:18]   , '-o', ms=5, markeredgewidth =2, alpha=1, mfc='black',linewidth = 2 , color='black' )
    S1 = ax1.plot(x[27:36],infusionPerSessionLast[18:27]   , '-o', ms=5, markeredgewidth =2, alpha=1, mfc='black',linewidth = 2 , color='black' )
    S1 = ax1.plot(x[36:45],infusionPerSessionLast[27:36]   , '-o', ms=5, markeredgewidth =2, alpha=1, mfc='black',linewidth = 2 , color='black' )
    
    S0 = ax1.plot(x[0:9]  ,infusionPerSessionFirst[0:9]    , '-o', ms=5, markeredgewidth =2, alpha=1, mfc='white',linewidth = 2 , color='black' )
    S0 = ax1.plot(x[9:18] ,infusionPerSessionFirst[9:18]   , '-o', ms=5, markeredgewidth =2, alpha=1, mfc='white',linewidth = 2 , color='black' )
    S0 = ax1.plot(x[18:27],infusionPerSessionFirst[18:27]  , '-o', ms=5, markeredgewidth =2, alpha=1, mfc='white',linewidth = 2 , color='black' )
    S0 = ax1.plot(x[27:36],infusionPerSessionFirst[27:36]  , '-o', ms=5, markeredgewidth =2, alpha=1, mfc='white',linewidth = 2 , color='black' )
    S0 = ax1.plot(x[36:45],infusionPerSessionFirst[36:45]  , '-o', ms=5, markeredgewidth =2, alpha=1, mfc='white',linewidth = 2 , color='black' )

    leg=fig1.legend((S0, S1), ('Pr1','Pr10'), loc = (0.16,0.23))
    leg.draw_frame(False)
    
    pylab.yticks(pylab.arange(0, 26, 5))
    pylab.ylim((-3,28))
    pylab.xlim( ( 0 , 46 ) )

    tick_lcs = []
    tick_lbs = []
    for i in range ( 0 , 6 ):
        tick_lcs.append( i*3*3  + 1 ) 
        tick_lbs.append( i*15*3 + 5 )
    pylab.xticks(tick_lcs, tick_lbs)


    for i in range ( 0 , 5 ):
        if i%2==0:
            p = pylab.axvspan( i*9 + 0.5 , (i+1) * 9 + 0.5, facecolor='0.75',edgecolor='none', alpha=0.5)        

    
    for line in ax1.get_xticklines() + ax1.get_yticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)

    ax1.set_ylabel('Responses / 5min')
    ax1.set_xlabel('Time (min)')
    ax1.set_title(' ')
    fig1.savefig('ResponsesPer5Min.eps', format='eps')

    return



'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plot all the results  ------------------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotting():

    loggingFinalization()

    plotInternalState() 
    plotLeverPressPerSession()    
    plotLeverPressPerDose()    
    plotLeverPressPer5Min()    
    plotEtimatedCocaineProbabilityLgA()    
    plotEtimatedCocaineProbabilityFirstLgA()

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
sessionsNum         = 15                            # Number of sessions of cocain seeking, followed by rest in home-cage
seekingHoursShA     = 1            
seekingHoursLgA     = 1            
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

totalTrialsNum      = extinctionTrialsNum + (extinctionTrialsNum+1) * 4 * sessionsNum

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

    extinction            ( 1, extinctionTrialsNum   ,'LgA')

    for session in range(0,sessionsNum):

        state[1] = initialInState
        
        noncontingentInfusion ( session*4+2 , 0                     ,'LgA')
        extinction            ( session*4+2 , extinctionTrialsNum   ,'LgA')
        noncontingentInfusion ( session*4+3 , cocaine*2             ,'LgA')
        extinction            ( session*4+3 , extinctionTrialsNum   ,'LgA')
        noncontingentInfusion ( session*4+4 , cocaine*4             ,'LgA')
        extinction            ( session*4+4 , extinctionTrialsNum   ,'LgA')    
        noncontingentInfusion ( session*4+5 , cocaine*8             ,'LgA')
        extinction            ( session*4+5 , extinctionTrialsNum   ,'LgA')    


plotting()
