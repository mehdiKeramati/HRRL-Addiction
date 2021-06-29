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
                estimatedTransition[i][j][k] = (1.0)/(statesNum*1.0)
                estimatedOutcome[i][j][k] = 0.0
                estimatedNonHomeostaticReward[i][j][k] = 0.0
    
#    Assuming that the animals know the energy cost (fatigue) of pressing a lever 
    for i in range(0,statesNum):
        for j in range(0,statesNum):
            estimatedNonHomeostaticReward[i][1][j] = -leverPressCost
    
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

        estimatedActionValues   = valueEstimation(exState,inState,setpointS,searchDepth)

        action                  = actionSelectionSoftmax(exState,estimatedActionValues)
        nextState               = getRealizedTransition(exState,action)
        out                     = getOutcome(exState,action,nextState)
        nonHomeoRew             = getNonHomeostaticReward(exState,action,nextState)
        HomeoRew                = driveReductionReward(inState,setpointS,out)

        if ratType=='ShA':  
            loggingShA (trial,action,inState,setpointS,out)    
            print "ShA rat number: %d / %d     Pre-training session     trial: %d / %d      animal seeking cocaine" %(animal+1,animalsNum,trial+1,trialsNum)
        elif ratType=='LgA':  
            loggingLgA (trial,action,inState,setpointS,out)    
            print "LgA rat number: %d / %d     Pre-training session     trial: %d / %d      animal seeking cocaine" %(animal+1,animalsNum,trial+1,trialsNum)

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
---------------------------------   Cocaine Seeking Sessions  --------------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def cocaineSeeking  (sessionNum , ratType):

    exState     = state[0]
    inState     = state[1]
    setpointS   = state[2]
    trialCount  = state[3]
    cocBuffer   = 0
    
    if sessionNum==1:
        if ratType=='ShA':
            inState = 100
        if ratType=='LgA':
            inState = 200
        
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
---------------------------------   Extinction Sessions  -------------------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def extinction  ( trialsNum , ratsType):

    exState     = state[0]
    inState     = state[1]
    setpointS   = state[2]
    trialCount  = state[3]
    cocBuffer   = 0
    
    for trial in range(trialCount,trialCount+trialsNum):
        
        estimatedActionValues   = valueEstimation               (exState,inState,setpointS,searchDepth)
        action                  = actionSelectionSoftmax        ( exState, estimatedActionValues    )
        nextState               = getRealizedTransition         ( exState, action                   )
        out                     = 0
        nonHomeoRew             = getNonHomeostaticReward       ( exState, action, nextState        )
        HomeoRew                = driveReductionReward          ( inState, setpointS, out           )
        
        if ratsType == 'ShA':
            loggingShA(trial,action,inState,setpointS,out)    
            print "ShA rat number: %d / %d     Extinction session       trial: %d / %d      Extinction of cocaine seeking" %(animal+1,animalsNum,trial-trialCount+1,trialsNum)
        if ratsType == 'LgA':
            loggingLgA(trial,action,inState,setpointS,out)    
            print "LgA rat number: %d / %d     Extinction session       trial: %d / %d      Extinction of cocaine seeking" %(animal+1,animalsNum,trial-trialCount+1,trialsNum)

        updateOutcomeFunction               ( exState, action, nextState, out            )
        updateNonHomeostaticRewardFunction  ( exState, action, nextState, nonHomeoRew    )
        updateTransitionFunction            ( exState, action, nextState,                )            
        
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
def noncontingentInfusion (sessionNum,ratType):

    exState     = state[0]
    inState     = state[1]
    setpointS   = state[2]
    trialCount  = state[3]

    inState     = updateInState(inState,cocaine)
    setpointS   = updateSetpoint(setpointS,cocaine)
    if ratType == 'ShA':
        loggingShA(trialCount,0,inState,setpointS,cocaine)    
        print "ShA rat number: %d / %d     Session Number: %d / %d                         animal receives non-contingent cocaine infusion" %(animal+1,animalsNum,sessionNum+1,sessionsNum)
    if ratType == 'LgA':
        loggingLgA(trialCount,0,inState,setpointS,cocaine)    
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
---------------------------------   Logging one animal   -------------------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def logIndividual(animalNum):

    infShA = numpy.zeros( [sessionsNum] , float)
    infLgA = numpy.zeros( [sessionsNum] , float)
    
    for i in range(0,sessionsNum):
        for j in range(trialsPerDay + i*(trialsPerDay),trialsPerDay + i*(trialsPerDay)+seekingTrialsNumShA):
            infShA[i] = infShA[i] + infusionShA[j]

    for i in range(0,sessionsNum):
        for j in range(trialsPerDay + i*(trialsPerDay),trialsPerDay + i*(trialsPerDay)+seekingTrialsNumLgA):
            infLgA[i] = infLgA[i] + infusionLgA[j]

    infusionPerSessionShA [animalNum] = infShA
    infusionPerSessionLgA [animalNum] = infLgA


    for i in range(0,sessionsNum+1):
        for j in range( i*(trialsPerDay), (i+1)*(trialsPerDay)):
            infusionShA[j] = 0
            infusionLgA[j] = 0

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
---------------------------------   Plot the infusions per session  --------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotSuppression():

    font = {'family' : 'normal', 'size'   : 16}
    pylab.rc('font', **font)
    pylab.rcParams.update({'legend.fontsize': 16})
    
    averageInfShA = numpy.zeros( [sessionsNum] , float)
    averageInfLgA = numpy.zeros( [sessionsNum] , float)

    s2ShA = numpy.zeros( [sessionsNum] , float)
    s2LgA = numpy.zeros( [sessionsNum] , float)

    varianceInfShA = numpy.zeros( [sessionsNum] , float)
    varianceInfLgA = numpy.zeros( [sessionsNum] , float)



    for j in range(0,sessionsNum):
        for i in range(0,animalsNum):
            averageInfShA[j] = averageInfShA[j] + infusionPerSessionShA[i][j]
            averageInfLgA[j] = averageInfLgA[j] + infusionPerSessionLgA[i][j]            
        averageInfShA[j] = averageInfShA[j] / animalsNum
        averageInfLgA[j] = averageInfLgA[j] / animalsNum


    for i in range(0,animalsNum):
        for j in range(0,sessionsNum):
            s2ShA[j]  = s2ShA[j]  +  infusionPerSessionShA[i][j]*infusionPerSessionShA[i][j]
            s2LgA[j]  = s2LgA[j]  +  infusionPerSessionLgA[i][j]*infusionPerSessionLgA[i][j]
    for j in range(0,sessionsNum):
        varianceInfShA[j] = numpy.sqrt(( s2ShA[j] -  averageInfShA[j]*averageInfShA[j]*animalsNum ) / animalsNum )
        varianceInfLgA[j] = numpy.sqrt(( s2LgA[j] -  averageInfLgA[j]*averageInfLgA[j]*animalsNum ) / animalsNum )


    for j in range(1,sessionsNum):
        averageInfShA [j] = averageInfShA [j]/averageInfShA [0]*100 
        averageInfLgA [j] = averageInfLgA [j]/averageInfLgA [0]*100 
        varianceInfShA[j] = varianceInfShA[j]*3  
        varianceInfLgA[j] = varianceInfLgA[j]*3

    averageInfShA [0] = 100 
    averageInfLgA [0] = 100  
    
    x = numpy.arange(1, sessionsNum+1, 1)
        
    fig1 = pylab.figure( figsize=(5,3.5) )
    fig1.subplots_adjust(left=0.16)
    fig1.subplots_adjust(bottom=0.2)
    ax1 = fig1.add_subplot(111)

    S0 = ax1.errorbar(x, averageInfShA , yerr= varianceInfShA,fmt='-o' , ms=8, markeredgewidth =2, alpha=1, mfc='black',linewidth = 2 , color='black')
    S1 = ax1.errorbar(x, averageInfLgA , yerr= varianceInfLgA,fmt='-o',  ms=8, markeredgewidth =2, alpha=1, mfc='white',linewidth = 2 , color='black' )

    leg=fig1.legend((S1, S0), ('LgA','ShA'), loc = (0.65,0.55))
    leg.draw_frame(False)
        
    pylab.ylim((0,110))
    pylab.yticks(pylab.arange(0, 111, 20))
    pylab.xlim((0,sessionsNum+1))
    
    tick_lcs = []
    tick_lbs = ['H1','H2','1','2','3']
    for i in range ( 1 , 6 ):
        tick_lcs.append( i ) 
    pylab.xticks(tick_lcs, tick_lbs)

    p = pylab.axvspan( 1.5 , 2.5 , facecolor='0.75',edgecolor='none', alpha=0.5)        
    
    for line in ax1.get_xticklines() + ax1.get_yticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)

    ax1.set_ylabel('% suppression')
    ax1.set_xlabel('                    Post-punishment day')
    ax1.set_title('Total intake')
    fig1.savefig('suppression.eps', format='eps')

    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plot punishment-induced suppression ----------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotSuppressionInPunishmentPhase():

    averageInfShA = numpy.zeros( [sessionsNum] , float)
    averageInfLgA = numpy.zeros( [sessionsNum] , float)

    suppShA = numpy.zeros( [animalsNum] , float)
    suppLgA = numpy.zeros( [animalsNum] , float)


    for j in range(0,sessionsNum):
        for i in range(0,animalsNum):
            averageInfShA[j] = averageInfShA[j] + infusionPerSessionShA[i][j]
            averageInfLgA[j] = averageInfLgA[j] + infusionPerSessionLgA[i][j]            
        averageInfShA[j] = averageInfShA[j] / animalsNum
        averageInfLgA[j] = averageInfLgA[j] / animalsNum


    for i in range(0,animalsNum):
        suppShA[i] = (infusionPerSessionShA[i][1]/infusionPerSessionShA[i][0])*100
        suppLgA[i] = (infusionPerSessionLgA[i][1]/infusionPerSessionLgA[i][0])*100

    sSuppShA = 0
    sSuppLgA = 0
    s2SuppShA = 0
    s2SuppLgA = 0
    for i in range(0,animalsNum):
        sSuppShA = sSuppShA + suppShA[i]
        sSuppLgA = sSuppLgA + suppLgA[i]
        s2SuppShA= s2SuppShA+ suppShA[i]*suppShA[i]
        s2SuppLgA= s2SuppLgA+ suppLgA[i]*suppLgA[i]
        
    varianceSuppShA = ( s2SuppShA - (sSuppShA*sSuppShA)/animalsNum ) / animalsNum
    varianceSuppLgA = ( s2SuppLgA - (sSuppLgA*sSuppLgA)/animalsNum ) / animalsNum



    supp = ( (averageInfLgA[1]/averageInfLgA[0])*100 , (averageInfShA[1]/averageInfShA[0])*100 )
    suppStd =   (varianceSuppShA,varianceSuppLgA)
    
    x = numpy.arange(2)  # the x locations for the groups
    wid = 0.6
    
    fig1 = pylab.figure( figsize=(2,3.5))
    fig1.subplots_adjust(bottom=0.2)
    fig1.subplots_adjust(left=0.26)
    ax1 = fig1.add_subplot(111)
    bars = ax1.bar(x, supp, color='black', width=wid , lw = 2 , yerr=suppStd,align='center',ecolor='black')
    
    bars[1].set_facecolor('white')
    pylab.xticks( x, ('ShA', 'LgA') )
    
    pylab.ylim((0,110))
    pylab.yticks(pylab.arange(0, 111, 20))
    pylab.xlim((-0.8,1.8))
    for line in ax1.get_xticklines() + ax1.get_yticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)

    ax1.set_title(' ')
    ax1.set_ylabel('% suppression')
    ax1.set_xticklabels( ('ShA', 'LgA') )
    fig1.savefig('suppressionInPunishmentPhase.eps', format='eps')

    return


'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plot all the results  ------------------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotting():

    plotSuppression()
    plotSuppressionInPunishmentPhase()
    
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
nonContingentCocaine = 50
leverPressCost  = 1              # Energy cost for pressing the lever
punishment      = 380.72

statesNum       = 5              # number of stater 
actionsNum      = 3              # number of action   action 0 = Null     action 1 = Inactive Lever Press    action 2 = Active Lever Press
initialExState  = 0

transition = numpy.zeros( [statesNum , actionsNum, statesNum] , float)
setTransition(0,0,0,1)          # From state s, and by taking a, we go to state s', with probability p
setTransition(0,1,0,1)
setTransition(0,2,1,1)
setTransition(1,0,2,1)
setTransition(1,1,2,1)
setTransition(1,2,2,1)
setTransition(2,0,3,1)
setTransition(2,1,3,1)
setTransition(2,2,3,1)
setTransition(3,0,4,1)
setTransition(3,1,4,1)
setTransition(3,2,4,1)
setTransition(4,0,0,1)
setTransition(4,1,0,1)
setTransition(4,2,0,1)

outcome = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )
setOutcome(0,2,1,cocaine)       # At state s, by doing action a and going to state s', we receive the outcome 

nonHomeostaticReward = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )
setNonHomeostaticReward(0,1,0,-leverPressCost)
setNonHomeostaticReward(0,2,1,-leverPressCost)
setNonHomeostaticReward(1,1,2,-leverPressCost)
setNonHomeostaticReward(1,2,2,-leverPressCost)
setNonHomeostaticReward(3,1,4,-leverPressCost)
setNonHomeostaticReward(3,2,4,-leverPressCost)
setNonHomeostaticReward(4,1,0,-leverPressCost)
setNonHomeostaticReward(4,2,0,-leverPressCost)

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Definition of the Animal   --------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''

#------------ Homeostatic System
initialInState          = 0
initialSetpoint         = 100
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
beta                    = 0.25  # Rate of exploration
searchDepth             = 3     # Depth of going into the decision tree for goal-directed valuation of choices
pruningThreshold        = 0.1   # If the probability of a transition like (s,a,s') is less than "pruningThreshold", cut it from the decision tree 

estimatedTransition              = numpy.zeros( [statesNum , actionsNum, statesNum] , float)
estimatedOutcome                 = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )
estimatedNonHomeostaticReward    = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )

state                            = numpy.zeros ( [4] , float )     # a vector of the external state, internal state, setpoint, and trial

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Simulation Parameters   -----------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''

animalsNum          = 6                                  # Number of animals

pretrainingHours    = 1
sessionsNum         = 5                                  # Number of sessions of cocain seeking, followed by rest in home-cage
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

totalTrialsNum      = trialsPerDay + sessionsNum * (trialsPerDay)  #+ extinctionTrialsNum*2 + 1

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

infusionPerSessionShA =  numpy.zeros( [animalsNum,5] , float)
infusionPerSessionLgA =  numpy.zeros( [animalsNum,5] , float)

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Simulation   ----------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''

for animal in range(0,animalsNum):

    initialSetpoint         = 100
    setNonHomeostaticReward ( 0 , 2 , 1 , -leverPressCost )
    initializeAnimal      (                         )
    pretraining           ( 'ShA'                   )    
    homeCage              ( 0,'afterPretrainingShA' ) 

    setNonHomeostaticReward ( 0 , 2 , 1 , -leverPressCost )
    cocaineSeeking        (  0 , 'ShA'              )
    homeCage              (  0 , 'ShA'              ) 
        
    setNonHomeostaticReward ( 0 , 2 , 1 , -leverPressCost - punishment )
    cocaineSeeking        (  1 , 'ShA'              )
    homeCage              (  1 , 'ShA'              ) 

    setNonHomeostaticReward ( 0 , 2 , 1 , -leverPressCost )
    cocaineSeeking        (  2 , 'ShA'              )
    homeCage              (  2 , 'ShA'              ) 
    cocaineSeeking        (  3 , 'ShA'              )
    homeCage              (  3 , 'ShA'              ) 
    cocaineSeeking        (  4 , 'ShA'              )
    homeCage              (  4 , 'ShA'              ) 



    initialSetpoint         = 200
    setNonHomeostaticReward ( 0 , 2 , 1 , -leverPressCost )
    initializeAnimal      (                         )
    pretraining           ( 'LgA'                   )    
    homeCage              ( 0,'afterPretrainingLgA' ) 

    setNonHomeostaticReward ( 0 , 2 , 1 , -leverPressCost )
    cocaineSeeking        (  0 , 'LgA'              )
    homeCage              (  0 , 'LgA'              ) 
        
    setNonHomeostaticReward ( 0 , 2 , 1 , -leverPressCost - punishment )
    cocaineSeeking        (  1 , 'LgA'              )
    homeCage              (  1 , 'LgA'              ) 

    setNonHomeostaticReward ( 0 , 2 , 1 , -leverPressCost )
    cocaineSeeking        (  2 , 'LgA'              )
    homeCage              (  2 , 'LgA'              ) 
    cocaineSeeking        (  3 , 'LgA'              )
    homeCage              (  3 , 'LgA'              ) 
    cocaineSeeking        (  4 , 'LgA'              )
    homeCage              (  4 , 'LgA'              ) 

    
    logIndividual         (  animal                 )


plotting()

