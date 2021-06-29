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

        if ratType=='1':  
            logging1 (trial,action,inState,setpointS,out)    
            print "1 hour rat number: %d / %d     Pre-training session     trial: %d / %d      animal seeking cocaine" %(animal+1,animalsNum,trial+1,trialsNum)
        elif ratType=='3':  
            logging3 (trial,action,inState,setpointS,out)    
            print "3 hour rat number: %d / %d     Pre-training session     trial: %d / %d      animal seeking cocaine" %(animal+1,animalsNum,trial+1,trialsNum)
        elif ratType=='6':  
            logging3 (trial,action,inState,setpointS,out)    
            print "6 hour rat number: %d / %d     Pre-training session     trial: %d / %d      animal seeking cocaine" %(animal+1,animalsNum,trial+1,trialsNum)
        elif ratType=='12':  
            logging12 (trial,action,inState,setpointS,out)    
            print "12 hour rat number: %d / %d    Pre-training session     trial: %d / %d      animal seeking cocaine" %(animal+1,animalsNum,trial+1,trialsNum)

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
    
    if ratType=='1':  
        trialsNum = seekingTrialsNum1    
    elif ratType=='3':  
        trialsNum = seekingTrialsNum3    
    elif ratType=='6':  
        trialsNum = seekingTrialsNum6    
    elif ratType=='12':  
        trialsNum = seekingTrialsNum12    
    
    for trial in range(trialCount,trialCount+trialsNum):

        estimatedActionValues   = valueEstimation(exState,inState,setpointS,searchDepth)
        action                  = actionSelectionSoftmax(exState,estimatedActionValues)
        nextState               = getRealizedTransition(exState,action)
        out                     = getOutcome(exState,action,nextState)
        nonHomeoRew             = getNonHomeostaticReward(exState,action,nextState)
        HomeoRew                = driveReductionReward(inState,setpointS,out)

        if ratType=='1':  
            logging1(trial,action,inState,setpointS,out)    
            print "1 hour rat number: %d / %d     Session Number: %d / %d     trial: %d / %d      animal seeking cocaine" %(animal+1,animalsNum,sessionNum+1,sessionsNum,trial-trialCount+1,trialsNum)
        elif ratType=='3':  
            logging3(trial,action,inState,setpointS,out)    
            print "3 hour rat number: %d / %d     Session Number: %d / %d     trial: %d / %d      animal seeking cocaine" %(animal+1,animalsNum,sessionNum+1,sessionsNum,trial-trialCount+1,trialsNum)
        elif ratType=='6':  
            logging6(trial,action,inState,setpointS,out)    
            print "6 hour rat number: %d / %d     Session Number: %d / %d     trial: %d / %d      animal seeking cocaine" %(animal+1,animalsNum,sessionNum+1,sessionsNum,trial-trialCount+1,trialsNum)
        elif ratType=='12':  
            logging12(trial,action,inState,setpointS,out)    
            print "12 hour rat number: %d / %d     Session Number: %d / %d     trial: %d / %d      animal seeking cocaine" %(animal+1,animalsNum,sessionNum+1,sessionsNum,trial-trialCount+1,trialsNum)

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
 
    if ratType=='1':  
        trialsNum = restTrialsNum1    
        print "1 hour rat number: %d / %d     Session Number: %d / %d                          animal rests in home cage" %(animal+1,animalsNum,sessionNum+1,sessionsNum)
    elif ratType=='3':  
        trialsNum = restTrialsNum3    
        print "3 hour rat number: %d / %d     Session Number: %d / %d                          animal rests in home cage" %(animal+1,animalsNum,sessionNum+1,sessionsNum)
    elif ratType=='6':  
        trialsNum = restTrialsNum6    
        print "6 hour rat number: %d / %d     Session Number: %d / %d                          animal rests in home cage" %(animal+1,animalsNum,sessionNum+1,sessionsNum)
    elif ratType=='12':  
        trialsNum = restTrialsNum12    
        print "12 hour rat number: %d / %d     Session Number: %d / %d                          animal rests in home cage" %(animal+1,animalsNum,sessionNum+1,sessionsNum)
    elif ratType=='afterPretraining1':  
        trialsNum = restAfterPretrainingTrialsNum    
        print "1 hour rat number: %d / %d     After pretraining                                animal rests in home cage" %(animal+1,animalsNum)
    elif ratType=='afterPretraining3':  
        trialsNum = restAfterPretrainingTrialsNum    
        print "3 hour rat number: %d / %d     After pretraining                                animal rests in home cage" %(animal+1,animalsNum)
    elif ratType=='afterPretraining6':  
        trialsNum = restAfterPretrainingTrialsNum    
        print "6 hour rat number: %d / %d     After pretraining                                animal rests in home cage" %(animal+1,animalsNum)
    elif ratType=='afterPretraining12':  
        trialsNum = restAfterPretrainingTrialsNum    
        print "12 hour rat number: %d / %d     After pretraining                                animal rests in home cage" %(animal+1,animalsNum)

     
    for trial in range(trialCount,trialCount+trialsNum):
        inState     = updateInState(inState,0)
        setpointS   = updateSetpoint(setpointS,0)


        if ratType=='1':  
            logging1(trial,0,inState,setpointS,0)    
        elif ratType=='3':  
            logging3(trial,0,inState,setpointS,0)    
        elif ratType=='6':  
            logging6(trial,0,inState,setpointS,0)    
        elif ratType=='12':  
            logging12(trial,0,inState,setpointS,0)    
        elif ratType=='afterPretraining1':  
            logging1(trial,0,inState,setpointS,0)    
        elif ratType=='afterPretraining3':  
            logging3(trial,0,inState,setpointS,0)    
        elif ratType=='afterPretraining6':  
            logging6(trial,0,inState,setpointS,0)    
        elif ratType=='afterPretraining12':  
            logging12(trial,0,inState,setpointS,0)    

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
---------------------------------   Logging the current information for the 1-hour group  ----------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def logging1(trial,action,inState,setpointS,coca):
   
    if action==0: 
        nulDoing1[trial]             = nulDoing1[trial] + 1
    elif action==1: 
        inactiveLeverPress1[trial]   = inactiveLeverPress1[trial] + 1
    elif action==2: 
        activeLeverPress1[trial]     = activeLeverPress1[trial] + 1
    internalState1[trial]    = internalState1[trial] + inState
    setpoint1[trial]         = setpoint1[trial] + setpointS    
    if coca==cocaine:
        infusion1[trial]     = infusion1[trial] + 1
    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Logging the current information for the 3-hour group  ----------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def logging3(trial,action,inState,setpointS,coca):
   
    if action==0: 
        nulDoing3[trial]             = nulDoing3[trial] + 1
    elif action==1: 
        inactiveLeverPress3[trial]   = inactiveLeverPress3[trial] + 1
    elif action==2: 
        activeLeverPress3[trial]     = activeLeverPress3[trial] + 1
    internalState3[trial]    = internalState3[trial] + inState
    setpoint3[trial]         = setpoint3[trial] + setpointS    
    if coca==cocaine:
        infusion3[trial]     = infusion3[trial] + 1
    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Logging the current information for the 6-hour group  ----------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def logging6(trial,action,inState,setpointS,coca):
   
    if action==0: 
        nulDoing6[trial]             = nulDoing6[trial] + 1
    elif action==1: 
        inactiveLeverPress6[trial]   = inactiveLeverPress6[trial] + 1
    elif action==2: 
        activeLeverPress6[trial]     = activeLeverPress6[trial] + 1
    internalState6[trial]    = internalState6[trial] + inState
    setpoint6[trial]         = setpoint6[trial] + setpointS    
    if coca==cocaine:
        infusion6[trial]     = infusion6[trial] + 1
    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Logging the current information for the 12-hour group  ---------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def logging12(trial,action,inState,setpointS,coca):
   
    if action==0: 
        nulDoing12[trial]             = nulDoing12[trial] + 1
    elif action==1: 
        inactiveLeverPress12[trial]   = inactiveLeverPress12[trial] + 1
    elif action==2: 
        activeLeverPress12[trial]     = activeLeverPress12[trial] + 1
    internalState12[trial]    = internalState12[trial] + inState
    setpoint12[trial]         = setpoint12[trial] + setpointS    
    if coca==cocaine:
        infusion12[trial]     = infusion12[trial] + 1
    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Wrap up all the logged data   ----------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def loggingFinalization():
    
    for trial in range(0,totalTrialsNum):
        nulDoing1[trial]             = nulDoing1[trial]/animalsNum
        inactiveLeverPress1[trial]   = inactiveLeverPress1[trial]/animalsNum
        activeLeverPress1[trial]     = activeLeverPress1[trial]/animalsNum
        internalState1[trial]        = internalState1[trial]/animalsNum
        setpoint1[trial]             = setpoint1[trial]/animalsNum  
        infusion1[trial]             = infusion1[trial]/animalsNum 

        nulDoing3[trial]             = nulDoing3[trial]/animalsNum
        inactiveLeverPress3[trial]   = inactiveLeverPress3[trial]/animalsNum
        activeLeverPress3[trial]     = activeLeverPress3[trial]/animalsNum
        internalState3[trial]        = internalState3[trial]/animalsNum
        setpoint3[trial]             = setpoint3[trial]/animalsNum  
        infusion3[trial]             = infusion3[trial]/animalsNum 

        nulDoing6[trial]             = nulDoing6[trial]/animalsNum
        inactiveLeverPress6[trial]   = inactiveLeverPress6[trial]/animalsNum
        activeLeverPress6[trial]     = activeLeverPress6[trial]/animalsNum
        internalState6[trial]        = internalState6[trial]/animalsNum
        setpoint6[trial]             = setpoint6[trial]/animalsNum  
        infusion6[trial]             = infusion6[trial]/animalsNum 

        nulDoing12[trial]            = nulDoing12[trial]/animalsNum
        inactiveLeverPress12[trial]  = inactiveLeverPress12[trial]/animalsNum
        activeLeverPress12[trial]    = activeLeverPress12[trial]/animalsNum
        internalState12[trial]       = internalState12[trial]/animalsNum
        setpoint12[trial]            = setpoint12[trial]/animalsNum  
        infusion12[trial]            = infusion12[trial]/animalsNum 

    return


'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plot the setpoint  ---------------------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotSetpoint():

    font = {'family' : 'normal', 'size'   : 16}
    pylab.rc('font', **font)
    pylab.rcParams.update({'legend.fontsize': 16})    
      
    fig1 = pylab.figure( figsize=(6.3,3.5) )
    fig1.subplots_adjust(left=0.16)
    fig1.subplots_adjust(bottom=0.2)
    ax1 = fig1.add_subplot(111)

    ax1.axhline(100,  color='0.25',ls='--', lw=1 )
    ax1.axhline(200, color='0.25',ls='--', lw=1 )

    S0 = ax1.plot(setpoint1 [trialsPerDay : trialsPerDay*(sessionsNum+1)] , '-' , linewidth = 1 , color='black' )
    S1 = ax1.plot(setpoint3 [trialsPerDay : trialsPerDay*(sessionsNum+1)] , '-' , linewidth = 1 , color='0.4' )
    S2 = ax1.plot(setpoint6 [trialsPerDay : trialsPerDay*(sessionsNum+1)] , '-' , linewidth = 2 , color='black' )
    S3 = ax1.plot(setpoint12[trialsPerDay : trialsPerDay*(sessionsNum+1)] , '-' , linewidth = 2 , color='0.4' )
  
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    leg = ax1.legend((S3,S2,S1, S0), ('12 hr','  6 hr','  3 hr','  1 hr'), loc='center left' , bbox_to_anchor=(1, 0.5))
    leg.draw_frame(False)
      
    pylab.yticks(pylab.arange(100, 201, 20))
    pylab.ylim((90,210))

    tick_lcs = []
    tick_lbs = []
    for i in range ( 0 , sessionsNum ):
        tick_lcs.append( trialsPerDay*i + trialsPerDay/2) 
        if i%4==0:
            tick_lbs.append(i+1)
        else:
            tick_lbs.append(' ')
    pylab.xticks(tick_lcs, tick_lbs)

    for i in range ( 0 , sessionsNum ):
        if i%2==0:
            p = pylab.axvspan( i*trialsPerDay , (i+1) * trialsPerDay , facecolor='0.75',edgecolor='none', alpha=0.5)        

    for line in ax1.get_yticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)

    for line in ax1.get_xticklines():
        line.set_markeredgewidth(0)
        line.set_markersize(0)
    
    ax1.set_ylabel('Homeostatic setpoint')
    ax1.set_xlabel('Day')
    ax1.set_title('')
    fig1.savefig('setpoint.eps', format='eps')

    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plot the setpoint during days 2, 3, and 4 ----------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotSetpoint234():
    
      
    fig1 = pylab.figure( figsize=(6.3,3.5) )
    fig1.subplots_adjust(left=0.16)
    fig1.subplots_adjust(bottom=0.2)
    ax1 = fig1.add_subplot(111)

    ax1.axhline(100,  color='0.25',ls='--', lw=1 )

    S0 = ax1.plot(setpoint1 [ trialsPerDay*2 - trialsPerHour*2 : trialsPerDay*5 + trialsPerHour*2] , '-' , linewidth = 1 , color='black' )
    S1 = ax1.plot(setpoint3 [ trialsPerDay*2 - trialsPerHour*2 : trialsPerDay*5 + trialsPerHour*2] , '-' , linewidth = 1 , color='0.4' )
    S2 = ax1.plot(setpoint6 [ trialsPerDay*2 - trialsPerHour*2 : trialsPerDay*5 + trialsPerHour*2] , '-' , linewidth = 2 , color='black' )
    S3 = ax1.plot(setpoint12[ trialsPerDay*2 - trialsPerHour*2 : trialsPerDay*5 + trialsPerHour*2] , '-' , linewidth = 2 , color='0.4'   )
  
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    leg = ax1.legend((S3,S2,S1, S0), ('12 hr','  6 hr','  3 hr','  1 hr'), loc='center left' , bbox_to_anchor=(1, 0.5))
    leg.draw_frame(False)
      
    pylab.yticks(pylab.arange(100, 151, 10))
    pylab.ylim((98,155))

    tick_lcs = []
    tick_lbs = []
    for i in range ( 2 , 5 ):
        tick_lcs.append( trialsPerHour*2 + trialsPerDay/2 + trialsPerDay*(i-2) ) 
        tick_lbs.append(i)
    pylab.xticks(tick_lcs, tick_lbs)

    p = pylab.axvspan( trialsPerHour*2 + trialsPerDay , trialsPerHour*2 + trialsPerDay*2 , ymin=0 , ymax=1 , facecolor='0.75',edgecolor='none', alpha=0.5)        

    for line in ax1.get_yticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)

    for line in ax1.get_xticklines():
        line.set_markeredgewidth(0)
        line.set_markersize(0)
    
    ax1.set_ylabel('Homeostatic setpoint')
    ax1.set_xlabel('Day')
    ax1.set_title('')
    fig1.savefig('setpoint234.eps', format='eps')

    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plot the infusions per session  --------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotInfusionPerSession():

    infusionPerSession1 = numpy.zeros( [sessionsNum+1] , float)
    infusionPerSession3 = numpy.zeros( [sessionsNum+1] , float)
    infusionPerSession6 = numpy.zeros( [sessionsNum+1] , float)
    infusionPerSession12 = numpy.zeros( [sessionsNum+1] , float)
    x = numpy.arange(1, sessionsNum+1, 1)
    
    for i in range(0,sessionsNum):
        for j in range(trialsPerDay + i*(trialsPerDay),trialsPerDay + i*(trialsPerDay)+seekingTrialsNum1):
            infusionPerSession1[i+1] = infusionPerSession1[i+1] + infusion1[j]
    for i in range(0,sessionsNum):
        for j in range(trialsPerDay + i*(trialsPerDay),trialsPerDay + i*(trialsPerDay)+seekingTrialsNum3):
            infusionPerSession3[i+1] = infusionPerSession3[i+1] + infusion3[j]
    for i in range(0,sessionsNum):
        for j in range(trialsPerDay + i*(trialsPerDay),trialsPerDay + i*(trialsPerDay)+seekingTrialsNum6):
            infusionPerSession6[i+1] = infusionPerSession6[i+1] + infusion6[j]
    for i in range(0,sessionsNum):
        for j in range(trialsPerDay + i*(trialsPerDay),trialsPerDay + i*(trialsPerDay)+seekingTrialsNum12):
            infusionPerSession12[i+1] = infusionPerSession12[i+1] + infusion12[j]

        
    fig1 = pylab.figure( figsize=(6.3,3.5) )
    fig1.subplots_adjust(left=0.16)
    fig1.subplots_adjust(bottom=0.2)
    ax1 = fig1.add_subplot(111)

    S0 = ax1.plot(x,infusionPerSession1[1:sessionsNum+1] , '-v', ms=5, markeredgewidth =2, alpha=1, mfc='black',linewidth = 2 , color='black' )
    S1 = ax1.plot(x,infusionPerSession3[1:sessionsNum+1] , '-s', ms=5, markeredgewidth =2, alpha=1, mfc='black',linewidth = 2 , color='black' )
    S2 = ax1.plot(x,infusionPerSession6[1:sessionsNum+1] , '-^', ms=5, markeredgewidth =2, alpha=1, mfc='black',linewidth = 2 , color='black' )
    S3 = ax1.plot(x,infusionPerSession12[1:sessionsNum+1], '-o', ms=5, markeredgewidth =2, alpha=1, mfc='black',linewidth = 2 , color='black' )

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    leg = ax1.legend((S3,S2,S1,S0), ('12 hr','  6 hr','  3 hr','  1 hr'), loc='center left' , bbox_to_anchor=(1, 0.5))
    leg.draw_frame(False)
    
    pylab.yticks(pylab.arange(0, 301, 50))
    pylab.ylim((0,310))
    pylab.xlim((0,sessionsNum+1))
    pylab.xticks(pylab.arange(1, sessionsNum+1, 4))
    
    for line in ax1.get_xticklines() + ax1.get_yticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)

    ax1.set_ylabel('Infusions / Session')
    ax1.set_xlabel('Session')
    ax1.set_title('Total intake')
    fig1.savefig('infusionPerSession.eps', format='eps')

    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plot the infusions per first hour  ------------------------------------------------------------ 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotInfusionPerFirstHour():

    infusionPerHour1 = numpy.zeros( [sessionsNum+1] , float)
    infusionPerHour3 = numpy.zeros( [sessionsNum+1] , float)
    infusionPerHour6 = numpy.zeros( [sessionsNum+1] , float)
    infusionPerHour12= numpy.zeros( [sessionsNum+1] , float)
    x = numpy.arange(1, sessionsNum+1, 1)
    
    for i in range(0,sessionsNum):
        for j in range(trialsPerDay + i*(trialsPerDay),trialsPerDay + i*(trialsPerDay)+trialsPerHour):
            infusionPerHour1[i+1] = infusionPerHour1[i+1] + infusion1[j]
            infusionPerHour3[i+1] = infusionPerHour3[i+1] + infusion3[j]
            infusionPerHour6[i+1] = infusionPerHour6[i+1] + infusion6[j]
            infusionPerHour12[i+1]= infusionPerHour12[i+1]+ infusion12[j]
        
    fig1 = pylab.figure( figsize=(6.3,3.5) )
    fig1.subplots_adjust(left=0.16)
    fig1.subplots_adjust(bottom=0.2)
    ax1 = fig1.add_subplot(111)
    S0 = ax1.plot(x,infusionPerHour1[1:sessionsNum+1] , '-v', ms=5, markeredgewidth =2, alpha=1, mfc='black',linewidth = 2 , color='black' )
    S1 = ax1.plot(x,infusionPerHour3[1:sessionsNum+1] , '-s', ms=5, markeredgewidth =2, alpha=1, mfc='black',linewidth = 2 , color='black' )
    S2 = ax1.plot(x,infusionPerHour6[1:sessionsNum+1] , '-^', ms=5, markeredgewidth =2, alpha=1, mfc='black',linewidth = 2 , color='black' )
    S3 = ax1.plot(x,infusionPerHour12[1:sessionsNum+1], '-o', ms=5, markeredgewidth =2, alpha=1, mfc='black',linewidth = 2 , color='black' )

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    leg = ax1.legend((S3,S2,S1, S0), ('12 hr','  6 hr','  3 hr','  1 hr'), loc='center left' , bbox_to_anchor=(1, 0.5))
    leg.draw_frame(False)
    
    pylab.yticks(pylab.arange(0, 31, 5))
    pylab.ylim((0,35))
    pylab.xlim((0,sessionsNum+1))
    pylab.xticks(pylab.arange(1, sessionsNum+1, 4))
    
    for line in ax1.get_xticklines() + ax1.get_yticklines():
        line.set_markeredgewidth(2)
        line.set_markersize(5)

    ax1.set_ylabel('Infusions / First hour')
    ax1.set_xlabel('Session')
    ax1.set_title('First hour')
    fig1.savefig('infusionPerFirstHour.eps', format='eps')

    return
'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plot all the results  ------------------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotting():

    loggingFinalization()

    plotSetpoint()
    plotSetpoint234()
    plotInfusionPerSession()
    plotInfusionPerFirstHour()
    
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

animalsNum          = 1                                  # Number of animals

pretrainingHours    = 1
sessionsNum         = 21                                  # Number of sessions of cocain seeking, followed by rest in home-cage
seekingHours1       = 1            
seekingHours3       = 3            
seekingHours6       = 6            
seekingHours12      = 12            
extinctionHours     = 0

trialsPerHour       = 60*60/4                            # Number of trials during one hour (as each trial is supposed to be 4 seconds)
trialsPerDay        = 24*trialsPerHour
pretrainingTrialsNum= pretrainingHours* trialsPerHour
restAfterPretrainingTrialsNum = (24 - pretrainingHours) *trialsPerHour

seekingTrialsNum1 = seekingHours1 * trialsPerHour    # Number of trials for each cocaine seeking session
restingHours1     = 24 - seekingHours1
restTrialsNum1    = restingHours1 * trialsPerHour    # Number of trials for each session of the animal being in the home cage

seekingTrialsNum3 = seekingHours3 * trialsPerHour    # Number of trials for each cocaine seeking session
restingHours3     = 24 - seekingHours3
restTrialsNum3    = restingHours3 * trialsPerHour    # Number of trials for each session of the animal being in the home cage

seekingTrialsNum6 = seekingHours6 * trialsPerHour    # Number of trials for each cocaine seeking session
restingHours6     = 24 - seekingHours6
restTrialsNum6    = restingHours6 * trialsPerHour    # Number of trials for each session of the animal being in the home cage

seekingTrialsNum12= seekingHours12 * trialsPerHour    # Number of trials for each cocaine seeking session
restingHours12    = 24 - seekingHours12
restTrialsNum12   = restingHours12 * trialsPerHour    # Number of trials for each session of the animal being in the home cage

extinctionTrialsNum = extinctionHours*trialsPerHour      # Number of trials for each extinction session

totalTrialsNum      = trialsPerDay + sessionsNum * (trialsPerDay)  #+ extinctionTrialsNum*2 + 1

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plotting Parameters   -------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''

trialsPerBlock = 10*60/4            # Each BLOCK is 10 minutes - Each minute 60 second - Each trial takes 4 seconds

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Logging Parameters   --------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''

nulDoing1            = numpy.zeros( [totalTrialsNum] , float)
inactiveLeverPress1  = numpy.zeros( [totalTrialsNum] , float)
activeLeverPress1    = numpy.zeros( [totalTrialsNum] , float)
internalState1       = numpy.zeros( [totalTrialsNum] , float)
setpoint1            = numpy.zeros( [totalTrialsNum] , float)
infusion1            = numpy.zeros( [totalTrialsNum] , float)

nulDoing3            = numpy.zeros( [totalTrialsNum] , float)
inactiveLeverPress3  = numpy.zeros( [totalTrialsNum] , float)
activeLeverPress3    = numpy.zeros( [totalTrialsNum] , float)
internalState3       = numpy.zeros( [totalTrialsNum] , float)
setpoint3            = numpy.zeros( [totalTrialsNum] , float)
infusion3            = numpy.zeros( [totalTrialsNum] , float)

nulDoing6            = numpy.zeros( [totalTrialsNum] , float)
inactiveLeverPress6  = numpy.zeros( [totalTrialsNum] , float)
activeLeverPress6    = numpy.zeros( [totalTrialsNum] , float)
internalState6       = numpy.zeros( [totalTrialsNum] , float)
setpoint6            = numpy.zeros( [totalTrialsNum] , float)
infusion6            = numpy.zeros( [totalTrialsNum] , float)

nulDoing12           = numpy.zeros( [totalTrialsNum] , float)
inactiveLeverPress12 = numpy.zeros( [totalTrialsNum] , float)
activeLeverPress12   = numpy.zeros( [totalTrialsNum] , float)
internalState12      = numpy.zeros( [totalTrialsNum] , float)
setpoint12           = numpy.zeros( [totalTrialsNum] , float)
infusion12           = numpy.zeros( [totalTrialsNum] , float)

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Simulation   ----------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''

for animal in range(0,animalsNum):

    initializeAnimal          (                       )
    pretraining               ( '1'                   )    
    homeCage                  ( 0,'afterPretraining1' ) 
    for session in range(0,sessionsNum):
        cocaineSeeking        (  session , '1'        )
        homeCage              (  session , '1'        ) 
        
    initializeAnimal          (                       )
    pretraining               ( '3'                   )    
    homeCage                  ( 0,'afterPretraining3' ) 
    for session in range(0,sessionsNum):
        cocaineSeeking        (  session , '3'        )
        homeCage              (  session , '3'        ) 
        
    initializeAnimal          (                       )
    pretraining               ( '6'                   )    
    homeCage                  ( 0,'afterPretraining6' ) 
    for session in range(0,sessionsNum):
        cocaineSeeking        (  session , '6'        )
        homeCage              (  session , '6'        ) 
        
    initializeAnimal          (                       )
    pretraining               ( '12'                  )    
    homeCage                  ( 0,'afterPretraining12') 
    for session in range(0,sessionsNum):
        cocaineSeeking        (  session , '12'       )
        homeCage              (  session , '12'       ) 
        


plotting()

