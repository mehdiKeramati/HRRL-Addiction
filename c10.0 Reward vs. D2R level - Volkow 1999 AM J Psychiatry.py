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



'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Homeostatically-regulated Reward   ------------------------------------------------------------ 
--------------------------------------------------------------------------------------------------------------------------------'''
def driveReductionReward(inState,setpointS,outcome):
    d1 = numpy.power(numpy.absolute(numpy.power(setpointS-inState,n*1.0)),(1.0/m))
    d2 = numpy.power(numpy.absolute(numpy.power(setpointS-inState-outcome,n*1.0)),(1.0/m))
    return d1-d2



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
initialSetpoint         = 125

#------------ Drive Function
m                       = 3     # Parameter of the drive function : m-th root
n                       = 4     # Parameter of the drive function : n-th pawer

beta                    = 10     # Rate of exploration



setpointsNum = 11
RewardPerSetpoint               = numpy.zeros([setpointsNum],float)
CocaineProbabilityPerSetpoint   = numpy.zeros([setpointsNum],float)
SetpoinPerSetpoint              = numpy.zeros([setpointsNum],float)

for i in range(0,setpointsNum):

    initialSetpoint = 100 + 10*i
    HomeoRew = driveReductionReward(0,initialSetpoint,cocaine)
    
    print "Setpoint num: %d / %d     " %(i+1,setpointsNum)

    RewardPerSetpoint[i] = HomeoRew

    sumEV = abs(cmath.exp( HomeoRew / beta )) + abs(cmath.exp( 200 / beta ))
    probab = abs(cmath.exp( HomeoRew / beta )) / sumEV
    CocaineProbabilityPerSetpoint[i] = probab

    SetpoinPerSetpoint[i] = initialSetpoint


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


t = numpy.zeros([setpointsNum],float)

for i in range(0,setpointsNum):
    t[i]         = 1000.0/(100.0 + 10.0*i)

S1 = ax1.plot(t,RewardPerSetpoint , 'o', ms=8, markeredgewidth =1.5, alpha=1, mfc='black',linewidth = 2 , color='red' )



pylab.yticks(pylab.arange(160, 241, 20))
pylab.ylim((155,245))
#
pylab.xticks(pylab.arange(5, 11, 1))
pylab.xlim((4.5,10.8))



for line in ax1.get_xticklines() + ax1.get_yticklines():
    line.set_markeredgewidth(2)
    line.set_markersize(5)

ax1.set_ylabel('Cocaine rewarding value')
ax1.set_xlabel('D2R Level (=setpoint$^{-1}$x1000)')
#ax1.set_title('Post-escelation')
fig1.savefig('CocaineReward-D2Level.eps', format='eps')


'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plotting 2  ------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''    
 
font = {'family' : 'normal', 'size'   : 16}
pylab.rc('font', **font)
pylab.rcParams.update({'legend.fontsize': 16})

fig1 = pylab.figure( figsize=(5,3.5) )
fig1.subplots_adjust(left=0.16)
fig1.subplots_adjust(bottom=0.2)
ax1 = fig1.add_subplot(111)


t = numpy.zeros([setpointsNum],float)

for i in range(0,setpointsNum):
    t[i]         = 1000.0/(100.0 + 10.0*i)

S1 = ax1.plot(t,CocaineProbabilityPerSetpoint , 'o', ms=8, markeredgewidth =1.5, alpha=1, mfc='black',linewidth = 2 , color='red' )



pylab.yticks(pylab.arange(0, 1.001, 0.2))
pylab.ylim((-0.1,1.1))

pylab.xticks(pylab.arange(5, 11, 1))
pylab.xlim((4.5,10.8))



for line in ax1.get_xticklines() + ax1.get_yticklines():
    line.set_markeredgewidth(2)
    line.set_markersize(5)

ax1.set_ylabel('Cocaine-choice probability')
ax1.set_xlabel('D2R Level (=setpoint$^{-1}$x1000)')
#ax1.set_title('Post-escelation')
fig1.savefig('CocaineChoiceProbability.eps', format='eps')



'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plotting 3  ------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''    
 
font = {'family' : 'normal', 'size'   : 16}
pylab.rc('font', **font)
pylab.rcParams.update({'legend.fontsize': 16})



fig1 = pylab.figure( figsize=(5,3.5) )
fig1.subplots_adjust(left=0.16)
fig1.subplots_adjust(bottom=0.2)
ax1 = fig1.add_subplot(111)

ax1.axhline(100,  color='0.25',ls='--', lw=1 )
ax1.axhline(200, color='0.25',ls='--', lw=1 )

t = numpy.zeros([setpointsNum],float)

for i in range(0,setpointsNum):
    t[i]         = 1000.0/(100.0 + 10.0*i)

S1 = ax1.plot(t,SetpoinPerSetpoint , 'o', ms=8, markeredgewidth =1.5, alpha=1, mfc='black',linewidth = 2 , color='red' )



pylab.yticks(pylab.arange(0, 201, 50))
pylab.ylim((-5,230))

pylab.xticks(pylab.arange(5, 11, 1))
pylab.xlim((4.5,10.8))



for line in ax1.get_xticklines() + ax1.get_yticklines():
    line.set_markeredgewidth(2)
    line.set_markersize(5)

ax1.set_ylabel('Homeostatic setpoint')
ax1.set_xlabel('D2R Level (=setpoint$^{-1}$x1000)')
#ax1.set_title('Post-escelation')
fig1.savefig('Setpoint.eps', format='eps')





pylab.show()
