import numpy as np
import matplotlib.pyplot as plt
import pylab
from scipy.optimize import curve_fit
from scipy import stats


font = {'family' : 'normal', 'size'   : 16}
pylab.rc('font', **font)
pylab.rcParams.update({'legend.fontsize': 16})

fig1 = pylab.figure( figsize=(5,3.5) )
fig1.subplots_adjust(left=0.16)
fig1.subplots_adjust(bottom=0.2)
ax1 = fig1.add_subplot(111)

x=[4,3,4,5,4,3,4,4,4,3,3,4,3,4,2,2,5]
y=[0.023475,0.015625, 0.023475,0.03125,0.023475,0.015625,0.023475,0.023475,0.023475,0.015625,0.015625,0.023475,0.015625,0.023475,0.0078125,0.0078125,0.03125]
z=[149.7520842,169.1389284,90.44927315,71.16265529,154.4030423,158.6997488,65.34962939,123.6612236,116.2418556,138.5880358,173.9780417,81.60427075,83.13657738,113.7201696,124.5055583,251.7408322,78.08190817]

pylab.xlim((0.004,0.035))
pylab.ylim((0,300))

tick_lcs = [0.0078125,0.015625,0.023475,0.03125]
tick_lbs = [7.8,15.6,23.475,31.25]
pylab.xticks(tick_lcs, tick_lbs)

for line in ax1.get_xticklines() + ax1.get_yticklines():
    line.set_markeredgewidth(2)
    line.set_markersize(5)

area = np.pi * (0.1 * 40)**2 # 0 to 15 point radiuses

ax1.scatter(y, z, area, marker='d',alpha=1 , color = 'black' ,linewidth = 2 )


#------------------------------------------------------------------------------
#--------------------------   Linear regression line   ------------------------
#------------------------------------------------------------------------------

slope, intercept, r_value, p_value, std_err = stats.linregress(y[0:16], z[0:16])

print 'slope              = ', slope
print 'intercept          = ', intercept
print 'r value            = ', r_value
print  'p_value           = ', p_value
print 'standard deviation = ', std_err

line = np.zeros ( [4] , float )

for i in range(0,4):
    line[i] = slope*tick_lcs[i] + intercept
print line
print tick_lcs
S2 = ax1.plot(tick_lcs,line , '-', ms=8, markeredgewidth =2, alpha=1, mfc='black',linewidth = 2 , color='blue' )
#------------------------------------------------------------------------------

ax1.set_ylabel('Maximum infusion rate')
ax1.set_xlabel('Critical dose (  g/infusion)')
fig1.savefig('Dose-ResponseRate-Vulnerability.eps', format='eps')

pylab.show()

