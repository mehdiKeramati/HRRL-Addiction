import numpy as np
import matplotlib.pyplot as plt
import pylab

font = {'family' : 'normal', 'size'   : 16}
pylab.rc('font', **font)
pylab.rcParams.update({'legend.fontsize': 16})

fig1 = pylab.figure( figsize=(5,3.5) )
fig1.subplots_adjust(left=0.16)
fig1.subplots_adjust(bottom=0.2)
ax1 = fig1.add_subplot(111)


x = [10,11,11,12,13]        # The dose's number at which jump occurs
y = [0.098, 0.111, 0.111,0.126,0.126,0.147]        # The dose at which jump occurs
z = [32,28,24,20,16,12]        # Maximum of response rate for each level of D2R (i.e. setpoint)
s = [200,180,160,140,120,100]   # Setpoint level


scale = 0.1

area = np.zeros([6],float)

area [0] = np.pi * (scale * 200)**2 # 0 to 15 point radiuses
area [1] = np.pi * (scale * 180)**2 # 0 to 15 point radiuses
area [2] = np.pi * (scale * 160)**2 # 0 to 15 point radiuses
area [3] = np.pi * (scale * 140)**2 # 0 to 15 point radiuses
area [4] = np.pi * (scale * 120)**2 # 0 to 15 point radiuses
area [5] = np.pi * (scale * 100)**2 # 0 to 15 point radiuses

pylab.xlim((0.09,0.150))
pylab.ylim((8,38))

tick_lcs = [0.098,0.111,0.126,0.126,0.147]
tick_lbs = [0.098,0.111,0.126,0.126,0.147]
pylab.xticks(tick_lcs, tick_lbs)

colori = '0.60' 
ax1.axvline(0.098, ymin=0, ymax=1, color=colori ,ls='--', lw=2 )
ax1.axvline(0.111, ymin=0, ymax=1, color=colori ,ls='--', lw=2 )
ax1.axvline(0.126, ymin=0, ymax=1, color=colori ,ls='--', lw=2 )
ax1.axvline(0.147, ymin=0, ymax=1, color=colori ,ls='--', lw=2 )

for line in ax1.get_xticklines() + ax1.get_yticklines():
    line.set_markeredgewidth(2)
    line.set_markersize(5)

ax1.scatter(y, z, s=area, alpha=0.5,linewidth = 2 )


ax1.set_ylabel('Maximum infusion rate')
ax1.set_xlabel('Critical dose (mg/infusion)')
fig1.savefig('Dose-ResponseRate-Vulnerability.eps', format='eps')


pylab.show()