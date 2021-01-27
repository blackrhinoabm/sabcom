import re, os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from __future__ import division


sim_pop = 100000
pop =  3740000
folder ='secondwave_nolearning_3iter'
empirical_fatalities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5668344944441515, 0.5668344944441515, 0.5668344944441515, 0.5668344944441515, 0.5668344944441515, 0.5668344944441515, 0.5668344944441515, 5.790672466505531, 5.790672466505531, 5.790672466505531, 5.790672466505531, 5.790672466505531, 5.790672466505531, 5.790672466505531, 11.401584695227209, 11.401584695227209, 11.401584695227209, 11.401584695227209, 11.401584695227209, 11.401584695227209, 11.401584695227209, 10.928548917227099, 10.928548917227099, 10.928548917227099, 10.928548917227099, 10.928548917227099, 10.928548917227099, 10.928548917227099, 11.017986409272664, 11.017986409272664, 11.017986409272664, 11.017986409272664, 11.017986409272664, 11.017986409272664, 11.017986409272664, 28.481690232374223, 28.481690232374223, 28.481690232374223, 28.481690232374223, 28.481690232374223, 28.481690232374223, 28.481690232374223, 35.55811813997199, 35.55811813997199, 35.55811813997199, 35.55811813997199, 35.55811813997199, 35.55811813997199, 35.55811813997199]
empirical_cumulative=pd.Series(empirical_fatalities).cumsum()
empirical_daily=pd.Series(empirical_fatalities)
sim_fatality_curves = {}
d="/Users/admin/Dropbox/SABCoM_data/other/".format(folder)
for subdir in os.walk(d):
    dirs=subdir[1]
    break
print(dirs)
# fig, axs = plt.subplots(2, 2)
count=0
for f in dirs:
    if f==folder :
        path="{}{}/".format(d,f)
        print(path)
        for subdir, dir, files in os.walk(path):
            arr = np.zeros((len(pd.read_csv(path+files[1]))))
        for x in range(50):
            try:
                sim_dead_curve = pd.DataFrame(pd.read_csv(path+'seed{}quantities_state_time.csv'.format(x))['d'] * (pop / sim_pop))
            #     print(sim_dead_curve)
                sim_dead_curve = sim_dead_curve.diff().ewm(span=10).mean()
                sim_fatality_curves['simulation ' + str(x)]= list(sim_dead_curve['d'])
            except:
                continue
        simulations = pd.DataFrame(sim_fatality_curves)
        arr = np.zeros((len(pd.read_csv(path+files[2]))))
        for i,v in enumerate(files):
            try:
                data=pd.read_csv(path+files[i])
    #             plt.plot(data.e+data.i1+data.i2+data.c)
    #             plt.plot(data.d)
                c=data.d.diff().ewm(span=5).mean().to_numpy()
                arr=np.vstack([arr,c])
            except:
                continue
        arr2=np.delete(arr,0,axis=0)
        a = np.matrix( arr2.T)
        b=pd.DataFrame(a)
        b['day']=b.index
        c=pd.melt(b , id_vars=['day'],  var_name='seed', value_name='deaths')
        b['deaths_mean']=b[b.columns[:-1]].mean(axis=1)
        b['deaths_scaled']=b['deaths_mean']/float((sim_pop/pop))
        c['deaths_scaled']=c['deaths']/float((sim_pop/pop))
        if f == folder :
            s=sns.lineplot(data=c[c.day>1], x='day', y="deaths_scaled",label="s_wave-priweight1",)
        plt.title( '  ') #str(round(pop/1000000,2))+
        plt.legend()
        empirical_daily.plot(style='r-', label='empirical')