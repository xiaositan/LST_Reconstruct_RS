# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 15:38:56 2020

@author: Administrator
"""

# ---------------------------------------------------------------------------
import os
import pandas as pd
import matplotlib.pyplot as plt
####==============================================================================
####=================== Process   MOD11B1  =======================================
####====================  Set local variables  ===================================
df_ndvi = pd.DataFrame()
ndvi_file = os.path.join(os.path.abspath(u'../../data/processed_data/result/ndvi.2013-2018/'),'NDVI-DMZ.csv')  
df = pd.read_csv(ndvi_file,header=0)
df_ndvi['DMZ'] = df['NDVI']
ndvi_file = os.path.join(os.path.abspath(u'../../data/processed_data/result/ndvi.2013-2018/'),'NDVI-SDZ.csv')  
df = pd.read_csv(ndvi_file,header=0)
df_ndvi['SDZ'] = df['NDVI']
ndvi_file = os.path.join(os.path.abspath(u'../../data/processed_data/result/ndvi.2013-2018/'),'NDVI-YKZ.csv')  
df = pd.read_csv(ndvi_file,header=0)
df_ndvi['YKZ'] = df['NDVI']
ndvi_file = os.path.join(os.path.abspath(u'../../data/processed_data/result/ndvi.2013-2018/'),'NDVI-ARC.csv')  
df = pd.read_csv(ndvi_file,header=0)
df_ndvi['ARC'] = df['NDVI']
ndvi_file = os.path.join(os.path.abspath(u'../../data/processed_data/result/ndvi.2013-2018/'),'NDVI-DSL.csv')  
df = pd.read_csv(ndvi_file,header=0)
df_ndvi['DSL'] = df['NDVI']
ndvi_file = os.path.join(os.path.abspath(u'../../data/processed_data/result/ndvi.2013-2018/'),'NDVI-HZZ.csv')  
df = pd.read_csv(ndvi_file,header=0)
df_ndvi['HZZ'] = df['NDVI']
ndvi_file = os.path.join(os.path.abspath(u'../../data/processed_data/result/ndvi.2013-2018/'),'NDVI-HMZ.csv')  
df = pd.read_csv(ndvi_file,header=0)
df_ndvi['HMZ'] = df['NDVI']
ndvi_file = os.path.join(os.path.abspath(u'../../data/processed_data/result/ndvi.2013-2018/'),'NDVI-JCHM.csv')  
df = pd.read_csv(ndvi_file,header=0)
df_ndvi['JCHM'] = df['NDVI']
ndvi_file = os.path.join(os.path.abspath(u'../../data/processed_data/result/ndvi.2013-2018/'),'NDVI-SDQ.csv')  
df = pd.read_csv(ndvi_file,header=0)
df_ndvi['SDQ'] = df['NDVI']
ndvi_file = os.path.join(os.path.abspath(u'../../data/processed_data/result/ndvi.2013-2018/'),'NDVI-HHL.csv')  
df = pd.read_csv(ndvi_file,header=0)
df_ndvi['HHL'] = df['NDVI']
df_ndvi['Month'] = list(range(1,13))



font = {'family' : 'Palatino Linotype',
    'weight' : 'normal',
    'size'   : 18}    
plt.rc('font', **font)  

y_min = 0
y_max = 1.0


####---------------------------------------------------------------------------
#### 10 site in corrections process
text_pos_x = 0.5
text_pos_y = 0.8
fig = plt.figure(figsize=[19,5])
ax = fig.add_subplot(141,ylim=(y_min,y_max))
plt.plot(df_ndvi['Month'],df_ndvi['ARC'],label='ARC')
plt.plot(df_ndvi['Month'],df_ndvi['SDZ'],label='SDZ') 
plt.plot(df_ndvi['Month'],df_ndvi['DMZ'],label='DMZ')
   
plt.legend(loc='upper center',fontsize=11,ncol=5)
ax.set_xlim(0,13) 
ax.set_xticks(range(1,13))
ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
ax.set_xlabel('Month',fontsize=18)
ax.set_ylabel('NDVI',color='black',fontsize=18)
ax.xaxis.set_ticks_position("bottom")
ax.yaxis.set_ticks_position("left")
ax.text(text_pos_x,text_pos_y,'(i)', fontsize=18, ha='left',va='bottom', wrap=True)
#ax.set_title('Arou Soil pf 2016')
for label in ax.get_xticklabels():  
    label.set_color('black') 
    label.set_fontsize(18) 
for label in ax.get_yticklabels(): 
    label.set_color('black')
    label.set_fontsize(18) 
    
ax = fig.add_subplot(142,ylim=(y_min,y_max))
plt.plot(df_ndvi['Month'],df_ndvi['DSL'],label='DSL')
plt.plot(df_ndvi['Month'],df_ndvi['YKZ'],label='YKZ')

plt.legend(loc='upper center',fontsize=11,ncol=5)
ax.set_xticks(range(1,13))
ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
ax.set_xlabel('Month',fontsize=18)
#ax.set_ylabel('NDVI',color='black',fontsize=18)
ax.xaxis.set_ticks_position("bottom")
ax.yaxis.set_ticks_position("left")
#ax.set_title('Arou Soil pf 2016')
ax.text(text_pos_x,text_pos_y,'(j)', fontsize=18, ha='left',va='bottom', wrap=True)
for label in ax.get_xticklabels():  
    label.set_color('black') 
    label.set_fontsize(18) 
for label in ax.get_yticklabels(): 
    label.set_color('black')
    label.set_fontsize(18) 
    
ax = fig.add_subplot(143,ylim=(y_min,y_max))
plt.plot(df_ndvi['Month'],df_ndvi['SDQ'],label='SDQ')
plt.plot(df_ndvi['Month'],df_ndvi['HHL'],label='HHL')

plt.legend(loc='upper center',fontsize=11,ncol=5)
ax.set_xticks(range(1,13))
ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
ax.set_xlabel('Month',fontsize=18)
#ax.set_ylabel('NDVI',color='black',fontsize=18)
ax.xaxis.set_ticks_position("bottom")
ax.yaxis.set_ticks_position("left")
#ax.set_title('Arou Soil pf 2016')
ax.text(text_pos_x,text_pos_y,'(k)', fontsize=18, ha='left',va='bottom', wrap=True)
for label in ax.get_xticklabels():  
    label.set_color('black') 
    label.set_fontsize(18) 
for label in ax.get_yticklabels(): 
    label.set_color('black')
    label.set_fontsize(18) 
    
ax = fig.add_subplot(144,ylim=(y_min,y_max))
plt.plot(df_ndvi['Month'],df_ndvi['HZZ'],label='HZZ')
plt.plot(df_ndvi['Month'],df_ndvi['JCHM'],label='JCHM')
plt.plot(df_ndvi['Month'],df_ndvi['HMZ'],label='HMZ')
plt.legend(loc='upper center',fontsize=11,ncol=5)
ax.set_xticks(range(1,13))
ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
ax.set_xlabel('Month',fontsize=18)
#ax.set_ylabel('NDVI',color='black',fontsize=18)
ax.xaxis.set_ticks_position("bottom")
ax.yaxis.set_ticks_position("left")
#ax.set_title('Arou Soil pf 2016')
ax.text(text_pos_x,text_pos_y,'(l)', fontsize=18, ha='left',va='bottom', wrap=True)
for label in ax.get_xticklabels():  
    label.set_color('black') 
    label.set_fontsize(18) 
for label in ax.get_yticklabels(): 
    label.set_color('black')
    label.set_fontsize(18)     
#outputjpg = u'Reconstruct.RMSE.Month.DMZ.SDZ.SDQ.HHL.ARC.jpg'
output_fig = u'Figure 6. NDVI.Month.2013-2018.4.types.tif'
output_fig = os.path.join(os.path.abspath(u'../../output_figures/'),output_fig)
fig.savefig(output_fig,dpi=300,bbox_inches='tight')  
plt.show()

# ####---------------------------------------------------------------------------
# #### 4 individual sites 
# ndvi_file = os.path.join(os.path.abspath(u'../../validation/result/ndvi.2013-2018/other'),'NDVI-JYL.csv')  
# df = pd.read_csv(ndvi_file,header=0)
# df_ndvi['JYL'] = df['NDVI']
# ndvi_file = os.path.join(os.path.abspath(u'../../validation/result/ndvi.2013-2018/other'),'NDVI-EBZ.csv')  
# df = pd.read_csv(ndvi_file,header=0)
# df_ndvi['EBZ'] = df['NDVI']
# ndvi_file = os.path.join(os.path.abspath(u'../../validation/result/ndvi.2013-2018/other'),'NDVI-HCG.csv')  
# df = pd.read_csv(ndvi_file,header=0)
# df_ndvi['HCG'] = df['NDVI']
# ndvi_file = os.path.join(os.path.abspath(u'../../validation/result/ndvi.2013-2018/other'),'NDVI-HYZ.csv')  
# df = pd.read_csv(ndvi_file,header=0)
# df_ndvi['HYZ'] = df['NDVI']
# ndvi_file = os.path.join(os.path.abspath(u'../../validation/result/ndvi.2013-2018/other'),'NDVI-SSW.csv')  
# df = pd.read_csv(ndvi_file,header=0)
# df_ndvi['SSW'] = df['NDVI']
# df_ndvi['Month'] = list(range(1,13))

# text_pos_x = 0.5
# text_pos_y = 0.8
# fig = plt.figure(figsize=[19,5])
# ax = fig.add_subplot(141,ylim=(y_min,y_max))
# plt.plot(df_ndvi['Month'],df_ndvi['HCG'],label='HCG')
# plt.plot(df_ndvi['Month'],df_ndvi['JYL'],label='JYL')
# plt.legend(loc='upper center',fontsize=11,ncol=5)
# ax.set_xlim(0,13) 
# ax.set_xticks(range(1,13))
# ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
# ax.set_xlabel('Month',fontsize=18)
# ax.set_ylabel('NDVI',color='black',fontsize=18)
# ax.xaxis.set_ticks_position("bottom")
# ax.yaxis.set_ticks_position("left")
# ax.text(text_pos_x,text_pos_y,'(i)', fontsize=18, ha='left',va='bottom', wrap=True)
# #ax.set_title('Arou Soil pf 2016')
# for label in ax.get_xticklabels():  
#     label.set_color('black') 
#     label.set_fontsize(18) 
# for label in ax.get_yticklabels(): 
#     label.set_color('black')
#     label.set_fontsize(18) 
    
# ax = fig.add_subplot(142,ylim=(y_min,y_max))
# plt.plot(df_ndvi['Month'],df_ndvi['EBZ'],label='EBZ')

# plt.legend(loc='upper center',fontsize=11,ncol=5)
# ax.set_xticks(range(1,13))
# ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
# ax.set_xlabel('Month',fontsize=18)
# #ax.set_ylabel('NDVI',color='black',fontsize=18)
# ax.xaxis.set_ticks_position("bottom")
# ax.yaxis.set_ticks_position("left")
# #ax.set_title('Arou Soil pf 2016')
# ax.text(text_pos_x,text_pos_y,'(j)', fontsize=18, ha='left',va='bottom', wrap=True)
# for label in ax.get_xticklabels():  
#     label.set_color('black') 
#     label.set_fontsize(18) 
# for label in ax.get_yticklabels(): 
#     label.set_color('black')
#     label.set_fontsize(18) 
    
# ax = fig.add_subplot(143,ylim=(y_min,y_max))
# plt.plot(df_ndvi['Month'],df_ndvi['HYZ'],label='HYZ')

# plt.legend(loc='upper center',fontsize=11,ncol=5)
# ax.set_xticks(range(1,13))
# ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
# ax.set_xlabel('Month',fontsize=18)
# #ax.set_ylabel('NDVI',color='black',fontsize=18)
# ax.xaxis.set_ticks_position("bottom")
# ax.yaxis.set_ticks_position("left")
# #ax.set_title('Arou Soil pf 2016')
# ax.text(text_pos_x,text_pos_y,'(k)', fontsize=18, ha='left',va='bottom', wrap=True)
# for label in ax.get_xticklabels():  
#     label.set_color('black') 
#     label.set_fontsize(18) 
# for label in ax.get_yticklabels(): 
#     label.set_color('black')
#     label.set_fontsize(18) 
    
# ax = fig.add_subplot(144,ylim=(y_min,y_max))
# plt.plot(df_ndvi['Month'],df_ndvi['SSW'],label='SSW')
# plt.legend(loc='upper center',fontsize=11,ncol=5)
# ax.set_xticks(range(1,13))
# ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
# ax.set_xlabel('Month',fontsize=18)
# #ax.set_ylabel('NDVI',color='black',fontsize=18)
# ax.xaxis.set_ticks_position("bottom")
# ax.yaxis.set_ticks_position("left")
# #ax.set_title('Arou Soil pf 2016')
# ax.text(text_pos_x,text_pos_y,'(l)', fontsize=18, ha='left',va='bottom', wrap=True)
# for label in ax.get_xticklabels():  
#     label.set_color('black') 
#     label.set_fontsize(18) 
# for label in ax.get_yticklabels(): 
#     label.set_color('black')
#     label.set_fontsize(18)     
# #outputjpg = u'Reconstruct.RMSE.Month.DMZ.SDZ.SDQ.HHL.ARC.jpg'
# outputjpg = u'NDVI.Month.2013-2018.4.individual.Sites.jpg'
# outputjpg = os.path.join(os.path.abspath(u'../../validation/result/ndvi.2013-2018/'),outputjpg)
# fig.savefig(outputjpg,dpi=600,bbox_inches='tight')  
# plt.show()