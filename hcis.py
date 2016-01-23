import matplotlib.pyplot as plt
plt.style.use('ggplot')
from scipy.io import arff
import pandas as pd
from scipy import stats
import itertools
import random
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics


def knn(df,k):
	nbrs = NearestNeighbors(n_neighbors=3)
	nbrs.fit(df)
	distances, indices = nbrs.kneighbors(df)
	return distances, indices

def reachDist(df,MinPts,knnDist):
	nbrs = NearestNeighbors(n_neighbors=MinPts)
	nbrs.fit(df)
	distancesMinPts, indicesMinPts = nbrs.kneighbors(df)
	distancesMinPts[:,0] = np.amax(distancesMinPts,axis=1)
	distancesMinPts[:,1] = np.amax(distancesMinPts,axis=1)
	distancesMinPts[:,2] = np.amax(distancesMinPts,axis=1)
	return distancesMinPts, indicesMinPts

def ird(MinPts,knnDistMinPts):
	return (MinPts/np.sum(knnDistMinPts,axis=1))

def lof(Ird,MinPts,dsts):
	lof=[]
	for item in dsts:
		tempIrd = np.divide(Ird[item[1:]],Ird[item[0]])
		lof.append(tempIrd.sum()/MinPts)
	return lof

def make_ticklabels_invisible(fig):
    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_visible(False)


data, meta = arff.loadarff('ann_thyroid.arff')
df = pd.DataFrame(data)

plt.figure(0)
ax1 = plt.subplot2grid((3,3), (0,0), colspan=2)
ax2 = plt.subplot2grid((3,3), (1,0), colspan=2, rowspan=2,sharex=ax1)
ax3 = plt.subplot2grid((3,3), (1, 2), rowspan=2,sharey=ax2)


plt.suptitle("High Contrast Subspace: var_0002 & var_0003")
#make_ticklabels_invisible(plt.gcf())

ax2.scatter(df['var_0003'],df['var_0002'])
df['var_0003'].hist(color='k', alpha=0.5, bins=30,ax=ax1,normed = True)
df['var_0002'].hist(color='k', alpha=0.5, bins=30,orientation="horizontal")

var0002_index = df['var_0002'].rank()/max(df['var_0002'].rank())
df[(var0002_index>0.65)]['var_0003'].hist(color='r', alpha=0.2, bins=20,ax=ax1,normed = True,linestyle='dotted')
ax1.text(0.8,5,'p-value:<0.0001***',va="center", ha="center")
plt.show()

t=stats.ttest_ind(df['var_0003'].values, df[(var0002_index>0.65)]['var_0003'].values, equal_var = False)
print t.pvalue

k = stats.ks_2samp(df['var_0003'].values, df[(var0002_index>0.65)]['var_0003'].values)
print "k-test"
print k.pvalue

plt.figure(1)
ax1 = plt.subplot2grid((3,3), (0,0), colspan=2)
ax2 = plt.subplot2grid((3,3), (1,0), colspan=2, rowspan=2,sharex=ax1)
ax3 = plt.subplot2grid((3,3), (1, 2), rowspan=2,sharey=ax2)


plt.suptitle("Low Contrast Subspace: var_0000 & var_0003")
#make_ticklabels_invisible(plt.gcf())



ax2.scatter(df['var_0000'],df['var_0003'])
df['var_0000'].hist(color='k', alpha=0.5, bins=30,ax=ax1,normed = True)
df['var_0003'].hist(color='k', alpha=0.5, bins=30,orientation="horizontal")

var0003_index = df['var_0003'].rank()/max(df['var_0003'].rank())
df[(var0003_index>0.65)]['var_0000'].hist(color='r', alpha=0.2, bins=30,ax=ax1,normed = True,linestyle='dotted')
ax1.text(0.8,5,'p-value:<0.35',va="center", ha="center")
plt.show()

t=stats.ttest_ind(df['var_0000'].values, df[(var0003_index>0.65)]['var_0000'].values, equal_var = False)
#print t.pvalue

k=stats.ks_2samp(df['var_0000'].values, df[(var0003_index>0.65)]['var_0000'].values)
#print "k-test"
#print k.pvalue

#calculate the index
index_df = (df.rank()/df.rank().max()).iloc[:,:-1]

def comboGenerator(startPoint,space,n):
	combosFinal=[]
	for item in itertools.combinations(list(set(space)-set(startPoint)),(n-len(startPoint))):
		combosFinal.append(sorted(startPoint+list(item)))
	return combosFinal

listOfCombos = comboGenerator([],df.columns[:-1],2)
testedCombos=[]
selection=[]
while(len(listOfCombos)>0):
	if listOfCombos[0] not in testedCombos:
		#print "Calculating {0}".format(listOfCombos[0])
		alpha1 = pow(0.2,(float(1)/float(len(listOfCombos[0]))))
		pvalue_Total =0
		pvalue_cnt = 0
		avg_pvalue=0
		for i in range(0,50):
			lband = random.random()
			uband = lband+alpha1
			v = random.randint(0,(len(listOfCombos[0])-1))
			rest = list(set(listOfCombos[0])-set([listOfCombos[0][v]]))
			k=stats.ks_2samp(df[listOfCombos[0][v]].values, df[((index_df[rest]<uband) & (index_df[rest]>lband)).all(axis=1)][listOfCombos[0][v]].values)
			#print "iter:{4},lband:{0},uband:{1},v:{2},pvalue:{3},length:{5},rest:{6}".format(lband,uband,v,k.pvalue,i,len(df[((index_df[rest]<uband) & (index_df[rest]>lband)).all(axis=1)][listOfCombos[0][v]]),rest)
			if not(np.isnan(k.pvalue)):
				pvalue_Total = pvalue_Total+k.pvalue
				pvalue_cnt = pvalue_cnt+1
		if pvalue_cnt>0:
			avg_pvalue = pvalue_Total/pvalue_cnt
			#print avg_pvalue
		if (1.0-avg_pvalue)>0.75:
			selection.append(listOfCombos[0])
			listOfCombos = listOfCombos + comboGenerator(listOfCombos[0],df.columns[:-1],(len(listOfCombos[0])+1))
		testedCombos.append(listOfCombos[0])
		listOfCombos.pop(0)
		listOfCombos = [list(t) for t in set(map(tuple,listOfCombos))]
	else:
		listOfCombos.pop(0)

scoresList=[]
for item in selection:
	m=50
	knndist, knnindices = knn(df[item],3)
	reachdist, reachindices = reachDist(df[item],m,knndist)
	irdMatrix = ird(m,reachdist)
	lofScores = lof(irdMatrix,m,reachindices)
	scoresList.append(lofScores)


avgs = np.nanmean(np.ma.masked_invalid(np.array(scoresList)),axis=0)

scaled_avgs = MinMaxScaler().fit_transform(avgs.reshape(-1,1))

print "HCiS AUC Score"
print metrics.roc_auc_score(pd.to_numeric(df['class'].values),scaled_avgs)

m=50
knndist, knnindices = knn(df.iloc[:,:-1],3)
reachdist, reachindices = reachDist(df.iloc[:,:-1],m,knndist)
irdMatrix = ird(m,reachdist)
lofScores = lof(irdMatrix,m,reachindices)
ss=MinMaxScaler().fit_transform(np.array(lofScores).reshape(-1,1))
print "LOF AUC Score"
print metrics.roc_auc_score(pd.to_numeric(df['class'].values),ss)

