import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def CCDF(data):
    '''
    Computes CCDF given data
    '''
    n = len(data)
    x = np.sort(data) # sort data\
    y = np.arange(1, n + 1) / n # compute cdf
    y = 1-y
    plt.plot(x, y)
    plt.grid(True)
    plt.xlabel('number of tweets')
    plt.ylabel('probability')
    plt.show()
a) Preliminaries

In [ ]:
data2 = pd.read_csv("tweet-time-series2.csv", header=None, sep=" ")
In [ ]:
data2 = data2.iloc[60000:70000, 1]
In [ ]:
data2
Out[ ]:
60000    33
60001    37
60002    27
60003    37
60004    28
         ..
69995     2
69996     1
69997     2
69998     1
69999     2
Name: 1, Length: 10000, dtype: int64
b) Selecting a model

Normal Distribution

In [ ]:
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns


distribN = np.random.normal(0, 1, 10000)
sns.distplot(distribN, hist = False, kde = True, kde_kws = {'linewidth': 3})
/usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `kdeplot` (an axes-level function for kernel density plots).
  warnings.warn(msg, FutureWarning)
Out[ ]:
<matplotlib.axes._subplots.AxesSubplot at 0x7f1e00fcdc90>

Weibull

In [ ]:
distribW = np.random.weibull(2, 10000)
sns.distplot(distribW, hist = False, kde = True, kde_kws = {'linewidth': 3})
/usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `kdeplot` (an axes-level function for kernel density plots).
  warnings.warn(msg, FutureWarning)
Out[ ]:
<matplotlib.axes._subplots.AxesSubplot at 0x7f1dfe45b050>

In [ ]:
import numpy as np
import seaborn as sns

DE = np.random.exponential(0.5, 10000)
sns.distplot(DE, hist = False, kde = True, kde_kws = {'linewidth': 3})
/usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `kdeplot` (an axes-level function for kernel density plots).
  warnings.warn(msg, FutureWarning)
Out[ ]:
<matplotlib.axes._subplots.AxesSubplot at 0x7fc01f5ed5d0>

Pareto

In [ ]:
import seaborn as sns


distribP = np.random.pareto(1.5, 10000)
sns.distplot(distribP, hist = False, kde = True, kde_kws = {'linewidth': 3})
/usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `kdeplot` (an axes-level function for kernel density plots).
  warnings.warn(msg, FutureWarning)
Out[ ]:
<matplotlib.axes._subplots.AxesSubplot at 0x7fc0236fe290>

CCDF Normal Distribution

In [ ]:
CCDF(distribN)

CCDF Weibull Distribution

In [ ]:
CCDF(distribW)

CCDF Pareto Pareto Distribution

In [ ]:
CCDF(distribP)

In [ ]:

b) Selecting a model When observing the CCDF this data, what can you say about the distribution and its tail? To help you, generate random numbers distributed according to Normal, Weibull, and Pareto distributions (see lecture slides to remember how to do this), and have a look at the CCDF for them.

Q3.1 - Model - MCQ: Choose among the following which ones seem possibly correct:


The tail is similar to a strict heavy-tailed distribution, i.e., it fits a straight line on the CCDF.
The tail is falling fast enough to be an exponential distribution.
The tail is falling fast enough to be either an exponential or an extreme value distribution.
Answer: I believe number 3 to be correct

The tail is falling fast enough to be either an exponential or an extreme value distribution.
As we can see the first drop followed by stagnation and then followed by another drop may suggest either one the Exponential or Extreme.
In [ ]:
CCDF(data2)

In [ ]:
import seaborn as sns


distribP = data2
sns.distplot(distribP, hist = False, kde = True, kde_kws = {'linewidth': 3})
/usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `kdeplot` (an axes-level function for kernel density plots).
  warnings.warn(msg, FutureWarning)
Out[ ]:
<matplotlib.axes._subplots.AxesSubplot at 0x7f1dfdfd4590>

Q3.2 - Model - Open: When observing the CCDF of the data (not the randomly generated numbers),
what model do you think is the most relevant?

Answer: I think that out of the selected models the normal distribution is the most relevant because the distribution itself due to the fist drop suggests to be from the exponential family however in Normal Distribution it can go into negative which is not observed here. Outside from the selection of models provided the bimodal poisson distribution could fit the data best due to the data showing a number of events over time and a possible two modes represented by the first down slope followed by the next.

c) Estimating distribution parameters Let’s assume for now that the distribution is not a heavy-tailed one, but either an extreme value or an exponential. This leaves with a lot of options, e.g., Weibull, Gamma, logistic, Normal, Poisson. Now, we need to estimate the parameters of the candidate distributions, i.e., step 2 of the methodology. Estimate the parameters of a few candidate distributions using the maximum likelihood estimation (MLE) method. Check the following distributions: Normal, Exponential, Gamma, and Lognorm. The first two are exponential distributions, while the last two are extreme value distributions.

MLE Normal Distribution

In [ ]:
import scipy 
dist = getattr(scipy.stats, 'norm')
paramN = dist.fit(data2)
In [ ]:
paramN
Out[ ]:
(21.69, 11.647896805861562)
MLE Exponential Distribution

In [ ]:
dist = getattr(scipy.stats, 'expon')
paramE = dist.fit(data2)
In [ ]:

Object `scipy.stats` not found.
In [ ]:
paramE
Out[ ]:
(1.0, 20.69)
MLE Gamma Distribution

In [ ]:
dist = getattr(scipy.stats, 'gamma')
paramG = dist.fit(data2)
In [ ]:
paramG
Out[ ]:
(276.8794890535214, -175.3586760889923, 0.7115100785350288)
MLE Lognorm

In [ ]:
dist = getattr(scipy.stats, 'lognorm')
paramL = dist.fit(data2)
In [ ]:
paramL
Out[ ]:
(0.00713012917387662, -1617.9885155458996, 1639.6527728478004)
Q3.3 - Estimating distribution parameters: Select which one(s) of the following assertions about the
value of the fitted parameters by MLE is (are) correct.

Correct:

The standard deviation fitted by MLE for the normal distribution is high, i.e., much higher than 1.
Q3.4 - Checking the fit based on the qqplot: Based on the visual inspection of the 4 different qqplots,
which one(s) of the following assertions is (are) correct:

The Normal distribution fits the data well.
The Exponential distribution fits the data well.
The Gamma distribution fits the data well.
The Lognorm distribution fits the data well.
None of the mentioned distributions fit the data well.
Answer:
the point '5. None of the mentioned distributions fit the data well.' is correct as shown above the data doesn't fit any of the models.

Normal Distribution

In [ ]:

In [ ]:
from statsmodels.api import qqplot
import scipy


qqplot(data2, dist=scipy.stats.norm, line='s')
/usr/local/lib/python3.7/dist-packages/statsmodels/tools/_testing.py:19: FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.
  import pandas.util.testing as tm
Out[ ]:


Exponential Distribution

In [ ]:
qqplot(data2, dist=scipy.stats.expon, line= 's')
Out[ ]:


Gamma Distribution

In [ ]:

In [ ]:
#Plotting Gamma Distribution
distribG = np.random.gamma(20, 1, 10000)
In [ ]:
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot_2samples
x = distribG
y = data2
pp_x = sm.ProbPlot(x)
pp_y = sm.ProbPlot(y)
qqplot_2samples(pp_x, pp_y, line='45')
plt.show()
print('Gamma Distribution')

Gamma Distribution
Lognorm Distrubution

In [ ]:
distribLN = np.random.lognormal(1, 1, 10000)
In [ ]:
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot_2samples
x = distribLN
y = data2
pp_x = sm.ProbPlot(x)
pp_y = sm.ProbPlot(y)
qqplot_2samples(pp_x, pp_y, line='45')
plt.show()
print('Lognorm Distribution')

Lognorm Distribution
