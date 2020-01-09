# Principal Component Analysis (PCA) for Wine

## Brief primer and history
Principal component analysis (PCA) is a statistical procedure that uses an [orthogonal transformation](https://en.wikipedia.org/wiki/Orthogonal_transformation) to convert a set of observations of possibly correlated variables into a set of values of [linearly uncorrelated](https://en.wikipedia.org/wiki/Correlation_and_dependence) variables called principal components. The number of distinct principal components is equal to the smaller of the number of original variables or the number of observations minus one. This transformation is defined in such a way that the first principal component has the largest possible [variance](https://en.wikipedia.org/wiki/Variance) (that is, accounts for as much of the variability in the data as possible), and each succeeding component in turn has the highest variance possible under the constraint that it is [orthogonal](https://en.wikipedia.org/wiki/Orthogonal) the preceding components. The resulting vectors are an uncorrelated [orthogonal basis set](https://en.wikipedia.org/wiki/Orthogonal_basis_set). 

PCA is sensitive to the relative scaling of the original variables.

PCA was invented in 1901 by [Karl Pearson](https://en.wikipedia.org/wiki/Karl_Pearson) as an analogue of the principal axis theorem in mechanics; it was later independently developed and named by [Harold Hotelling](https://en.wikipedia.org/wiki/Harold_Hotelling) in the 1930s.

## Mathematical details
PCA is mathematically defined as an orthogonal linear transformation that transforms the data to a new coordinate system such that the greatest variance by some projection of the data comes to lie on the first coordinate (called the first principal component), the second greatest variance on the second coordinate, and so on.[3]

Consider a data matrix, $\mathbf{X}$, with column-wise zero empirical mean (the sample mean of each column has been shifted to zero), where each of the $n$ rows represents a different repetition of the experiment, and each of the $p$ columns gives a particular kind of feature (say, the results from a particular sensor).

Mathematically, the transformation is defined by a set of p-dimensional vectors of weights or loadings
${\displaystyle \mathbf {w} _{(k)}=(w_{1},\dots ,w_{p})_{(k)}} \mathbf {w} _{(k)}=(w_{1},\dots ,w_{p})_{(k)}$ that map each row vector ${\displaystyle \mathbf {x} _{(i)}} \mathbf{x}_{(i)}$ of $\mathbf{X}$ to a new vector of principal component scores ${\displaystyle \mathbf {t} _{(i)}=(t_{1},\dots ,t_{m})_{(i)}}$ given by

$${\displaystyle {t_{k}}_{(i)}=\mathbf {x} _{(i)}\cdot \mathbf {w} _{(k)}\qquad \mathrm {for} \qquad i=1,\dots ,n\qquad k=1,\dots ,m} {\displaystyle {t_{k}}_{(i)}=\mathbf {x} _{(i)}\cdot \mathbf {w} _{(k)}\qquad \mathrm {for} \qquad i=1,\dots ,n\qquad k=1,\dots ,m}$$

in such a way that the individual variables ${\displaystyle t_{1},\dots ,t_{m}}$ of t considered over the data set successively inherit the maximum possible variance from $\mathbf{x}$, with each loading vector $\mathbf{w}$ constrained to be a unit vector.

In order to maximize variance, the first loading vector $\mathbf {w} _{(1)}$ thus has to satisfy

$$ {\displaystyle \mathbf {w} _{(1)}={\underset {\Vert \mathbf {w} \Vert =1}{\operatorname {\arg \,max} }}\,\left\{\sum _{i}\left(t_{1}\right)_{(i)}^{2}\right\}={\underset {\Vert \mathbf {w} \Vert =1}{\operatorname {\arg \,max} }}\,\left\{\sum _{i}\left(\mathbf {x} _{(i)}\cdot \mathbf {w} \right)^{2}\right\}}$$

Equivalently, writing this in matrix form gives

$${\displaystyle \mathbf {w} _{(1)}={\underset {\Vert \mathbf {w} \Vert =1}{\operatorname {\arg \,max} }}\,\{\Vert \mathbf {Xw} \Vert ^{2}\}={\underset {\Vert \mathbf {w} \Vert =1}{\operatorname {\arg \,max} }}\,\left\{\mathbf {w} ^{T}\mathbf {X} ^{T}\mathbf {Xw} \right\}}$$

Since $\mathbf {w} _{(1)}$ has been defined to be a unit vector, it equivalently also satisfies
$${\displaystyle \mathbf {w} _{(1)}={\operatorname {\arg \,max} }\,\left\{{\frac {\mathbf {w} ^{T}\mathbf {X} ^{T}\mathbf {Xw} }{\mathbf {w} ^{T}\mathbf {w} }}\right\}}$$.

With $\mathbf {w} _{(1)}$ found, the first principal component of a data vector $\mathbf {x} _{(i)}$ can then be given as a score $\mathbf {t} _{(i)}$ = $\mathbf {x} _{(i)}$ ⋅ $\mathbf {w} _{(1)}$ in the transformed co-ordinates, or as the corresponding vector in the original variables, {$\mathbf {x} _{(i)}$ ⋅ $\mathbf {w} _{(1)}$} $\mathbf {w} _{(1)}$.

The $k^{th}$ component can be found by subtracting the first $k$ − 1 principal components from $\mathbf{X}$:

$${\displaystyle \mathbf {\hat {X}} _{k}=\mathbf {X} -\sum _{s=1}^{k-1}\mathbf {X} \mathbf {w} _{(s)}\mathbf {w} _{(s)}^{\rm {T}}}$$
and then finding the loading vector which extracts the maximum variance from this new data matrix

$${\displaystyle \mathbf {w} _{(k)}={\underset {\Vert \mathbf {w} \Vert =1}{\operatorname {arg\,max} }}\left\{\Vert \mathbf {\hat {X}} _{k}\mathbf {w} \Vert ^{2}\right\}={\operatorname {\arg \,max} }\,\left\{{\tfrac {\mathbf {w} ^{T}\mathbf {\hat {X}} _{k}^{T}\mathbf {\hat {X}} _{k}\mathbf {w} }{\mathbf {w} ^{T}\mathbf {w} }}\right\}}$$

Computing the [singular value decomposition (SVD)](https://en.wikipedia.org/wiki/Singular_value_decomposition) is now the standard way to calculate a principal components analysis from a data matrix


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```

## Read in the data and perform basic exploratory analysis


```python
df = pd.read_csv('wine.data.csv')
df.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Class</th>
      <th>Alcohol</th>
      <th>Malic acid</th>
      <th>Ash</th>
      <th>Alcalinity of ash</th>
      <th>Magnesium</th>
      <th>Total phenols</th>
      <th>Flavanoids</th>
      <th>Nonflavanoid phenols</th>
      <th>Proanthocyanins</th>
      <th>Color intensity</th>
      <th>Hue</th>
      <th>OD280/OD315 of diluted wines</th>
      <th>Proline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>14.23</td>
      <td>1.71</td>
      <td>2.43</td>
      <td>15.6</td>
      <td>127</td>
      <td>2.80</td>
      <td>3.06</td>
      <td>0.28</td>
      <td>2.29</td>
      <td>5.64</td>
      <td>1.04</td>
      <td>3.92</td>
      <td>1065</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>13.20</td>
      <td>1.78</td>
      <td>2.14</td>
      <td>11.2</td>
      <td>100</td>
      <td>2.65</td>
      <td>2.76</td>
      <td>0.26</td>
      <td>1.28</td>
      <td>4.38</td>
      <td>1.05</td>
      <td>3.40</td>
      <td>1050</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>13.16</td>
      <td>2.36</td>
      <td>2.67</td>
      <td>18.6</td>
      <td>101</td>
      <td>2.80</td>
      <td>3.24</td>
      <td>0.30</td>
      <td>2.81</td>
      <td>5.68</td>
      <td>1.03</td>
      <td>3.17</td>
      <td>1185</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>14.37</td>
      <td>1.95</td>
      <td>2.50</td>
      <td>16.8</td>
      <td>113</td>
      <td>3.85</td>
      <td>3.49</td>
      <td>0.24</td>
      <td>2.18</td>
      <td>7.80</td>
      <td>0.86</td>
      <td>3.45</td>
      <td>1480</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>13.24</td>
      <td>2.59</td>
      <td>2.87</td>
      <td>21.0</td>
      <td>118</td>
      <td>2.80</td>
      <td>2.69</td>
      <td>0.39</td>
      <td>1.82</td>
      <td>4.32</td>
      <td>1.04</td>
      <td>2.93</td>
      <td>735</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>14.20</td>
      <td>1.76</td>
      <td>2.45</td>
      <td>15.2</td>
      <td>112</td>
      <td>3.27</td>
      <td>3.39</td>
      <td>0.34</td>
      <td>1.97</td>
      <td>6.75</td>
      <td>1.05</td>
      <td>2.85</td>
      <td>1450</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>14.39</td>
      <td>1.87</td>
      <td>2.45</td>
      <td>14.6</td>
      <td>96</td>
      <td>2.50</td>
      <td>2.52</td>
      <td>0.30</td>
      <td>1.98</td>
      <td>5.25</td>
      <td>1.02</td>
      <td>3.58</td>
      <td>1290</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>14.06</td>
      <td>2.15</td>
      <td>2.61</td>
      <td>17.6</td>
      <td>121</td>
      <td>2.60</td>
      <td>2.51</td>
      <td>0.31</td>
      <td>1.25</td>
      <td>5.05</td>
      <td>1.06</td>
      <td>3.58</td>
      <td>1295</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>14.83</td>
      <td>1.64</td>
      <td>2.17</td>
      <td>14.0</td>
      <td>97</td>
      <td>2.80</td>
      <td>2.98</td>
      <td>0.29</td>
      <td>1.98</td>
      <td>5.20</td>
      <td>1.08</td>
      <td>2.85</td>
      <td>1045</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>13.86</td>
      <td>1.35</td>
      <td>2.27</td>
      <td>16.0</td>
      <td>98</td>
      <td>2.98</td>
      <td>3.15</td>
      <td>0.22</td>
      <td>1.85</td>
      <td>7.22</td>
      <td>1.01</td>
      <td>3.55</td>
      <td>1045</td>
    </tr>
  </tbody>
</table>
</div>



#### Basic statistics


```python
df.iloc[:,1:].describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Alcohol</th>
      <th>Malic acid</th>
      <th>Ash</th>
      <th>Alcalinity of ash</th>
      <th>Magnesium</th>
      <th>Total phenols</th>
      <th>Flavanoids</th>
      <th>Nonflavanoid phenols</th>
      <th>Proanthocyanins</th>
      <th>Color intensity</th>
      <th>Hue</th>
      <th>OD280/OD315 of diluted wines</th>
      <th>Proline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>13.000618</td>
      <td>2.336348</td>
      <td>2.366517</td>
      <td>19.494944</td>
      <td>99.741573</td>
      <td>2.295112</td>
      <td>2.029270</td>
      <td>0.361854</td>
      <td>1.590899</td>
      <td>5.058090</td>
      <td>0.957449</td>
      <td>2.611685</td>
      <td>746.893258</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.811827</td>
      <td>1.117146</td>
      <td>0.274344</td>
      <td>3.339564</td>
      <td>14.282484</td>
      <td>0.625851</td>
      <td>0.998859</td>
      <td>0.124453</td>
      <td>0.572359</td>
      <td>2.318286</td>
      <td>0.228572</td>
      <td>0.709990</td>
      <td>314.907474</td>
    </tr>
    <tr>
      <th>min</th>
      <td>11.030000</td>
      <td>0.740000</td>
      <td>1.360000</td>
      <td>10.600000</td>
      <td>70.000000</td>
      <td>0.980000</td>
      <td>0.340000</td>
      <td>0.130000</td>
      <td>0.410000</td>
      <td>1.280000</td>
      <td>0.480000</td>
      <td>1.270000</td>
      <td>278.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>12.362500</td>
      <td>1.602500</td>
      <td>2.210000</td>
      <td>17.200000</td>
      <td>88.000000</td>
      <td>1.742500</td>
      <td>1.205000</td>
      <td>0.270000</td>
      <td>1.250000</td>
      <td>3.220000</td>
      <td>0.782500</td>
      <td>1.937500</td>
      <td>500.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>13.050000</td>
      <td>1.865000</td>
      <td>2.360000</td>
      <td>19.500000</td>
      <td>98.000000</td>
      <td>2.355000</td>
      <td>2.135000</td>
      <td>0.340000</td>
      <td>1.555000</td>
      <td>4.690000</td>
      <td>0.965000</td>
      <td>2.780000</td>
      <td>673.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>13.677500</td>
      <td>3.082500</td>
      <td>2.557500</td>
      <td>21.500000</td>
      <td>107.000000</td>
      <td>2.800000</td>
      <td>2.875000</td>
      <td>0.437500</td>
      <td>1.950000</td>
      <td>6.200000</td>
      <td>1.120000</td>
      <td>3.170000</td>
      <td>985.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>14.830000</td>
      <td>5.800000</td>
      <td>3.230000</td>
      <td>30.000000</td>
      <td>162.000000</td>
      <td>3.880000</td>
      <td>5.080000</td>
      <td>0.660000</td>
      <td>3.580000</td>
      <td>13.000000</td>
      <td>1.710000</td>
      <td>4.000000</td>
      <td>1680.000000</td>
    </tr>
  </tbody>
</table>
</div>



#### Boxplots by output labels/classes


```python
for c in df.columns[1:]:
    df.boxplot(c,by='Class',figsize=(7,4),fontsize=14)
    plt.title("{}\n".format(c),fontsize=16)
    plt.xlabel("Wine Class", fontsize=16)
```


![png](output_7_0.png)



![png](output_7_1.png)



![png](output_7_2.png)



![png](output_7_3.png)



![png](output_7_4.png)



![png](output_7_5.png)



![png](output_7_6.png)



![png](output_7_7.png)



![png](output_7_8.png)



![png](output_7_9.png)



![png](output_7_10.png)



![png](output_7_11.png)



![png](output_7_12.png)


**It can be seen that some features classify the wine labels pretty clearly.** For example, Alcalinity, Total Phenols, or Flavonoids produce boxplots with well-separated medians, which are clearly indicative of wine classes.

Below is an example of class seperation using two variables


```python
plt.figure(figsize=(10,6))
plt.scatter(df['OD280/OD315 of diluted wines'],df['Flavanoids'],c=df['Class'],edgecolors='k',alpha=0.75,s=150)
plt.grid(True)
plt.title("Scatter plot of two features showing the \ncorrelation and class seperation",fontsize=15)
plt.xlabel("OD280/OD315 of diluted wines",fontsize=15)
plt.ylabel("Flavanoids",fontsize=15)
plt.show()
```


![png](output_9_0.png)


#### Are the features independent? Plot co-variance matrix

It can be seen that there are some good amount of correlation between features i.e. they are not independent of each other, as assumed in Naive Bayes technique. However, we will still go ahead and apply the classifier to see its performance.


```python
def correlation_matrix(df):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure(figsize=(16,12))
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Wine data set features correlation\n',fontsize=15)
    labels=df.columns
    ax1.set_xticklabels(labels,fontsize=9)
    ax1.set_yticklabels(labels,fontsize=9)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[0.1*i for i in range(-11,11)])
    plt.show()

correlation_matrix(df)
```


![png](output_11_0.png)


## Principal Component Analysis

### Data scaling
PCA requires scaling/normalization of the data to work properly


```python
from sklearn.preprocessing import StandardScaler
```


```python
scaler = StandardScaler()
```


```python
X = df.drop('Class',axis=1)
y = df['Class']
```


```python
X = scaler.fit_transform(X)
```


```python
dfx = pd.DataFrame(data=X,columns=df.columns[1:])
```


```python
dfx.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Alcohol</th>
      <th>Malic acid</th>
      <th>Ash</th>
      <th>Alcalinity of ash</th>
      <th>Magnesium</th>
      <th>Total phenols</th>
      <th>Flavanoids</th>
      <th>Nonflavanoid phenols</th>
      <th>Proanthocyanins</th>
      <th>Color intensity</th>
      <th>Hue</th>
      <th>OD280/OD315 of diluted wines</th>
      <th>Proline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.518613</td>
      <td>-0.562250</td>
      <td>0.232053</td>
      <td>-1.169593</td>
      <td>1.913905</td>
      <td>0.808997</td>
      <td>1.034819</td>
      <td>-0.659563</td>
      <td>1.224884</td>
      <td>0.251717</td>
      <td>0.362177</td>
      <td>1.847920</td>
      <td>1.013009</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.246290</td>
      <td>-0.499413</td>
      <td>-0.827996</td>
      <td>-2.490847</td>
      <td>0.018145</td>
      <td>0.568648</td>
      <td>0.733629</td>
      <td>-0.820719</td>
      <td>-0.544721</td>
      <td>-0.293321</td>
      <td>0.406051</td>
      <td>1.113449</td>
      <td>0.965242</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.196879</td>
      <td>0.021231</td>
      <td>1.109334</td>
      <td>-0.268738</td>
      <td>0.088358</td>
      <td>0.808997</td>
      <td>1.215533</td>
      <td>-0.498407</td>
      <td>2.135968</td>
      <td>0.269020</td>
      <td>0.318304</td>
      <td>0.788587</td>
      <td>1.395148</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.691550</td>
      <td>-0.346811</td>
      <td>0.487926</td>
      <td>-0.809251</td>
      <td>0.930918</td>
      <td>2.491446</td>
      <td>1.466525</td>
      <td>-0.981875</td>
      <td>1.032155</td>
      <td>1.186068</td>
      <td>-0.427544</td>
      <td>1.184071</td>
      <td>2.334574</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.295700</td>
      <td>0.227694</td>
      <td>1.840403</td>
      <td>0.451946</td>
      <td>1.281985</td>
      <td>0.808997</td>
      <td>0.663351</td>
      <td>0.226796</td>
      <td>0.401404</td>
      <td>-0.319276</td>
      <td>0.362177</td>
      <td>0.449601</td>
      <td>-0.037874</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.481555</td>
      <td>-0.517367</td>
      <td>0.305159</td>
      <td>-1.289707</td>
      <td>0.860705</td>
      <td>1.562093</td>
      <td>1.366128</td>
      <td>-0.176095</td>
      <td>0.664217</td>
      <td>0.731870</td>
      <td>0.406051</td>
      <td>0.336606</td>
      <td>2.239039</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.716255</td>
      <td>-0.418624</td>
      <td>0.305159</td>
      <td>-1.469878</td>
      <td>-0.262708</td>
      <td>0.328298</td>
      <td>0.492677</td>
      <td>-0.498407</td>
      <td>0.681738</td>
      <td>0.083015</td>
      <td>0.274431</td>
      <td>1.367689</td>
      <td>1.729520</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.308617</td>
      <td>-0.167278</td>
      <td>0.890014</td>
      <td>-0.569023</td>
      <td>1.492625</td>
      <td>0.488531</td>
      <td>0.482637</td>
      <td>-0.417829</td>
      <td>-0.597284</td>
      <td>-0.003499</td>
      <td>0.449924</td>
      <td>1.367689</td>
      <td>1.745442</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2.259772</td>
      <td>-0.625086</td>
      <td>-0.718336</td>
      <td>-1.650049</td>
      <td>-0.192495</td>
      <td>0.808997</td>
      <td>0.954502</td>
      <td>-0.578985</td>
      <td>0.681738</td>
      <td>0.061386</td>
      <td>0.537671</td>
      <td>0.336606</td>
      <td>0.949319</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.061565</td>
      <td>-0.885409</td>
      <td>-0.352802</td>
      <td>-1.049479</td>
      <td>-0.122282</td>
      <td>1.097417</td>
      <td>1.125176</td>
      <td>-1.143031</td>
      <td>0.453967</td>
      <td>0.935177</td>
      <td>0.230557</td>
      <td>1.325316</td>
      <td>0.949319</td>
    </tr>
  </tbody>
</table>
</div>




```python
dfx.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Alcohol</th>
      <th>Malic acid</th>
      <th>Ash</th>
      <th>Alcalinity of ash</th>
      <th>Magnesium</th>
      <th>Total phenols</th>
      <th>Flavanoids</th>
      <th>Nonflavanoid phenols</th>
      <th>Proanthocyanins</th>
      <th>Color intensity</th>
      <th>Hue</th>
      <th>OD280/OD315 of diluted wines</th>
      <th>Proline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.780000e+02</td>
      <td>1.780000e+02</td>
      <td>1.780000e+02</td>
      <td>1.780000e+02</td>
      <td>1.780000e+02</td>
      <td>1.780000e+02</td>
      <td>1.780000e+02</td>
      <td>1.780000e+02</td>
      <td>1.780000e+02</td>
      <td>1.780000e+02</td>
      <td>1.780000e+02</td>
      <td>1.780000e+02</td>
      <td>1.780000e+02</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-8.619821e-16</td>
      <td>-8.357859e-17</td>
      <td>-8.657245e-16</td>
      <td>-1.160121e-16</td>
      <td>-1.995907e-17</td>
      <td>-2.972030e-16</td>
      <td>-4.016762e-16</td>
      <td>4.079134e-16</td>
      <td>-1.699639e-16</td>
      <td>-1.247442e-18</td>
      <td>3.717376e-16</td>
      <td>2.919013e-16</td>
      <td>-7.484650e-18</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.002821e+00</td>
      <td>1.002821e+00</td>
      <td>1.002821e+00</td>
      <td>1.002821e+00</td>
      <td>1.002821e+00</td>
      <td>1.002821e+00</td>
      <td>1.002821e+00</td>
      <td>1.002821e+00</td>
      <td>1.002821e+00</td>
      <td>1.002821e+00</td>
      <td>1.002821e+00</td>
      <td>1.002821e+00</td>
      <td>1.002821e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.434235e+00</td>
      <td>-1.432983e+00</td>
      <td>-3.679162e+00</td>
      <td>-2.671018e+00</td>
      <td>-2.088255e+00</td>
      <td>-2.107246e+00</td>
      <td>-1.695971e+00</td>
      <td>-1.868234e+00</td>
      <td>-2.069034e+00</td>
      <td>-1.634288e+00</td>
      <td>-2.094732e+00</td>
      <td>-1.895054e+00</td>
      <td>-1.493188e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-7.882448e-01</td>
      <td>-6.587486e-01</td>
      <td>-5.721225e-01</td>
      <td>-6.891372e-01</td>
      <td>-8.244151e-01</td>
      <td>-8.854682e-01</td>
      <td>-8.275393e-01</td>
      <td>-7.401412e-01</td>
      <td>-5.972835e-01</td>
      <td>-7.951025e-01</td>
      <td>-7.675624e-01</td>
      <td>-9.522483e-01</td>
      <td>-7.846378e-01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>6.099988e-02</td>
      <td>-4.231120e-01</td>
      <td>-2.382132e-02</td>
      <td>1.518295e-03</td>
      <td>-1.222817e-01</td>
      <td>9.595986e-02</td>
      <td>1.061497e-01</td>
      <td>-1.760948e-01</td>
      <td>-6.289785e-02</td>
      <td>-1.592246e-01</td>
      <td>3.312687e-02</td>
      <td>2.377348e-01</td>
      <td>-2.337204e-01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8.361286e-01</td>
      <td>6.697929e-01</td>
      <td>6.981085e-01</td>
      <td>6.020883e-01</td>
      <td>5.096384e-01</td>
      <td>8.089974e-01</td>
      <td>8.490851e-01</td>
      <td>6.095413e-01</td>
      <td>6.291754e-01</td>
      <td>4.939560e-01</td>
      <td>7.131644e-01</td>
      <td>7.885875e-01</td>
      <td>7.582494e-01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.259772e+00</td>
      <td>3.109192e+00</td>
      <td>3.156325e+00</td>
      <td>3.154511e+00</td>
      <td>4.371372e+00</td>
      <td>2.539515e+00</td>
      <td>3.062832e+00</td>
      <td>2.402403e+00</td>
      <td>3.485073e+00</td>
      <td>3.435432e+00</td>
      <td>3.301694e+00</td>
      <td>1.960915e+00</td>
      <td>2.971473e+00</td>
    </tr>
  </tbody>
</table>
</div>



### PCA class import and analysis


```python
from sklearn.decomposition import PCA
```


```python
pca = PCA(n_components=None)
```


```python
dfx_pca = pca.fit(dfx)
```

#### Plot the _explained variance ratio_


```python
plt.figure(figsize=(10,6))
plt.scatter(x=[i+1 for i in range(len(dfx_pca.explained_variance_ratio_))],
            y=dfx_pca.explained_variance_ratio_,
           s=200, alpha=0.75,c='orange',edgecolor='k')
plt.grid(True)
plt.title("Explained variance ratio of the \nfitted principal component vector\n",fontsize=25)
plt.xlabel("Principal components",fontsize=15)
plt.xticks([i+1 for i in range(len(dfx_pca.explained_variance_ratio_))],fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("Explained variance ratio",fontsize=15)
plt.show()
```


![png](output_26_0.png)


**The above plot means that the $1^{st}$ principal component explains about 36% of the total variance in the data and the $2^{nd}$ component explians further 20%. Therefore, if we just consider first two components, they together explain 56% of the total variance.**

### Showing better class separation using principal components

#### Transform the scaled data set using the fitted PCA object


```python
dfx_trans = pca.transform(dfx)
```

#### Put it in a data frame


```python
dfx_trans = pd.DataFrame(data=dfx_trans)
dfx_trans.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.316751</td>
      <td>-1.443463</td>
      <td>-0.165739</td>
      <td>-0.215631</td>
      <td>0.693043</td>
      <td>-0.223880</td>
      <td>0.596427</td>
      <td>0.065139</td>
      <td>0.641443</td>
      <td>1.020956</td>
      <td>-0.451563</td>
      <td>0.540810</td>
      <td>-0.066239</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.209465</td>
      <td>0.333393</td>
      <td>-2.026457</td>
      <td>-0.291358</td>
      <td>-0.257655</td>
      <td>-0.927120</td>
      <td>0.053776</td>
      <td>1.024416</td>
      <td>-0.308847</td>
      <td>0.159701</td>
      <td>-0.142657</td>
      <td>0.388238</td>
      <td>0.003637</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.516740</td>
      <td>-1.031151</td>
      <td>0.982819</td>
      <td>0.724902</td>
      <td>-0.251033</td>
      <td>0.549276</td>
      <td>0.424205</td>
      <td>-0.344216</td>
      <td>-1.177834</td>
      <td>0.113361</td>
      <td>-0.286673</td>
      <td>0.000584</td>
      <td>0.021717</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.757066</td>
      <td>-2.756372</td>
      <td>-0.176192</td>
      <td>0.567983</td>
      <td>-0.311842</td>
      <td>0.114431</td>
      <td>-0.383337</td>
      <td>0.643593</td>
      <td>0.052544</td>
      <td>0.239413</td>
      <td>0.759584</td>
      <td>-0.242020</td>
      <td>-0.369484</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.008908</td>
      <td>-0.869831</td>
      <td>2.026688</td>
      <td>-0.409766</td>
      <td>0.298458</td>
      <td>-0.406520</td>
      <td>0.444074</td>
      <td>0.416700</td>
      <td>0.326819</td>
      <td>-0.078366</td>
      <td>-0.525945</td>
      <td>-0.216664</td>
      <td>-0.079364</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3.050254</td>
      <td>-2.122401</td>
      <td>-0.629396</td>
      <td>-0.515637</td>
      <td>-0.632019</td>
      <td>0.123431</td>
      <td>0.401654</td>
      <td>0.394893</td>
      <td>-0.152146</td>
      <td>-0.101996</td>
      <td>0.405585</td>
      <td>-0.379433</td>
      <td>0.145155</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2.449090</td>
      <td>-1.174850</td>
      <td>-0.977095</td>
      <td>-0.065831</td>
      <td>-1.027762</td>
      <td>-0.620121</td>
      <td>0.052891</td>
      <td>-0.371934</td>
      <td>-0.457016</td>
      <td>1.016563</td>
      <td>-0.442433</td>
      <td>0.141230</td>
      <td>-0.271778</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2.059437</td>
      <td>-1.608963</td>
      <td>0.146282</td>
      <td>-1.192608</td>
      <td>0.076903</td>
      <td>-1.439806</td>
      <td>0.032376</td>
      <td>0.232979</td>
      <td>0.123370</td>
      <td>0.735600</td>
      <td>0.293555</td>
      <td>0.379663</td>
      <td>-0.110164</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2.510874</td>
      <td>-0.918071</td>
      <td>-1.770969</td>
      <td>0.056270</td>
      <td>-0.892257</td>
      <td>-0.129181</td>
      <td>0.125285</td>
      <td>-0.499578</td>
      <td>0.606589</td>
      <td>0.174107</td>
      <td>-0.508933</td>
      <td>-0.635249</td>
      <td>0.142084</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2.753628</td>
      <td>-0.789438</td>
      <td>-0.984247</td>
      <td>0.349382</td>
      <td>-0.468553</td>
      <td>0.163392</td>
      <td>-0.874352</td>
      <td>0.150580</td>
      <td>0.230489</td>
      <td>0.179420</td>
      <td>0.012478</td>
      <td>0.550327</td>
      <td>-0.042455</td>
    </tr>
  </tbody>
</table>
</div>



#### Plot the first two columns of this transformed data set with the color set to original ground truth class label


```python
plt.figure(figsize=(10,6))
plt.scatter(dfx_trans[0],dfx_trans[1],c=df['Class'],edgecolors='k',alpha=0.75,s=150)
plt.grid(True)
plt.title("Class separation using first two principal components\n",fontsize=20)
plt.xlabel("Principal component-1",fontsize=15)
plt.ylabel("Principal component-2",fontsize=15)
plt.show()
```


![png](output_34_0.png)

