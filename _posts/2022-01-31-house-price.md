```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
submission=pd.read_csv("sample_submission.csv")
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
```


```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>Overall Qual</th>
      <th>Gr Liv Area</th>
      <th>Exter Qual</th>
      <th>Garage Cars</th>
      <th>Garage Area</th>
      <th>Kitchen Qual</th>
      <th>Total Bsmt SF</th>
      <th>1st Flr SF</th>
      <th>Bsmt Qual</th>
      <th>Full Bath</th>
      <th>Year Built</th>
      <th>Year Remod/Add</th>
      <th>Garage Yr Blt</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>10</td>
      <td>2392</td>
      <td>Ex</td>
      <td>3</td>
      <td>968</td>
      <td>Ex</td>
      <td>2392</td>
      <td>2392</td>
      <td>Ex</td>
      <td>2</td>
      <td>2003</td>
      <td>2003</td>
      <td>2003</td>
      <td>386250</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>7</td>
      <td>1352</td>
      <td>Gd</td>
      <td>2</td>
      <td>466</td>
      <td>Gd</td>
      <td>1352</td>
      <td>1352</td>
      <td>Ex</td>
      <td>2</td>
      <td>2006</td>
      <td>2007</td>
      <td>2006</td>
      <td>194000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>5</td>
      <td>900</td>
      <td>TA</td>
      <td>1</td>
      <td>288</td>
      <td>TA</td>
      <td>864</td>
      <td>900</td>
      <td>TA</td>
      <td>1</td>
      <td>1967</td>
      <td>1967</td>
      <td>1967</td>
      <td>123000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>5</td>
      <td>1174</td>
      <td>TA</td>
      <td>2</td>
      <td>576</td>
      <td>Gd</td>
      <td>680</td>
      <td>680</td>
      <td>TA</td>
      <td>1</td>
      <td>1900</td>
      <td>2006</td>
      <td>2000</td>
      <td>135000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>7</td>
      <td>1958</td>
      <td>Gd</td>
      <td>3</td>
      <td>936</td>
      <td>Gd</td>
      <td>1026</td>
      <td>1026</td>
      <td>Gd</td>
      <td>2</td>
      <td>2005</td>
      <td>2005</td>
      <td>2005</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.shape
```




    (1350, 15)




```python
'''OverallQual : 전반적 재료와 마감 품질
YearBuilt : 완공 연도
YearRemodAdd : 리모델링 연도
ExterQual : 외관 재료 품질
BsmtQual : 지하실 높이
TotalBsmtSF : 지하실 면적 
1stFlrSF : 1층 면적 
GrLivArea : 지상층 생활 면적
FullBath : 지상층 화장실 개수 
KitchenQual : 부억 품질 
GarageYrBlt : 차고 완공 연도
GarageCars: 차고 자리 개수
GarageArea: 차고 면적 
target : 집값(달러 단위)'''


```




    'OverallQual : 전반적 재료와 마감 품질\nYearBuilt : 완공 연도\nYearRemodAdd : 리모델링 연도\nExterQual : 외관 재료 품질\nBsmtQual : 지하실 높이\nTotalBsmtSF : 지하실 면적 \n1stFlrSF : 1층 면적 \nGrLivArea : 지상층 생활 면적\nFullBath : 지상층 화장실 개수 \nKitchenQual : 부억 품질 \nGarageYrBlt : 차고 완공 연도\nGarageCars: 차고 자리 개수\nGarageArea: 차고 면적 \ntarget : 집값(달러 단위)'




```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1350 entries, 0 to 1349
    Data columns (total 15 columns):
     #   Column          Non-Null Count  Dtype 
    ---  ------          --------------  ----- 
     0   id              1350 non-null   int64 
     1   Overall Qual    1350 non-null   int64 
     2   Gr Liv Area     1350 non-null   int64 
     3   Exter Qual      1350 non-null   object
     4   Garage Cars     1350 non-null   int64 
     5   Garage Area     1350 non-null   int64 
     6   Kitchen Qual    1350 non-null   object
     7   Total Bsmt SF   1350 non-null   int64 
     8   1st Flr SF      1350 non-null   int64 
     9   Bsmt Qual       1350 non-null   object
     10  Full Bath       1350 non-null   int64 
     11  Year Built      1350 non-null   int64 
     12  Year Remod/Add  1350 non-null   int64 
     13  Garage Yr Blt   1350 non-null   int64 
     14  target          1350 non-null   int64 
    dtypes: int64(12), object(3)
    memory usage: 158.3+ KB
    


```python
train=train.drop('id',axis=1)
```


```python
X_train=train.drop("target",axis=1)
y_train=train['target']
```


```python
X_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Overall Qual</th>
      <th>Gr Liv Area</th>
      <th>Exter Qual</th>
      <th>Garage Cars</th>
      <th>Garage Area</th>
      <th>Kitchen Qual</th>
      <th>Total Bsmt SF</th>
      <th>1st Flr SF</th>
      <th>Bsmt Qual</th>
      <th>Full Bath</th>
      <th>Year Built</th>
      <th>Year Remod/Add</th>
      <th>Garage Yr Blt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>2392</td>
      <td>Ex</td>
      <td>3</td>
      <td>968</td>
      <td>Ex</td>
      <td>2392</td>
      <td>2392</td>
      <td>Ex</td>
      <td>2</td>
      <td>2003</td>
      <td>2003</td>
      <td>2003</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>1352</td>
      <td>Gd</td>
      <td>2</td>
      <td>466</td>
      <td>Gd</td>
      <td>1352</td>
      <td>1352</td>
      <td>Ex</td>
      <td>2</td>
      <td>2006</td>
      <td>2007</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>900</td>
      <td>TA</td>
      <td>1</td>
      <td>288</td>
      <td>TA</td>
      <td>864</td>
      <td>900</td>
      <td>TA</td>
      <td>1</td>
      <td>1967</td>
      <td>1967</td>
      <td>1967</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>1174</td>
      <td>TA</td>
      <td>2</td>
      <td>576</td>
      <td>Gd</td>
      <td>680</td>
      <td>680</td>
      <td>TA</td>
      <td>1</td>
      <td>1900</td>
      <td>2006</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>1958</td>
      <td>Gd</td>
      <td>3</td>
      <td>936</td>
      <td>Gd</td>
      <td>1026</td>
      <td>1026</td>
      <td>Gd</td>
      <td>2</td>
      <td>2005</td>
      <td>2005</td>
      <td>2005</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1345</th>
      <td>6</td>
      <td>1756</td>
      <td>Gd</td>
      <td>2</td>
      <td>422</td>
      <td>TA</td>
      <td>872</td>
      <td>888</td>
      <td>Ex</td>
      <td>2</td>
      <td>1996</td>
      <td>1997</td>
      <td>1996</td>
    </tr>
    <tr>
      <th>1346</th>
      <td>9</td>
      <td>2748</td>
      <td>Gd</td>
      <td>3</td>
      <td>850</td>
      <td>Ex</td>
      <td>1850</td>
      <td>1850</td>
      <td>Ex</td>
      <td>2</td>
      <td>2006</td>
      <td>2006</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>1347</th>
      <td>5</td>
      <td>1214</td>
      <td>TA</td>
      <td>1</td>
      <td>318</td>
      <td>TA</td>
      <td>1214</td>
      <td>1214</td>
      <td>TA</td>
      <td>2</td>
      <td>1967</td>
      <td>1967</td>
      <td>1967</td>
    </tr>
    <tr>
      <th>1348</th>
      <td>5</td>
      <td>894</td>
      <td>TA</td>
      <td>2</td>
      <td>440</td>
      <td>TA</td>
      <td>864</td>
      <td>894</td>
      <td>Gd</td>
      <td>1</td>
      <td>1974</td>
      <td>1974</td>
      <td>1974</td>
    </tr>
    <tr>
      <th>1349</th>
      <td>5</td>
      <td>907</td>
      <td>TA</td>
      <td>1</td>
      <td>343</td>
      <td>TA</td>
      <td>907</td>
      <td>907</td>
      <td>Gd</td>
      <td>1</td>
      <td>1978</td>
      <td>1978</td>
      <td>1978</td>
    </tr>
  </tbody>
</table>
<p>1350 rows × 13 columns</p>
</div>




```python
X_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Overall Qual</th>
      <th>Gr Liv Area</th>
      <th>Exter Qual</th>
      <th>Garage Cars</th>
      <th>Garage Area</th>
      <th>Kitchen Qual</th>
      <th>Total Bsmt SF</th>
      <th>1st Flr SF</th>
      <th>Bsmt Qual</th>
      <th>Full Bath</th>
      <th>Year Built</th>
      <th>Year Remod/Add</th>
      <th>Garage Yr Blt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>2392</td>
      <td>Ex</td>
      <td>3</td>
      <td>968</td>
      <td>Ex</td>
      <td>2392</td>
      <td>2392</td>
      <td>Ex</td>
      <td>2</td>
      <td>2003</td>
      <td>2003</td>
      <td>2003</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>1352</td>
      <td>Gd</td>
      <td>2</td>
      <td>466</td>
      <td>Gd</td>
      <td>1352</td>
      <td>1352</td>
      <td>Ex</td>
      <td>2</td>
      <td>2006</td>
      <td>2007</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>900</td>
      <td>TA</td>
      <td>1</td>
      <td>288</td>
      <td>TA</td>
      <td>864</td>
      <td>900</td>
      <td>TA</td>
      <td>1</td>
      <td>1967</td>
      <td>1967</td>
      <td>1967</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>1174</td>
      <td>TA</td>
      <td>2</td>
      <td>576</td>
      <td>Gd</td>
      <td>680</td>
      <td>680</td>
      <td>TA</td>
      <td>1</td>
      <td>1900</td>
      <td>2006</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>1958</td>
      <td>Gd</td>
      <td>3</td>
      <td>936</td>
      <td>Gd</td>
      <td>1026</td>
      <td>1026</td>
      <td>Gd</td>
      <td>2</td>
      <td>2005</td>
      <td>2005</td>
      <td>2005</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1345</th>
      <td>6</td>
      <td>1756</td>
      <td>Gd</td>
      <td>2</td>
      <td>422</td>
      <td>TA</td>
      <td>872</td>
      <td>888</td>
      <td>Ex</td>
      <td>2</td>
      <td>1996</td>
      <td>1997</td>
      <td>1996</td>
    </tr>
    <tr>
      <th>1346</th>
      <td>9</td>
      <td>2748</td>
      <td>Gd</td>
      <td>3</td>
      <td>850</td>
      <td>Ex</td>
      <td>1850</td>
      <td>1850</td>
      <td>Ex</td>
      <td>2</td>
      <td>2006</td>
      <td>2006</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>1347</th>
      <td>5</td>
      <td>1214</td>
      <td>TA</td>
      <td>1</td>
      <td>318</td>
      <td>TA</td>
      <td>1214</td>
      <td>1214</td>
      <td>TA</td>
      <td>2</td>
      <td>1967</td>
      <td>1967</td>
      <td>1967</td>
    </tr>
    <tr>
      <th>1348</th>
      <td>5</td>
      <td>894</td>
      <td>TA</td>
      <td>2</td>
      <td>440</td>
      <td>TA</td>
      <td>864</td>
      <td>894</td>
      <td>Gd</td>
      <td>1</td>
      <td>1974</td>
      <td>1974</td>
      <td>1974</td>
    </tr>
    <tr>
      <th>1349</th>
      <td>5</td>
      <td>907</td>
      <td>TA</td>
      <td>1</td>
      <td>343</td>
      <td>TA</td>
      <td>907</td>
      <td>907</td>
      <td>Gd</td>
      <td>1</td>
      <td>1978</td>
      <td>1978</td>
      <td>1978</td>
    </tr>
  </tbody>
</table>
<p>1350 rows × 13 columns</p>
</div>




```python
numeric_feature=train.columns[(train.dtypes==np.int64)]
category_feature=train.columns[(train.dtypes==object)]
```


```python
numeric_feature
```




    Index(['Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Garage Area',
           'Total Bsmt SF', '1st Flr SF', 'Full Bath', 'Year Built',
           'Year Remod/Add', 'Garage Yr Blt', 'target'],
          dtype='object')




```python
numeric_train=train[numeric_feature]
category_train=train[category_feature]
```


```python
numeric_train.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Overall Qual</th>
      <th>Gr Liv Area</th>
      <th>Garage Cars</th>
      <th>Garage Area</th>
      <th>Total Bsmt SF</th>
      <th>1st Flr SF</th>
      <th>Full Bath</th>
      <th>Year Built</th>
      <th>Year Remod/Add</th>
      <th>Garage Yr Blt</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Overall Qual</th>
      <td>1.000000</td>
      <td>0.588243</td>
      <td>0.571066</td>
      <td>0.517218</td>
      <td>0.509058</td>
      <td>0.476398</td>
      <td>0.554248</td>
      <td>0.582875</td>
      <td>0.579142</td>
      <td>0.553927</td>
      <td>0.810607</td>
    </tr>
    <tr>
      <th>Gr Liv Area</th>
      <td>0.588243</td>
      <td>1.000000</td>
      <td>0.516287</td>
      <td>0.480863</td>
      <td>0.419220</td>
      <td>0.522535</td>
      <td>0.612423</td>
      <td>0.232281</td>
      <td>0.315723</td>
      <td>0.261861</td>
      <td>0.742061</td>
    </tr>
    <tr>
      <th>Garage Cars</th>
      <td>0.571066</td>
      <td>0.516287</td>
      <td>1.000000</td>
      <td>0.840054</td>
      <td>0.466017</td>
      <td>0.445017</td>
      <td>0.513078</td>
      <td>0.505803</td>
      <td>0.433307</td>
      <td>0.562320</td>
      <td>0.634497</td>
    </tr>
    <tr>
      <th>Garage Area</th>
      <td>0.517218</td>
      <td>0.480863</td>
      <td>0.840054</td>
      <td>1.000000</td>
      <td>0.499634</td>
      <td>0.485843</td>
      <td>0.418852</td>
      <td>0.445816</td>
      <td>0.364369</td>
      <td>0.536310</td>
      <td>0.617151</td>
    </tr>
    <tr>
      <th>Total Bsmt SF</th>
      <td>0.509058</td>
      <td>0.419220</td>
      <td>0.466017</td>
      <td>0.499634</td>
      <td>1.000000</td>
      <td>0.868811</td>
      <td>0.367648</td>
      <td>0.403767</td>
      <td>0.265506</td>
      <td>0.347908</td>
      <td>0.664047</td>
    </tr>
    <tr>
      <th>1st Flr SF</th>
      <td>0.476398</td>
      <td>0.522535</td>
      <td>0.445017</td>
      <td>0.485843</td>
      <td>0.868811</td>
      <td>1.000000</td>
      <td>0.370299</td>
      <td>0.290443</td>
      <td>0.236692</td>
      <td>0.259898</td>
      <td>0.646843</td>
    </tr>
    <tr>
      <th>Full Bath</th>
      <td>0.554248</td>
      <td>0.612423</td>
      <td>0.513078</td>
      <td>0.418852</td>
      <td>0.367648</td>
      <td>0.370299</td>
      <td>1.000000</td>
      <td>0.508745</td>
      <td>0.472537</td>
      <td>0.498692</td>
      <td>0.554453</td>
    </tr>
    <tr>
      <th>Year Built</th>
      <td>0.582875</td>
      <td>0.232281</td>
      <td>0.505803</td>
      <td>0.445816</td>
      <td>0.403767</td>
      <td>0.290443</td>
      <td>0.508745</td>
      <td>1.000000</td>
      <td>0.616008</td>
      <td>0.815615</td>
      <td>0.546037</td>
    </tr>
    <tr>
      <th>Year Remod/Add</th>
      <td>0.579142</td>
      <td>0.315723</td>
      <td>0.433307</td>
      <td>0.364369</td>
      <td>0.265506</td>
      <td>0.236692</td>
      <td>0.472537</td>
      <td>0.616008</td>
      <td>1.000000</td>
      <td>0.644251</td>
      <td>0.529477</td>
    </tr>
    <tr>
      <th>Garage Yr Blt</th>
      <td>0.553927</td>
      <td>0.261861</td>
      <td>0.562320</td>
      <td>0.536310</td>
      <td>0.347908</td>
      <td>0.259898</td>
      <td>0.498692</td>
      <td>0.815615</td>
      <td>0.644251</td>
      <td>1.000000</td>
      <td>0.517973</td>
    </tr>
    <tr>
      <th>target</th>
      <td>0.810607</td>
      <td>0.742061</td>
      <td>0.634497</td>
      <td>0.617151</td>
      <td>0.664047</td>
      <td>0.646843</td>
      <td>0.554453</td>
      <td>0.546037</td>
      <td>0.529477</td>
      <td>0.517973</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(100,60))
```




    <Figure size 7200x4320 with 0 Axes>




    <Figure size 7200x4320 with 0 Axes>



```python
plt.figure(figsize=(60,60))
plt.suptitle('boxplots',fontsize=100)
for i in range(len(numeric_feature)):
    plt.subplot(3,4,i+1)
    plt.title(numeric_feature[i],fontsize=30)
    plt.boxplot(numeric_train[numeric_feature[i]])
    plt.figsize=(20,15)
    plt.tick_params(labelsize=40)
```


    
![png](output_15_0.png)
    



```python
#outliers 이상치 제거
#iqr(1분위수와 3분위수 사이 차이)
#보통 q1-weight*iqr 부터 q3+weight*iqr 밖에 있는 값들을 이상치로 탐지
```


```python
#set 집합을 이용해 중복 제거
```


```python
category_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Exter Qual</th>
      <th>Kitchen Qual</th>
      <th>Bsmt Qual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ex</td>
      <td>Ex</td>
      <td>Ex</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Gd</td>
      <td>Gd</td>
      <td>Ex</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TA</td>
      <td>TA</td>
      <td>TA</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TA</td>
      <td>Gd</td>
      <td>TA</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Gd</td>
      <td>Gd</td>
      <td>Gd</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1345</th>
      <td>Gd</td>
      <td>TA</td>
      <td>Ex</td>
    </tr>
    <tr>
      <th>1346</th>
      <td>Gd</td>
      <td>Ex</td>
      <td>Ex</td>
    </tr>
    <tr>
      <th>1347</th>
      <td>TA</td>
      <td>TA</td>
      <td>TA</td>
    </tr>
    <tr>
      <th>1348</th>
      <td>TA</td>
      <td>TA</td>
      <td>Gd</td>
    </tr>
    <tr>
      <th>1349</th>
      <td>TA</td>
      <td>TA</td>
      <td>Gd</td>
    </tr>
  </tbody>
</table>
<p>1350 rows × 3 columns</p>
</div>




```python
category_train['Kitchen Qual'].value_counts()
```




    TA    660
    Gd    560
    Ex    107
    Fa     23
    Name: Kitchen Qual, dtype: int64




```python
sns.heatmap(numeric_train.corr(),annot=True,cmap='coolwarm')
```




    <AxesSubplot:>




    
![png](output_20_1.png)
    



```python
plt.figure(figsize=(60,60))
for i in range(len(numeric_feature)):
    plt.subplot(4,3,i+1)
    plt.scatter(numeric_train[numeric_feature[i]],numeric_train['target'],color='r',label=numeric_feature[i])
    plt.legend(fontsize=40)
```


    
![png](output_21_0.png)
    



```python
train.isna().sum()
```




    Overall Qual      0
    Gr Liv Area       0
    Exter Qual        0
    Garage Cars       0
    Garage Area       0
    Kitchen Qual      0
    Total Bsmt SF     0
    1st Flr SF        0
    Bsmt Qual         0
    Full Bath         0
    Year Built        0
    Year Remod/Add    0
    Garage Yr Blt     0
    target            0
    dtype: int64




```python
#labelencoder은 순위 없음
#ordinalencoder 은 숫자 사이에 순위가 존재함
#ordinal 보단 map으로 간단하게 가능
```


```python
train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Overall Qual</th>
      <th>Gr Liv Area</th>
      <th>Exter Qual</th>
      <th>Garage Cars</th>
      <th>Garage Area</th>
      <th>Kitchen Qual</th>
      <th>Total Bsmt SF</th>
      <th>1st Flr SF</th>
      <th>Bsmt Qual</th>
      <th>Full Bath</th>
      <th>Year Built</th>
      <th>Year Remod/Add</th>
      <th>Garage Yr Blt</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>2392</td>
      <td>Ex</td>
      <td>3</td>
      <td>968</td>
      <td>Ex</td>
      <td>2392</td>
      <td>2392</td>
      <td>Ex</td>
      <td>2</td>
      <td>2003</td>
      <td>2003</td>
      <td>2003</td>
      <td>386250</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>1352</td>
      <td>Gd</td>
      <td>2</td>
      <td>466</td>
      <td>Gd</td>
      <td>1352</td>
      <td>1352</td>
      <td>Ex</td>
      <td>2</td>
      <td>2006</td>
      <td>2007</td>
      <td>2006</td>
      <td>194000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>900</td>
      <td>TA</td>
      <td>1</td>
      <td>288</td>
      <td>TA</td>
      <td>864</td>
      <td>900</td>
      <td>TA</td>
      <td>1</td>
      <td>1967</td>
      <td>1967</td>
      <td>1967</td>
      <td>123000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>1174</td>
      <td>TA</td>
      <td>2</td>
      <td>576</td>
      <td>Gd</td>
      <td>680</td>
      <td>680</td>
      <td>TA</td>
      <td>1</td>
      <td>1900</td>
      <td>2006</td>
      <td>2000</td>
      <td>135000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>1958</td>
      <td>Gd</td>
      <td>3</td>
      <td>936</td>
      <td>Gd</td>
      <td>1026</td>
      <td>1026</td>
      <td>Gd</td>
      <td>2</td>
      <td>2005</td>
      <td>2005</td>
      <td>2005</td>
      <td>250000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1345</th>
      <td>6</td>
      <td>1756</td>
      <td>Gd</td>
      <td>2</td>
      <td>422</td>
      <td>TA</td>
      <td>872</td>
      <td>888</td>
      <td>Ex</td>
      <td>2</td>
      <td>1996</td>
      <td>1997</td>
      <td>1996</td>
      <td>204000</td>
    </tr>
    <tr>
      <th>1346</th>
      <td>9</td>
      <td>2748</td>
      <td>Gd</td>
      <td>3</td>
      <td>850</td>
      <td>Ex</td>
      <td>1850</td>
      <td>1850</td>
      <td>Ex</td>
      <td>2</td>
      <td>2006</td>
      <td>2006</td>
      <td>2006</td>
      <td>390000</td>
    </tr>
    <tr>
      <th>1347</th>
      <td>5</td>
      <td>1214</td>
      <td>TA</td>
      <td>1</td>
      <td>318</td>
      <td>TA</td>
      <td>1214</td>
      <td>1214</td>
      <td>TA</td>
      <td>2</td>
      <td>1967</td>
      <td>1967</td>
      <td>1967</td>
      <td>143000</td>
    </tr>
    <tr>
      <th>1348</th>
      <td>5</td>
      <td>894</td>
      <td>TA</td>
      <td>2</td>
      <td>440</td>
      <td>TA</td>
      <td>864</td>
      <td>894</td>
      <td>Gd</td>
      <td>1</td>
      <td>1974</td>
      <td>1974</td>
      <td>1974</td>
      <td>131000</td>
    </tr>
    <tr>
      <th>1349</th>
      <td>5</td>
      <td>907</td>
      <td>TA</td>
      <td>1</td>
      <td>343</td>
      <td>TA</td>
      <td>907</td>
      <td>907</td>
      <td>Gd</td>
      <td>1</td>
      <td>1978</td>
      <td>1978</td>
      <td>1978</td>
      <td>140000</td>
    </tr>
  </tbody>
</table>
<p>1350 rows × 14 columns</p>
</div>




```python
train['Exter Qual'].value_counts()
```




    TA    808
    Gd    485
    Ex     49
    Fa      8
    Name: Exter Qual, dtype: int64




```python
#품질이 높을수록 높은값 지정
```


```python
mapping={'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1}
```


```python
X_train['Exter Qual']=X_train['Exter Qual'].map(mapping)
X_train['Kitchen Qual']=X_train['Kitchen Qual'].map(mapping)
X_train['Bsmt Qual']=X_train['Bsmt Qual'].map(mapping)
```


```python
train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Overall Qual</th>
      <th>Gr Liv Area</th>
      <th>Exter Qual</th>
      <th>Garage Cars</th>
      <th>Garage Area</th>
      <th>Kitchen Qual</th>
      <th>Total Bsmt SF</th>
      <th>1st Flr SF</th>
      <th>Bsmt Qual</th>
      <th>Full Bath</th>
      <th>Year Built</th>
      <th>Year Remod/Add</th>
      <th>Garage Yr Blt</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>2392</td>
      <td>Ex</td>
      <td>3</td>
      <td>968</td>
      <td>Ex</td>
      <td>2392</td>
      <td>2392</td>
      <td>Ex</td>
      <td>2</td>
      <td>2003</td>
      <td>2003</td>
      <td>2003</td>
      <td>386250</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>1352</td>
      <td>Gd</td>
      <td>2</td>
      <td>466</td>
      <td>Gd</td>
      <td>1352</td>
      <td>1352</td>
      <td>Ex</td>
      <td>2</td>
      <td>2006</td>
      <td>2007</td>
      <td>2006</td>
      <td>194000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>900</td>
      <td>TA</td>
      <td>1</td>
      <td>288</td>
      <td>TA</td>
      <td>864</td>
      <td>900</td>
      <td>TA</td>
      <td>1</td>
      <td>1967</td>
      <td>1967</td>
      <td>1967</td>
      <td>123000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>1174</td>
      <td>TA</td>
      <td>2</td>
      <td>576</td>
      <td>Gd</td>
      <td>680</td>
      <td>680</td>
      <td>TA</td>
      <td>1</td>
      <td>1900</td>
      <td>2006</td>
      <td>2000</td>
      <td>135000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>1958</td>
      <td>Gd</td>
      <td>3</td>
      <td>936</td>
      <td>Gd</td>
      <td>1026</td>
      <td>1026</td>
      <td>Gd</td>
      <td>2</td>
      <td>2005</td>
      <td>2005</td>
      <td>2005</td>
      <td>250000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1345</th>
      <td>6</td>
      <td>1756</td>
      <td>Gd</td>
      <td>2</td>
      <td>422</td>
      <td>TA</td>
      <td>872</td>
      <td>888</td>
      <td>Ex</td>
      <td>2</td>
      <td>1996</td>
      <td>1997</td>
      <td>1996</td>
      <td>204000</td>
    </tr>
    <tr>
      <th>1346</th>
      <td>9</td>
      <td>2748</td>
      <td>Gd</td>
      <td>3</td>
      <td>850</td>
      <td>Ex</td>
      <td>1850</td>
      <td>1850</td>
      <td>Ex</td>
      <td>2</td>
      <td>2006</td>
      <td>2006</td>
      <td>2006</td>
      <td>390000</td>
    </tr>
    <tr>
      <th>1347</th>
      <td>5</td>
      <td>1214</td>
      <td>TA</td>
      <td>1</td>
      <td>318</td>
      <td>TA</td>
      <td>1214</td>
      <td>1214</td>
      <td>TA</td>
      <td>2</td>
      <td>1967</td>
      <td>1967</td>
      <td>1967</td>
      <td>143000</td>
    </tr>
    <tr>
      <th>1348</th>
      <td>5</td>
      <td>894</td>
      <td>TA</td>
      <td>2</td>
      <td>440</td>
      <td>TA</td>
      <td>864</td>
      <td>894</td>
      <td>Gd</td>
      <td>1</td>
      <td>1974</td>
      <td>1974</td>
      <td>1974</td>
      <td>131000</td>
    </tr>
    <tr>
      <th>1349</th>
      <td>5</td>
      <td>907</td>
      <td>TA</td>
      <td>1</td>
      <td>343</td>
      <td>TA</td>
      <td>907</td>
      <td>907</td>
      <td>Gd</td>
      <td>1</td>
      <td>1978</td>
      <td>1978</td>
      <td>1978</td>
      <td>140000</td>
    </tr>
  </tbody>
</table>
<p>1350 rows × 14 columns</p>
</div>




```python
def get_outlier(df=None,column=None,weight=1.5):
    data=df[column]
    quantile_25=np.percentile(data.values,25)
    quantile_75=np.percentile(data.values,75)
    iqr=quantile_75-quantile_25
    w_iqr=weight*iqr
    lowest=quantile_25-w_iqr
    highest=quantile_75+w_iqr
    outlier_index=data[(data<lowest)|(data>highest)].index
    return outlier_index

```


```python
index_list={}
for i,colname in enumerate(numeric_feature[:-1]):
    index_list[i]=get_outlier(train,colname)
```


```python
index_list
```




    {0: Int64Index([], dtype='int64'),
     1: Int64Index([  18,   94,  110,  132,  201,  297,  462,  476,  586,  663,  677,
                  683,  686,  735,  747,  752,  816,  856,  864,  939,  941,  948,
                  962, 1087, 1098, 1138, 1258, 1264, 1346],
                dtype='int64'),
     2: Int64Index([93, 297, 398, 503, 718, 735, 812, 939, 1010, 1163, 1173, 1228], dtype='int64'),
     3: Int64Index([   0,    4,   42,   90,  118,  242,  282,  297,  327,  377,  380,
                  503,  511,  660,  664,  721,  732,  839,  859,  934,  973, 1002,
                 1006, 1010, 1055, 1113, 1163, 1173, 1185, 1194, 1212, 1230, 1241,
                 1251, 1259, 1298],
                dtype='int64'),
     4: Int64Index([   0,    7,  163,  263,  273,  327,  380,  628,  732,  745,  856,
                  871,  942, 1002, 1034, 1101, 1221, 1298, 1311],
                dtype='int64'),
     5: Int64Index([   0,    7,  163,  263,  665,  732,  745,  811,  856,  871, 1002,
                 1034, 1098, 1149, 1161, 1191, 1221, 1311],
                dtype='int64'),
     6: Int64Index([735, 939], dtype='int64'),
     7: Int64Index([286, 812, 888, 940], dtype='int64'),
     8: Int64Index([], dtype='int64'),
     9: Int64Index([254], dtype='int64')}




```python
final_list=[]
for i in range(len(index_list)):
    if list(index_list[i].values)==[]:
        continue
    for j in index_list[i].values:
        final_list.append(j)


#set 집합 이용해 중복 제거
final_list=set(final_list)
```


```python
final_list
```




    {0,
     4,
     7,
     18,
     42,
     90,
     93,
     94,
     110,
     118,
     132,
     163,
     201,
     242,
     254,
     263,
     273,
     282,
     286,
     297,
     327,
     377,
     380,
     398,
     462,
     476,
     503,
     511,
     586,
     628,
     660,
     663,
     664,
     665,
     677,
     683,
     686,
     718,
     721,
     732,
     735,
     745,
     747,
     752,
     811,
     812,
     816,
     839,
     856,
     859,
     864,
     871,
     888,
     934,
     939,
     940,
     941,
     942,
     948,
     962,
     973,
     1002,
     1006,
     1010,
     1034,
     1055,
     1087,
     1098,
     1101,
     1113,
     1138,
     1149,
     1161,
     1163,
     1173,
     1185,
     1191,
     1194,
     1212,
     1221,
     1228,
     1230,
     1241,
     1251,
     1258,
     1259,
     1264,
     1298,
     1311,
     1346}




```python
for i in final_list:
    X_train.drop(i,axis=0,inplace=True)
    
for i in final_list:
    y_train.drop(i,axis=0,inplace=True)
```


```python
X_train.shape
```




    (1260, 13)




```python
y_train.shape
```




    (1260,)




```python
sns.heatmap(train.corr(),annot=True)
```




    <AxesSubplot:>




    
![png](output_38_1.png)
    



```python
X_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Overall Qual</th>
      <th>Gr Liv Area</th>
      <th>Exter Qual</th>
      <th>Garage Cars</th>
      <th>Garage Area</th>
      <th>Kitchen Qual</th>
      <th>Total Bsmt SF</th>
      <th>1st Flr SF</th>
      <th>Bsmt Qual</th>
      <th>Full Bath</th>
      <th>Year Built</th>
      <th>Year Remod/Add</th>
      <th>Garage Yr Blt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>1352</td>
      <td>4</td>
      <td>2</td>
      <td>466</td>
      <td>4</td>
      <td>1352</td>
      <td>1352</td>
      <td>5</td>
      <td>2</td>
      <td>2006</td>
      <td>2007</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>900</td>
      <td>3</td>
      <td>1</td>
      <td>288</td>
      <td>3</td>
      <td>864</td>
      <td>900</td>
      <td>3</td>
      <td>1</td>
      <td>1967</td>
      <td>1967</td>
      <td>1967</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>1174</td>
      <td>3</td>
      <td>2</td>
      <td>576</td>
      <td>4</td>
      <td>680</td>
      <td>680</td>
      <td>3</td>
      <td>1</td>
      <td>1900</td>
      <td>2006</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>8</td>
      <td>1968</td>
      <td>4</td>
      <td>3</td>
      <td>680</td>
      <td>5</td>
      <td>774</td>
      <td>774</td>
      <td>5</td>
      <td>2</td>
      <td>2009</td>
      <td>2010</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>1478</td>
      <td>3</td>
      <td>2</td>
      <td>442</td>
      <td>3</td>
      <td>1478</td>
      <td>1478</td>
      <td>3</td>
      <td>1</td>
      <td>1957</td>
      <td>1957</td>
      <td>1957</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1344</th>
      <td>6</td>
      <td>865</td>
      <td>3</td>
      <td>1</td>
      <td>216</td>
      <td>3</td>
      <td>660</td>
      <td>740</td>
      <td>3</td>
      <td>1</td>
      <td>1920</td>
      <td>1995</td>
      <td>1920</td>
    </tr>
    <tr>
      <th>1345</th>
      <td>6</td>
      <td>1756</td>
      <td>4</td>
      <td>2</td>
      <td>422</td>
      <td>3</td>
      <td>872</td>
      <td>888</td>
      <td>5</td>
      <td>2</td>
      <td>1996</td>
      <td>1997</td>
      <td>1996</td>
    </tr>
    <tr>
      <th>1347</th>
      <td>5</td>
      <td>1214</td>
      <td>3</td>
      <td>1</td>
      <td>318</td>
      <td>3</td>
      <td>1214</td>
      <td>1214</td>
      <td>3</td>
      <td>2</td>
      <td>1967</td>
      <td>1967</td>
      <td>1967</td>
    </tr>
    <tr>
      <th>1348</th>
      <td>5</td>
      <td>894</td>
      <td>3</td>
      <td>2</td>
      <td>440</td>
      <td>3</td>
      <td>864</td>
      <td>894</td>
      <td>4</td>
      <td>1</td>
      <td>1974</td>
      <td>1974</td>
      <td>1974</td>
    </tr>
    <tr>
      <th>1349</th>
      <td>5</td>
      <td>907</td>
      <td>3</td>
      <td>1</td>
      <td>343</td>
      <td>3</td>
      <td>907</td>
      <td>907</td>
      <td>4</td>
      <td>1</td>
      <td>1978</td>
      <td>1978</td>
      <td>1978</td>
    </tr>
  </tbody>
</table>
<p>1260 rows × 13 columns</p>
</div>




```python
'''OverallQual : 전반적 재료와 마감 품질
YearBuilt : 완공 연도
YearRemodAdd : 리모델링 연도
ExterQual : 외관 재료 품질
BsmtQual : 지하실 높이
TotalBsmtSF : 지하실 면적 
1stFlrSF : 1층 면적 
GrLivArea : 지상층 생활 면적
FullBath : 지상층 화장실 개수 
KitchenQual : 부억 품질 
GarageYrBlt : 차고 완공 연도
GarageCars: 차고 자리 개수
GarageArea: 차고 면적 
target : 집값(달러 단위)'''


```




    'OverallQual : 전반적 재료와 마감 품질\nYearBuilt : 완공 연도\nYearRemodAdd : 리모델링 연도\nExterQual : 외관 재료 품질\nBsmtQual : 지하실 높이\nTotalBsmtSF : 지하실 면적 \n1stFlrSF : 1층 면적 \nGrLivArea : 지상층 생활 면적\nFullBath : 지상층 화장실 개수 \nKitchenQual : 부억 품질 \nGarageYrBlt : 차고 완공 연도\nGarageCars: 차고 자리 개수\nGarageArea: 차고 면적 \ntarget : 집값(달러 단위)'




```python
X_train.shape
```




    (1260, 13)




```python
y_train.shape
```




    (1260,)




```python
X_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Overall Qual</th>
      <th>Gr Liv Area</th>
      <th>Exter Qual</th>
      <th>Garage Cars</th>
      <th>Garage Area</th>
      <th>Kitchen Qual</th>
      <th>Total Bsmt SF</th>
      <th>1st Flr SF</th>
      <th>Bsmt Qual</th>
      <th>Full Bath</th>
      <th>Year Built</th>
      <th>Year Remod/Add</th>
      <th>Garage Yr Blt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>1352</td>
      <td>4</td>
      <td>2</td>
      <td>466</td>
      <td>4</td>
      <td>1352</td>
      <td>1352</td>
      <td>5</td>
      <td>2</td>
      <td>2006</td>
      <td>2007</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>900</td>
      <td>3</td>
      <td>1</td>
      <td>288</td>
      <td>3</td>
      <td>864</td>
      <td>900</td>
      <td>3</td>
      <td>1</td>
      <td>1967</td>
      <td>1967</td>
      <td>1967</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>1174</td>
      <td>3</td>
      <td>2</td>
      <td>576</td>
      <td>4</td>
      <td>680</td>
      <td>680</td>
      <td>3</td>
      <td>1</td>
      <td>1900</td>
      <td>2006</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>8</td>
      <td>1968</td>
      <td>4</td>
      <td>3</td>
      <td>680</td>
      <td>5</td>
      <td>774</td>
      <td>774</td>
      <td>5</td>
      <td>2</td>
      <td>2009</td>
      <td>2010</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>1478</td>
      <td>3</td>
      <td>2</td>
      <td>442</td>
      <td>3</td>
      <td>1478</td>
      <td>1478</td>
      <td>3</td>
      <td>1</td>
      <td>1957</td>
      <td>1957</td>
      <td>1957</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1344</th>
      <td>6</td>
      <td>865</td>
      <td>3</td>
      <td>1</td>
      <td>216</td>
      <td>3</td>
      <td>660</td>
      <td>740</td>
      <td>3</td>
      <td>1</td>
      <td>1920</td>
      <td>1995</td>
      <td>1920</td>
    </tr>
    <tr>
      <th>1345</th>
      <td>6</td>
      <td>1756</td>
      <td>4</td>
      <td>2</td>
      <td>422</td>
      <td>3</td>
      <td>872</td>
      <td>888</td>
      <td>5</td>
      <td>2</td>
      <td>1996</td>
      <td>1997</td>
      <td>1996</td>
    </tr>
    <tr>
      <th>1347</th>
      <td>5</td>
      <td>1214</td>
      <td>3</td>
      <td>1</td>
      <td>318</td>
      <td>3</td>
      <td>1214</td>
      <td>1214</td>
      <td>3</td>
      <td>2</td>
      <td>1967</td>
      <td>1967</td>
      <td>1967</td>
    </tr>
    <tr>
      <th>1348</th>
      <td>5</td>
      <td>894</td>
      <td>3</td>
      <td>2</td>
      <td>440</td>
      <td>3</td>
      <td>864</td>
      <td>894</td>
      <td>4</td>
      <td>1</td>
      <td>1974</td>
      <td>1974</td>
      <td>1974</td>
    </tr>
    <tr>
      <th>1349</th>
      <td>5</td>
      <td>907</td>
      <td>3</td>
      <td>1</td>
      <td>343</td>
      <td>3</td>
      <td>907</td>
      <td>907</td>
      <td>4</td>
      <td>1</td>
      <td>1978</td>
      <td>1978</td>
      <td>1978</td>
    </tr>
  </tbody>
</table>
<p>1260 rows × 13 columns</p>
</div>




```python
numeric_feature
```




    Index(['Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Garage Area',
           'Total Bsmt SF', '1st Flr SF', 'Full Bath', 'Year Built',
           'Year Remod/Add', 'Garage Yr Blt', 'target'],
          dtype='object')




```python
X_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Overall Qual</th>
      <th>Gr Liv Area</th>
      <th>Exter Qual</th>
      <th>Garage Cars</th>
      <th>Garage Area</th>
      <th>Kitchen Qual</th>
      <th>Total Bsmt SF</th>
      <th>1st Flr SF</th>
      <th>Bsmt Qual</th>
      <th>Full Bath</th>
      <th>Year Built</th>
      <th>Year Remod/Add</th>
      <th>Garage Yr Blt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>1352</td>
      <td>4</td>
      <td>2</td>
      <td>466</td>
      <td>4</td>
      <td>1352</td>
      <td>1352</td>
      <td>5</td>
      <td>2</td>
      <td>2006</td>
      <td>2007</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>900</td>
      <td>3</td>
      <td>1</td>
      <td>288</td>
      <td>3</td>
      <td>864</td>
      <td>900</td>
      <td>3</td>
      <td>1</td>
      <td>1967</td>
      <td>1967</td>
      <td>1967</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>1174</td>
      <td>3</td>
      <td>2</td>
      <td>576</td>
      <td>4</td>
      <td>680</td>
      <td>680</td>
      <td>3</td>
      <td>1</td>
      <td>1900</td>
      <td>2006</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>8</td>
      <td>1968</td>
      <td>4</td>
      <td>3</td>
      <td>680</td>
      <td>5</td>
      <td>774</td>
      <td>774</td>
      <td>5</td>
      <td>2</td>
      <td>2009</td>
      <td>2010</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>1478</td>
      <td>3</td>
      <td>2</td>
      <td>442</td>
      <td>3</td>
      <td>1478</td>
      <td>1478</td>
      <td>3</td>
      <td>1</td>
      <td>1957</td>
      <td>1957</td>
      <td>1957</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1344</th>
      <td>6</td>
      <td>865</td>
      <td>3</td>
      <td>1</td>
      <td>216</td>
      <td>3</td>
      <td>660</td>
      <td>740</td>
      <td>3</td>
      <td>1</td>
      <td>1920</td>
      <td>1995</td>
      <td>1920</td>
    </tr>
    <tr>
      <th>1345</th>
      <td>6</td>
      <td>1756</td>
      <td>4</td>
      <td>2</td>
      <td>422</td>
      <td>3</td>
      <td>872</td>
      <td>888</td>
      <td>5</td>
      <td>2</td>
      <td>1996</td>
      <td>1997</td>
      <td>1996</td>
    </tr>
    <tr>
      <th>1347</th>
      <td>5</td>
      <td>1214</td>
      <td>3</td>
      <td>1</td>
      <td>318</td>
      <td>3</td>
      <td>1214</td>
      <td>1214</td>
      <td>3</td>
      <td>2</td>
      <td>1967</td>
      <td>1967</td>
      <td>1967</td>
    </tr>
    <tr>
      <th>1348</th>
      <td>5</td>
      <td>894</td>
      <td>3</td>
      <td>2</td>
      <td>440</td>
      <td>3</td>
      <td>864</td>
      <td>894</td>
      <td>4</td>
      <td>1</td>
      <td>1974</td>
      <td>1974</td>
      <td>1974</td>
    </tr>
    <tr>
      <th>1349</th>
      <td>5</td>
      <td>907</td>
      <td>3</td>
      <td>1</td>
      <td>343</td>
      <td>3</td>
      <td>907</td>
      <td>907</td>
      <td>4</td>
      <td>1</td>
      <td>1978</td>
      <td>1978</td>
      <td>1978</td>
    </tr>
  </tbody>
</table>
<p>1260 rows × 13 columns</p>
</div>




```python
X_train['Overall Qual'].value_counts()
```




    5     371
    6     341
    7     289
    8     129
    4      73
    9      40
    3       7
    10      7
    2       3
    Name: Overall Qual, dtype: int64




```python
test['Overall Qual'].value_counts()
```




    6     356
    5     349
    7     287
    8     191
    4      87
    9      51
    10     16
    3       9
    2       4
    Name: Overall Qual, dtype: int64




```python
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
for column in numeric_feature[:-1]:
    scaler.fit(pd.DataFrame(X_train[column]))
    X_train[column]=scaler.transform(pd.DataFrame(X_train[column]))
    
```


```python
X_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Overall Qual</th>
      <th>Gr Liv Area</th>
      <th>Exter Qual</th>
      <th>Garage Cars</th>
      <th>Garage Area</th>
      <th>Kitchen Qual</th>
      <th>Total Bsmt SF</th>
      <th>1st Flr SF</th>
      <th>Bsmt Qual</th>
      <th>Full Bath</th>
      <th>Year Built</th>
      <th>Year Remod/Add</th>
      <th>Garage Yr Blt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.625</td>
      <td>0.390506</td>
      <td>4</td>
      <td>0.5</td>
      <td>0.450739</td>
      <td>4</td>
      <td>0.642452</td>
      <td>0.520286</td>
      <td>5</td>
      <td>0.666667</td>
      <td>0.968</td>
      <td>0.950000</td>
      <td>0.963636</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.375</td>
      <td>0.188088</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.231527</td>
      <td>3</td>
      <td>0.391036</td>
      <td>0.250597</td>
      <td>3</td>
      <td>0.333333</td>
      <td>0.656</td>
      <td>0.283333</td>
      <td>0.609091</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.375</td>
      <td>0.310793</td>
      <td>3</td>
      <td>0.5</td>
      <td>0.586207</td>
      <td>4</td>
      <td>0.296239</td>
      <td>0.119332</td>
      <td>3</td>
      <td>0.333333</td>
      <td>0.120</td>
      <td>0.933333</td>
      <td>0.909091</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.750</td>
      <td>0.666368</td>
      <td>4</td>
      <td>1.0</td>
      <td>0.714286</td>
      <td>5</td>
      <td>0.344668</td>
      <td>0.175418</td>
      <td>5</td>
      <td>0.666667</td>
      <td>0.992</td>
      <td>1.000000</td>
      <td>0.990909</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.500</td>
      <td>0.446932</td>
      <td>3</td>
      <td>0.5</td>
      <td>0.421182</td>
      <td>3</td>
      <td>0.707367</td>
      <td>0.595465</td>
      <td>3</td>
      <td>0.333333</td>
      <td>0.576</td>
      <td>0.116667</td>
      <td>0.518182</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1344</th>
      <td>0.500</td>
      <td>0.172414</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.142857</td>
      <td>3</td>
      <td>0.285935</td>
      <td>0.155131</td>
      <td>3</td>
      <td>0.333333</td>
      <td>0.280</td>
      <td>0.750000</td>
      <td>0.181818</td>
    </tr>
    <tr>
      <th>1345</th>
      <td>0.500</td>
      <td>0.571429</td>
      <td>4</td>
      <td>0.5</td>
      <td>0.396552</td>
      <td>3</td>
      <td>0.395157</td>
      <td>0.243437</td>
      <td>5</td>
      <td>0.666667</td>
      <td>0.888</td>
      <td>0.783333</td>
      <td>0.872727</td>
    </tr>
    <tr>
      <th>1347</th>
      <td>0.375</td>
      <td>0.328706</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.268473</td>
      <td>3</td>
      <td>0.571355</td>
      <td>0.437947</td>
      <td>3</td>
      <td>0.666667</td>
      <td>0.656</td>
      <td>0.283333</td>
      <td>0.609091</td>
    </tr>
    <tr>
      <th>1348</th>
      <td>0.375</td>
      <td>0.185401</td>
      <td>3</td>
      <td>0.5</td>
      <td>0.418719</td>
      <td>3</td>
      <td>0.391036</td>
      <td>0.247017</td>
      <td>4</td>
      <td>0.333333</td>
      <td>0.712</td>
      <td>0.400000</td>
      <td>0.672727</td>
    </tr>
    <tr>
      <th>1349</th>
      <td>0.375</td>
      <td>0.191223</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.299261</td>
      <td>3</td>
      <td>0.413189</td>
      <td>0.254773</td>
      <td>4</td>
      <td>0.333333</td>
      <td>0.744</td>
      <td>0.466667</td>
      <td>0.709091</td>
    </tr>
  </tbody>
</table>
<p>1260 rows × 13 columns</p>
</div>




```python
cols_categorical = ['Exter Qual', 'Kitchen Qual', 'Bsmt Qual']
for colName in cols_categorical:
    X_train[colName] = X_train[colName].astype('category')
    test[colName] = test[colName].astype('category')
```


```python
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
```


```python
test
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>Overall Qual</th>
      <th>Gr Liv Area</th>
      <th>Exter Qual</th>
      <th>Garage Cars</th>
      <th>Garage Area</th>
      <th>Kitchen Qual</th>
      <th>Total Bsmt SF</th>
      <th>1st Flr SF</th>
      <th>Bsmt Qual</th>
      <th>Full Bath</th>
      <th>Year Built</th>
      <th>Year Remod/Add</th>
      <th>Garage Yr Blt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>9</td>
      <td>1800</td>
      <td>Gd</td>
      <td>2</td>
      <td>702</td>
      <td>Ex</td>
      <td>1800</td>
      <td>1800</td>
      <td>Ex</td>
      <td>2</td>
      <td>2007</td>
      <td>2007</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>6</td>
      <td>1082</td>
      <td>TA</td>
      <td>1</td>
      <td>240</td>
      <td>TA</td>
      <td>1082</td>
      <td>1082</td>
      <td>TA</td>
      <td>1</td>
      <td>1948</td>
      <td>1950</td>
      <td>1948</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>6</td>
      <td>1573</td>
      <td>Gd</td>
      <td>2</td>
      <td>440</td>
      <td>Gd</td>
      <td>756</td>
      <td>769</td>
      <td>Gd</td>
      <td>2</td>
      <td>2000</td>
      <td>2000</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>6</td>
      <td>2443</td>
      <td>Gd</td>
      <td>3</td>
      <td>744</td>
      <td>Gd</td>
      <td>1158</td>
      <td>1158</td>
      <td>Gd</td>
      <td>2</td>
      <td>2004</td>
      <td>2004</td>
      <td>2004</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5</td>
      <td>1040</td>
      <td>TA</td>
      <td>2</td>
      <td>686</td>
      <td>TA</td>
      <td>1040</td>
      <td>1040</td>
      <td>TA</td>
      <td>1</td>
      <td>1968</td>
      <td>1968</td>
      <td>1991</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1345</th>
      <td>1346</td>
      <td>8</td>
      <td>1932</td>
      <td>Ex</td>
      <td>3</td>
      <td>774</td>
      <td>Ex</td>
      <td>1932</td>
      <td>1932</td>
      <td>Ex</td>
      <td>2</td>
      <td>2008</td>
      <td>2008</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>1346</th>
      <td>1347</td>
      <td>5</td>
      <td>912</td>
      <td>TA</td>
      <td>1</td>
      <td>288</td>
      <td>TA</td>
      <td>912</td>
      <td>912</td>
      <td>TA</td>
      <td>1</td>
      <td>1964</td>
      <td>1964</td>
      <td>1964</td>
    </tr>
    <tr>
      <th>1347</th>
      <td>1348</td>
      <td>4</td>
      <td>861</td>
      <td>TA</td>
      <td>2</td>
      <td>288</td>
      <td>TA</td>
      <td>861</td>
      <td>861</td>
      <td>Fa</td>
      <td>1</td>
      <td>1920</td>
      <td>1950</td>
      <td>1920</td>
    </tr>
    <tr>
      <th>1348</th>
      <td>1349</td>
      <td>5</td>
      <td>1430</td>
      <td>TA</td>
      <td>2</td>
      <td>624</td>
      <td>Gd</td>
      <td>1430</td>
      <td>1430</td>
      <td>Ex</td>
      <td>2</td>
      <td>2004</td>
      <td>2005</td>
      <td>2004</td>
    </tr>
    <tr>
      <th>1349</th>
      <td>1350</td>
      <td>5</td>
      <td>2337</td>
      <td>TA</td>
      <td>2</td>
      <td>560</td>
      <td>TA</td>
      <td>662</td>
      <td>1422</td>
      <td>TA</td>
      <td>2</td>
      <td>1900</td>
      <td>1950</td>
      <td>1945</td>
    </tr>
  </tbody>
</table>
<p>1350 rows × 14 columns</p>
</div>




```python
test=test.drop('id',axis=1)
```


```python
mapping={'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1}
test['Exter Qual']=test['Exter Qual'].map(mapping)
test['Kitchen Qual']=test['Kitchen Qual'].map(mapping)
test['Bsmt Qual']=test['Bsmt Qual'].map(mapping)
```


```python
test
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Overall Qual</th>
      <th>Gr Liv Area</th>
      <th>Exter Qual</th>
      <th>Garage Cars</th>
      <th>Garage Area</th>
      <th>Kitchen Qual</th>
      <th>Total Bsmt SF</th>
      <th>1st Flr SF</th>
      <th>Bsmt Qual</th>
      <th>Full Bath</th>
      <th>Year Built</th>
      <th>Year Remod/Add</th>
      <th>Garage Yr Blt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9</td>
      <td>1800</td>
      <td>4</td>
      <td>2</td>
      <td>702</td>
      <td>5</td>
      <td>1800</td>
      <td>1800</td>
      <td>5</td>
      <td>2</td>
      <td>2007</td>
      <td>2007</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>1082</td>
      <td>3</td>
      <td>1</td>
      <td>240</td>
      <td>3</td>
      <td>1082</td>
      <td>1082</td>
      <td>3</td>
      <td>1</td>
      <td>1948</td>
      <td>1950</td>
      <td>1948</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>1573</td>
      <td>4</td>
      <td>2</td>
      <td>440</td>
      <td>4</td>
      <td>756</td>
      <td>769</td>
      <td>4</td>
      <td>2</td>
      <td>2000</td>
      <td>2000</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>2443</td>
      <td>4</td>
      <td>3</td>
      <td>744</td>
      <td>4</td>
      <td>1158</td>
      <td>1158</td>
      <td>4</td>
      <td>2</td>
      <td>2004</td>
      <td>2004</td>
      <td>2004</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1040</td>
      <td>3</td>
      <td>2</td>
      <td>686</td>
      <td>3</td>
      <td>1040</td>
      <td>1040</td>
      <td>3</td>
      <td>1</td>
      <td>1968</td>
      <td>1968</td>
      <td>1991</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1345</th>
      <td>8</td>
      <td>1932</td>
      <td>5</td>
      <td>3</td>
      <td>774</td>
      <td>5</td>
      <td>1932</td>
      <td>1932</td>
      <td>5</td>
      <td>2</td>
      <td>2008</td>
      <td>2008</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>1346</th>
      <td>5</td>
      <td>912</td>
      <td>3</td>
      <td>1</td>
      <td>288</td>
      <td>3</td>
      <td>912</td>
      <td>912</td>
      <td>3</td>
      <td>1</td>
      <td>1964</td>
      <td>1964</td>
      <td>1964</td>
    </tr>
    <tr>
      <th>1347</th>
      <td>4</td>
      <td>861</td>
      <td>3</td>
      <td>2</td>
      <td>288</td>
      <td>3</td>
      <td>861</td>
      <td>861</td>
      <td>2</td>
      <td>1</td>
      <td>1920</td>
      <td>1950</td>
      <td>1920</td>
    </tr>
    <tr>
      <th>1348</th>
      <td>5</td>
      <td>1430</td>
      <td>3</td>
      <td>2</td>
      <td>624</td>
      <td>4</td>
      <td>1430</td>
      <td>1430</td>
      <td>5</td>
      <td>2</td>
      <td>2004</td>
      <td>2005</td>
      <td>2004</td>
    </tr>
    <tr>
      <th>1349</th>
      <td>5</td>
      <td>2337</td>
      <td>3</td>
      <td>2</td>
      <td>560</td>
      <td>3</td>
      <td>662</td>
      <td>1422</td>
      <td>3</td>
      <td>2</td>
      <td>1900</td>
      <td>1950</td>
      <td>1945</td>
    </tr>
  </tbody>
</table>
<p>1350 rows × 13 columns</p>
</div>




```python
X_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Overall Qual</th>
      <th>Gr Liv Area</th>
      <th>Exter Qual</th>
      <th>Garage Cars</th>
      <th>Garage Area</th>
      <th>Kitchen Qual</th>
      <th>Total Bsmt SF</th>
      <th>1st Flr SF</th>
      <th>Bsmt Qual</th>
      <th>Full Bath</th>
      <th>Year Built</th>
      <th>Year Remod/Add</th>
      <th>Garage Yr Blt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.625</td>
      <td>0.390506</td>
      <td>4</td>
      <td>0.5</td>
      <td>0.450739</td>
      <td>4</td>
      <td>0.642452</td>
      <td>0.520286</td>
      <td>5</td>
      <td>0.666667</td>
      <td>0.968</td>
      <td>0.950000</td>
      <td>0.963636</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.375</td>
      <td>0.188088</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.231527</td>
      <td>3</td>
      <td>0.391036</td>
      <td>0.250597</td>
      <td>3</td>
      <td>0.333333</td>
      <td>0.656</td>
      <td>0.283333</td>
      <td>0.609091</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.375</td>
      <td>0.310793</td>
      <td>3</td>
      <td>0.5</td>
      <td>0.586207</td>
      <td>4</td>
      <td>0.296239</td>
      <td>0.119332</td>
      <td>3</td>
      <td>0.333333</td>
      <td>0.120</td>
      <td>0.933333</td>
      <td>0.909091</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.750</td>
      <td>0.666368</td>
      <td>4</td>
      <td>1.0</td>
      <td>0.714286</td>
      <td>5</td>
      <td>0.344668</td>
      <td>0.175418</td>
      <td>5</td>
      <td>0.666667</td>
      <td>0.992</td>
      <td>1.000000</td>
      <td>0.990909</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.500</td>
      <td>0.446932</td>
      <td>3</td>
      <td>0.5</td>
      <td>0.421182</td>
      <td>3</td>
      <td>0.707367</td>
      <td>0.595465</td>
      <td>3</td>
      <td>0.333333</td>
      <td>0.576</td>
      <td>0.116667</td>
      <td>0.518182</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1344</th>
      <td>0.500</td>
      <td>0.172414</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.142857</td>
      <td>3</td>
      <td>0.285935</td>
      <td>0.155131</td>
      <td>3</td>
      <td>0.333333</td>
      <td>0.280</td>
      <td>0.750000</td>
      <td>0.181818</td>
    </tr>
    <tr>
      <th>1345</th>
      <td>0.500</td>
      <td>0.571429</td>
      <td>4</td>
      <td>0.5</td>
      <td>0.396552</td>
      <td>3</td>
      <td>0.395157</td>
      <td>0.243437</td>
      <td>5</td>
      <td>0.666667</td>
      <td>0.888</td>
      <td>0.783333</td>
      <td>0.872727</td>
    </tr>
    <tr>
      <th>1347</th>
      <td>0.375</td>
      <td>0.328706</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.268473</td>
      <td>3</td>
      <td>0.571355</td>
      <td>0.437947</td>
      <td>3</td>
      <td>0.666667</td>
      <td>0.656</td>
      <td>0.283333</td>
      <td>0.609091</td>
    </tr>
    <tr>
      <th>1348</th>
      <td>0.375</td>
      <td>0.185401</td>
      <td>3</td>
      <td>0.5</td>
      <td>0.418719</td>
      <td>3</td>
      <td>0.391036</td>
      <td>0.247017</td>
      <td>4</td>
      <td>0.333333</td>
      <td>0.712</td>
      <td>0.400000</td>
      <td>0.672727</td>
    </tr>
    <tr>
      <th>1349</th>
      <td>0.375</td>
      <td>0.191223</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.299261</td>
      <td>3</td>
      <td>0.413189</td>
      <td>0.254773</td>
      <td>4</td>
      <td>0.333333</td>
      <td>0.744</td>
      <td>0.466667</td>
      <td>0.709091</td>
    </tr>
  </tbody>
</table>
<p>1260 rows × 13 columns</p>
</div>




```python
for c in numeric_feature[:-1]:
    print(c)
```

    Overall Qual
    Gr Liv Area
    Garage Cars
    Garage Area
    Total Bsmt SF
    1st Flr SF
    Full Bath
    Year Built
    Year Remod/Add
    Garage Yr Blt
    


```python
for column in numeric_feature[:-1]:
    test[column]=scaler.fit_transform(pd.DataFrame(test[column]))
```


```python
test
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Overall Qual</th>
      <th>Gr Liv Area</th>
      <th>Exter Qual</th>
      <th>Garage Cars</th>
      <th>Garage Area</th>
      <th>Kitchen Qual</th>
      <th>Total Bsmt SF</th>
      <th>1st Flr SF</th>
      <th>Bsmt Qual</th>
      <th>Full Bath</th>
      <th>Year Built</th>
      <th>Year Remod/Add</th>
      <th>Garage Yr Blt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.875</td>
      <td>0.266094</td>
      <td>4</td>
      <td>0.333333</td>
      <td>0.430843</td>
      <td>5</td>
      <td>0.274044</td>
      <td>0.297142</td>
      <td>5</td>
      <td>0.50</td>
      <td>0.977099</td>
      <td>0.950000</td>
      <td>0.973913</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.500</td>
      <td>0.128940</td>
      <td>3</td>
      <td>0.000000</td>
      <td>0.063593</td>
      <td>3</td>
      <td>0.153108</td>
      <td>0.143985</td>
      <td>3</td>
      <td>0.25</td>
      <td>0.526718</td>
      <td>0.000000</td>
      <td>0.460870</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.500</td>
      <td>0.222732</td>
      <td>4</td>
      <td>0.333333</td>
      <td>0.222576</td>
      <td>4</td>
      <td>0.098198</td>
      <td>0.077218</td>
      <td>4</td>
      <td>0.50</td>
      <td>0.923664</td>
      <td>0.833333</td>
      <td>0.913043</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.500</td>
      <td>0.388921</td>
      <td>4</td>
      <td>0.666667</td>
      <td>0.464229</td>
      <td>4</td>
      <td>0.165909</td>
      <td>0.160196</td>
      <td>4</td>
      <td>0.50</td>
      <td>0.954198</td>
      <td>0.900000</td>
      <td>0.947826</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.375</td>
      <td>0.120917</td>
      <td>3</td>
      <td>0.333333</td>
      <td>0.418124</td>
      <td>3</td>
      <td>0.146033</td>
      <td>0.135026</td>
      <td>3</td>
      <td>0.25</td>
      <td>0.679389</td>
      <td>0.300000</td>
      <td>0.834783</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1345</th>
      <td>0.750</td>
      <td>0.291309</td>
      <td>5</td>
      <td>0.666667</td>
      <td>0.488076</td>
      <td>5</td>
      <td>0.296278</td>
      <td>0.325299</td>
      <td>5</td>
      <td>0.50</td>
      <td>0.984733</td>
      <td>0.966667</td>
      <td>0.982609</td>
    </tr>
    <tr>
      <th>1346</th>
      <td>0.375</td>
      <td>0.096466</td>
      <td>3</td>
      <td>0.000000</td>
      <td>0.101749</td>
      <td>3</td>
      <td>0.124474</td>
      <td>0.107722</td>
      <td>3</td>
      <td>0.25</td>
      <td>0.648855</td>
      <td>0.233333</td>
      <td>0.600000</td>
    </tr>
    <tr>
      <th>1347</th>
      <td>0.250</td>
      <td>0.086724</td>
      <td>3</td>
      <td>0.333333</td>
      <td>0.101749</td>
      <td>3</td>
      <td>0.115883</td>
      <td>0.096843</td>
      <td>2</td>
      <td>0.25</td>
      <td>0.312977</td>
      <td>0.000000</td>
      <td>0.217391</td>
    </tr>
    <tr>
      <th>1348</th>
      <td>0.375</td>
      <td>0.195415</td>
      <td>3</td>
      <td>0.333333</td>
      <td>0.368839</td>
      <td>4</td>
      <td>0.211723</td>
      <td>0.218217</td>
      <td>5</td>
      <td>0.50</td>
      <td>0.954198</td>
      <td>0.916667</td>
      <td>0.947826</td>
    </tr>
    <tr>
      <th>1349</th>
      <td>0.375</td>
      <td>0.368672</td>
      <td>3</td>
      <td>0.333333</td>
      <td>0.317965</td>
      <td>3</td>
      <td>0.082365</td>
      <td>0.216510</td>
      <td>3</td>
      <td>0.50</td>
      <td>0.160305</td>
      <td>0.000000</td>
      <td>0.434783</td>
    </tr>
  </tbody>
</table>
<p>1350 rows × 13 columns</p>
</div>




```python
X_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1260 entries, 1 to 1349
    Data columns (total 13 columns):
     #   Column          Non-Null Count  Dtype   
    ---  ------          --------------  -----   
     0   Overall Qual    1260 non-null   float64 
     1   Gr Liv Area     1260 non-null   float64 
     2   Exter Qual      1260 non-null   category
     3   Garage Cars     1260 non-null   float64 
     4   Garage Area     1260 non-null   float64 
     5   Kitchen Qual    1260 non-null   category
     6   Total Bsmt SF   1260 non-null   float64 
     7   1st Flr SF      1260 non-null   float64 
     8   Bsmt Qual       1260 non-null   category
     9   Full Bath       1260 non-null   float64 
     10  Year Built      1260 non-null   float64 
     11  Year Remod/Add  1260 non-null   float64 
     12  Garage Yr Blt   1260 non-null   float64 
    dtypes: category(3), float64(10)
    memory usage: 112.6 KB
    


```python
y_train.shape
```




    (1260,)




```python
test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1350 entries, 0 to 1349
    Data columns (total 13 columns):
     #   Column          Non-Null Count  Dtype   
    ---  ------          --------------  -----   
     0   Overall Qual    1350 non-null   float64 
     1   Gr Liv Area     1350 non-null   float64 
     2   Exter Qual      1350 non-null   category
     3   Garage Cars     1350 non-null   float64 
     4   Garage Area     1350 non-null   float64 
     5   Kitchen Qual    1350 non-null   category
     6   Total Bsmt SF   1350 non-null   float64 
     7   1st Flr SF      1350 non-null   float64 
     8   Bsmt Qual       1350 non-null   category
     9   Full Bath       1350 non-null   float64 
     10  Year Built      1350 non-null   float64 
     11  Year Remod/Add  1350 non-null   float64 
     12  Garage Yr Blt   1350 non-null   float64 
    dtypes: category(3), float64(10)
    memory usage: 110.2 KB
    


```python
##############################################3
```


```python
#################################################################
```


```python
from sklearn.decomposition import PCA
```


```python
pca=PCA()
pca.fit(X_train)
```




    PCA()




```python
cumsum=np.cumsum(pca.explained_variance_ratio_)
```


```python
cumsum
```




    array([0.65021832, 0.76630864, 0.82645247, 0.87946305, 0.91898405,
           0.9432364 , 0.96276351, 0.97717839, 0.98390147, 0.9895272 ,
           0.99374822, 0.99757372, 1.        ])




```python
import matplotlib.pyplot as plt
plt.plot(range(1,14),cumsum)
```




    [<matplotlib.lines.Line2D at 0x29190c0e9d0>]




    
![png](output_69_1.png)
    



```python
pca=PCA(n_components=7)
```


```python
X_reduced=pca.fit_transform(X_train)
```


```python
X_reduced_df=pd.DataFrame(X_reduced,columns=['pc1','pc2','pc3','pc4','pc5','pc6','pc7'])
```


```python
X_reduced_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pc1</th>
      <th>pc2</th>
      <th>pc3</th>
      <th>pc4</th>
      <th>pc5</th>
      <th>pc6</th>
      <th>pc7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.566795</td>
      <td>-0.593651</td>
      <td>0.217845</td>
      <td>-0.296519</td>
      <td>-0.068165</td>
      <td>0.148673</td>
      <td>-0.055063</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.047498</td>
      <td>0.011730</td>
      <td>0.042917</td>
      <td>-0.264720</td>
      <td>-0.000380</td>
      <td>0.019978</td>
      <td>-0.187684</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.289535</td>
      <td>0.718037</td>
      <td>0.366544</td>
      <td>0.219293</td>
      <td>0.466756</td>
      <td>-0.207091</td>
      <td>0.035948</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.216708</td>
      <td>0.052702</td>
      <td>0.571878</td>
      <td>0.102810</td>
      <td>0.055012</td>
      <td>-0.485767</td>
      <td>0.174375</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.905562</td>
      <td>-0.027025</td>
      <td>-0.228917</td>
      <td>0.349602</td>
      <td>-0.319761</td>
      <td>0.093888</td>
      <td>-0.134978</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1255</th>
      <td>-1.082213</td>
      <td>0.132609</td>
      <td>0.142939</td>
      <td>-0.406494</td>
      <td>0.194536</td>
      <td>0.178521</td>
      <td>0.236763</td>
    </tr>
    <tr>
      <th>1256</th>
      <td>0.931564</td>
      <td>-1.262654</td>
      <td>-0.162993</td>
      <td>-0.514268</td>
      <td>-0.002758</td>
      <td>-0.082363</td>
      <td>0.277513</td>
    </tr>
    <tr>
      <th>1257</th>
      <td>-0.966545</td>
      <td>-0.001492</td>
      <td>-0.061759</td>
      <td>-0.067498</td>
      <td>-0.076969</td>
      <td>0.247191</td>
      <td>-0.061695</td>
    </tr>
    <tr>
      <th>1258</th>
      <td>-0.320085</td>
      <td>-0.721685</td>
      <td>0.259203</td>
      <td>-0.036642</td>
      <td>-0.034916</td>
      <td>-0.199717</td>
      <td>-0.150371</td>
    </tr>
    <tr>
      <th>1259</th>
      <td>-0.398331</td>
      <td>-0.680702</td>
      <td>0.389203</td>
      <td>-0.369832</td>
      <td>-0.024715</td>
      <td>0.071333</td>
      <td>-0.155089</td>
    </tr>
  </tbody>
</table>
<p>1260 rows × 7 columns</p>
</div>




```python
test.shape
```




    (1350, 13)




```python
test_reduced=pca.fit_transform(test)
```


```python
test_reduced_df=pd.DataFrame(test_reduced,columns=['pc1','pc2','pc3','pc4','pc5','pc6','pc7'])
```


```python
test_reduced_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pc1</th>
      <th>pc2</th>
      <th>pc3</th>
      <th>pc4</th>
      <th>pc5</th>
      <th>pc6</th>
      <th>pc7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.007741</td>
      <td>0.123091</td>
      <td>0.814743</td>
      <td>-0.498785</td>
      <td>-0.063434</td>
      <td>-0.182301</td>
      <td>-0.098489</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.207254</td>
      <td>0.046495</td>
      <td>0.134796</td>
      <td>-0.437195</td>
      <td>-0.221387</td>
      <td>-0.228483</td>
      <td>-0.081553</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.771843</td>
      <td>0.123844</td>
      <td>0.110384</td>
      <td>-0.672663</td>
      <td>0.260711</td>
      <td>-0.312531</td>
      <td>-0.055630</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.916708</td>
      <td>0.082463</td>
      <td>-0.049786</td>
      <td>-0.299178</td>
      <td>0.315822</td>
      <td>-0.381680</td>
      <td>-0.018928</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.963208</td>
      <td>-0.042470</td>
      <td>-0.020914</td>
      <td>-0.097716</td>
      <td>0.215314</td>
      <td>-0.371197</td>
      <td>-0.276003</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1345</th>
      <td>2.529369</td>
      <td>0.254421</td>
      <td>0.022926</td>
      <td>-0.747290</td>
      <td>-0.168572</td>
      <td>-0.386890</td>
      <td>-0.094327</td>
    </tr>
    <tr>
      <th>1346</th>
      <td>-1.127958</td>
      <td>0.023491</td>
      <td>0.138694</td>
      <td>-0.452234</td>
      <td>0.055933</td>
      <td>-0.196041</td>
      <td>-0.166117</td>
    </tr>
    <tr>
      <th>1347</th>
      <td>-1.820614</td>
      <td>0.743584</td>
      <td>-0.256326</td>
      <td>-0.151462</td>
      <td>-0.064902</td>
      <td>-0.393114</td>
      <td>-0.048182</td>
    </tr>
    <tr>
      <th>1348</th>
      <td>0.933118</td>
      <td>-0.729482</td>
      <td>1.128748</td>
      <td>-0.217538</td>
      <td>0.219166</td>
      <td>-0.112707</td>
      <td>-0.115002</td>
    </tr>
    <tr>
      <th>1349</th>
      <td>-1.149679</td>
      <td>0.055736</td>
      <td>-0.009143</td>
      <td>-0.049077</td>
      <td>-0.270582</td>
      <td>-0.368907</td>
      <td>0.277169</td>
    </tr>
  </tbody>
</table>
<p>1350 rows × 7 columns</p>
</div>




```python
from sklearn.ensemble import VotingRegressor
```


```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.model_selection import GridSearchCV
```


```python
best_models = [
    ('rf', RandomForestRegressor(min_samples_split=3, n_estimators=120)),
    ('GBR', GradientBoostingRegressor(learning_rate=0.09, loss='huber', n_estimators=130)),
    ('ET', ExtraTreesRegressor(n_estimators=90))
]

```


```python
voting_rg=VotingRegressor(best_models)
```


```python
voting_rg.fit(X_reduced_df,y_train)
```




    VotingRegressor(estimators=[('rf',
                                 RandomForestRegressor(min_samples_split=3,
                                                       n_estimators=120)),
                                ('GBR',
                                 GradientBoostingRegressor(learning_rate=0.09,
                                                           loss='huber',
                                                           n_estimators=130)),
                                ('ET', ExtraTreesRegressor(n_estimators=90))])




```python
from sklearn.model_selection import cross_val_score,KFold
```


```python
kfold=KFold(n_splits=5)
score=cross_val_score(voting_rg,X_reduced_df,y_train,cv=kfold,scoring='neg_mean_absolute_error')
-score
```




    array([19511.21304318, 18122.37489475, 18349.5391212 , 17674.89957131,
           19638.9621449 ])




```python
from sklearn.model_selection import GridSearchCV
```


```python
from sklearn.ensemble import RandomForestRegressor

# 모델 선언
rd_clf= RandomForestRegressor()

# 모델 학습
rd_clf.fit(X_train,y_train)
```




    RandomForestRegressor()




```python
rd_clf.get_params()
```




    {'bootstrap': True,
     'ccp_alpha': 0.0,
     'criterion': 'squared_error',
     'max_depth': None,
     'max_features': 'auto',
     'max_leaf_nodes': None,
     'max_samples': None,
     'min_impurity_decrease': 0.0,
     'min_samples_leaf': 1,
     'min_samples_split': 2,
     'min_weight_fraction_leaf': 0.0,
     'n_estimators': 100,
     'n_jobs': None,
     'oob_score': False,
     'random_state': None,
     'verbose': 0,
     'warm_start': False}




```python
score=cross_val_score(rd_clf,X_train,y_train,cv=kfold,scoring='neg_mean_absolute_error')
```


```python
-score
```




    array([17288.76583466, 16376.54354497, 15690.90381763, 16451.72479375,
           17898.9769709 ])




```python
y_pred=voting_rg.predict(test_reduced_df)
```


```python
y_pred = y_pred.astype('int')
```


```python
y_pred
```




    array([325263, 125177, 180895, ..., 104881, 235066, 135893])




```python
submission['target']=y_pred
```


```python
submission
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>325263</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>125177</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>180895</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>203277</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>136524</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1345</th>
      <td>1346</td>
      <td>344392</td>
    </tr>
    <tr>
      <th>1346</th>
      <td>1347</td>
      <td>120874</td>
    </tr>
    <tr>
      <th>1347</th>
      <td>1348</td>
      <td>104881</td>
    </tr>
    <tr>
      <th>1348</th>
      <td>1349</td>
      <td>235066</td>
    </tr>
    <tr>
      <th>1349</th>
      <td>1350</td>
      <td>135893</td>
    </tr>
  </tbody>
</table>
<p>1350 rows × 2 columns</p>
</div>




```python
submission.to_csv('house-submit.csv',index=False)
```


```python

```
