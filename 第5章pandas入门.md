# Day 3


```python
from pandas import Series,DataFrame   # series和dataframe用的多，引入本地命名空间中会更方便
```


```python
import pandas as pd
```


```python
import numpy as np
```

# 1.pandas的数据结构介绍

# Series
Series是一种类似于一维数组的对象，它是由一组数据(各种Numpy数据类型)以及一组与之相关的数据标签(即索引)组成。仅由一组数据可以产生最简单的Series：


```python
obj = Series([4,7,-5,3])
obj
```




    0    4
    1    7
    2   -5
    3    3
    dtype: int64



Series的字符串表现形式为:索引在左边，值在右边。由于我们没有为数据指定索引，于是会自动创建一个0-N-1(N为数据的长度)的整数型索引。你可以通过Series的valus和index属性获取其数组表示形式和索引对象：


```python
obj.values
```




    array([ 4,  7, -5,  3], dtype=int64)




```python
obj.index
```




    RangeIndex(start=0, stop=4, step=1)



通常，我们希望所创建的Series带有一个可以对各个数据点进行标记的索引：


```python
obj2 = Series([4,7,-5,3],index=['d','b','a','c'])
obj2
```




    d    4
    b    7
    a   -5
    c    3
    dtype: int64




```python
obj2.index
```




    Index(['d', 'b', 'a', 'c'], dtype='object')



与普通Numpy数组相比，你可以通过索引的方式选用Series中的单个或一组值：


```python
obj2['a']
```




    -5




```python
obj2['d'] = 6
```


```python
obj2[['c','a','d']]
```




    c    3
    a   -5
    d    6
    dtype: int64



NumPy数组运算(如根据布尔数组进行过滤、标量乘法、应用数组函数等)都会保留索引和值之间的链接：


```python
obj2
```




    d    6
    b    7
    a   -5
    c    3
    dtype: int64




```python
obj2[obj2>0]
```




    d    6
    b    7
    c    3
    dtype: int64




```python
obj2 * 2
```




    d    12
    b    14
    a   -10
    c     6
    dtype: int64




```python
np.exp(obj2)
```




    d     403.428793
    b    1096.633158
    a       0.006738
    c      20.085537
    dtype: float64



还可以将Series看成是一个字长的有序字典，因此它是索引值到数据值的一个映射。它可以用在许多原本需要字典参数的函数中：


```python
'b' in obj2
```




    True




```python
'e' in obj2
```




    False



如果数据被存放在一个python字典中，也可以直接通过这个字典来创建Series：


```python
sdata = {'Ohio':35000,'Texas':71000,'Oregon':16000,'Utah':5000}
obj3 = Series(sdata)
obj3
```




    Ohio      35000
    Texas     71000
    Oregon    16000
    Utah       5000
    dtype: int64



如果只传入一个字典，则结果Series中的索引就是原字典的键(有序排列)。


```python
states = ['California','Ohio','Oregon','Texas']
obj4 = Series(data=sdata,index=states)
obj4
```




    California        NaN
    Ohio          35000.0
    Oregon        16000.0
    Texas         71000.0
    dtype: float64



在这个例子中，sdata中跟states索引相匹配的那3个值会被找出来并放到相应的位置上，但由于'California'所对应的sdata值找不到，所有其结果为NaN(即'非数字'(not a number)，在pandas中，它用于表示缺失或NA值)。我将使用缺失(missing)或NA表示缺失数据。pandas的isnull和notnull函数可用于检测缺失数据：


```python
pd.isnull(obj4)
```




    California     True
    Ohio          False
    Oregon        False
    Texas         False
    dtype: bool




```python
pd.notnull(obj4)
```




    California    False
    Ohio           True
    Oregon         True
    Texas          True
    dtype: bool



Series也有类似的实例方法：


```python
obj4.isnull()
```




    California     True
    Ohio          False
    Oregon        False
    Texas         False
    dtype: bool



我们将在本章详细讲解如何处理缺失数据。
对于许多应用而言，Series最重要的一个功能是：它在算术运算中会自动对齐不同索引的数据。


```python
obj3
```




    Ohio      35000
    Texas     71000
    Oregon    16000
    Utah       5000
    dtype: int64




```python
obj4
```




    California        NaN
    Ohio          35000.0
    Oregon        16000.0
    Texas         71000.0
    dtype: float64




```python
obj3+obj4
```




    California         NaN
    Ohio           70000.0
    Oregon         32000.0
    Texas         142000.0
    Utah               NaN
    dtype: float64



数据对齐功能将在一个单独的主题中讲解。

Series对象本身及其索引都有一个name属性，该属性跟pandas其他的关键功能关系非常密切：


```python
obj4.name = 'population'
```


```python
obj4.index.name = 'state'
```


```python
obj4
```




    state
    California        NaN
    Ohio          35000.0
    Oregon        16000.0
    Texas         71000.0
    Name: population, dtype: float64



Series的索引可以通过赋值的方法就地修改：


```python
obj
```




    0    4
    1    7
    2   -5
    3    3
    dtype: int64




```python
obj.index = ['Bob','Steve','Jeff','Ryan']
obj
```




    Bob      4
    Steve    7
    Jeff    -5
    Ryan     3
    dtype: int64



# DataFrame
DataFrame是一个表格型的数据结构，它含有一组有序的列，每列可以是不同的值类型(数值、字符串、布尔值等)。DataFrame既有行索引和列索引，它可以被看做由Series组成的字典(共用一个索引)(即竖着的第一列)。跟其他类似的数据结构相比(如R的data.frame),DataFrame中面向行和面向列的操作基本上是平衡的。其实，DataFrame中的数据是以一个或多个二维块存放的(而不是列表、字典或别的一维数据结构)。

构建DataFrame的办法很多，最常用的一种是直接传入一个由等长列表或NumPy数组组成的字典：


```python
data = {'state':['Ohio','Ohio','Ohio','Nevada','Nevada'],'year':[2000,2001,2002,2001,2002],'pop':[1.5,1.7,3.6,2.4,2.9]}
frame = DataFrame(data)
frame
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
      <th>state</th>
      <th>year</th>
      <th>pop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ohio</td>
      <td>2000</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ohio</td>
      <td>2001</td>
      <td>1.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ohio</td>
      <td>2002</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Nevada</td>
      <td>2001</td>
      <td>2.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Nevada</td>
      <td>2002</td>
      <td>2.9</td>
    </tr>
  </tbody>
</table>
</div>



如果指定了列序列，则DataFrame的列就会按照指定的顺序进行排序：


```python
DataFrame(data,columns=['year','state','pop'])
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
      <th>year</th>
      <th>state</th>
      <th>pop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2000</td>
      <td>Ohio</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2001</td>
      <td>Ohio</td>
      <td>1.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2002</td>
      <td>Ohio</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2001</td>
      <td>Nevada</td>
      <td>2.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2002</td>
      <td>Nevada</td>
      <td>2.9</td>
    </tr>
  </tbody>
</table>
</div>



跟Series一样，如果传入的列在数据中找不到，就会产生NA值：


```python
frame2 = DataFrame(data=data,columns=['year','state','pop','debt'],index=['one','two','three','four','five'])
frame2
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
      <th>year</th>
      <th>state</th>
      <th>pop</th>
      <th>debt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>2000</td>
      <td>Ohio</td>
      <td>1.5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>two</th>
      <td>2001</td>
      <td>Ohio</td>
      <td>1.7</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>three</th>
      <td>2002</td>
      <td>Ohio</td>
      <td>3.6</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>four</th>
      <td>2001</td>
      <td>Nevada</td>
      <td>2.4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>five</th>
      <td>2002</td>
      <td>Nevada</td>
      <td>2.9</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame2.columns   # column有栏的意思
```




    Index(['year', 'state', 'pop', 'debt'], dtype='object')



通过类似字典标记的方式或属性的方法，可以将DataFrame的列获取为一个Series：


```python
frame2['state']     # 字典标记
```




    one        Ohio
    two        Ohio
    three      Ohio
    four     Nevada
    five     Nevada
    Name: state, dtype: object




```python
frame2.year   # 属性
```




    one      2000
    two      2001
    three    2002
    four     2001
    five     2002
    Name: year, dtype: int64



注意，返回的Series拥有原DataFrame详图的索引，且其name属性也已经被相应的设置好了。行也可以通过位置或名称的方式进行获取，比如用索引字段ix(后做详细介绍)


```python
frame2.loc['three']    # 注，书上的ix已不适用
```




    year     2002
    state    Ohio
    pop       3.6
    debt      NaN
    Name: three, dtype: object



列可以通过赋值的方式进行修改。例如，我们可以给那个空的"debt"列赋上一个标量值或者一组值


```python
frame2['debt'] = 16.5
frame2
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
      <th>year</th>
      <th>state</th>
      <th>pop</th>
      <th>debt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>2000</td>
      <td>Ohio</td>
      <td>1.5</td>
      <td>16.5</td>
    </tr>
    <tr>
      <th>two</th>
      <td>2001</td>
      <td>Ohio</td>
      <td>1.7</td>
      <td>16.5</td>
    </tr>
    <tr>
      <th>three</th>
      <td>2002</td>
      <td>Ohio</td>
      <td>3.6</td>
      <td>16.5</td>
    </tr>
    <tr>
      <th>four</th>
      <td>2001</td>
      <td>Nevada</td>
      <td>2.4</td>
      <td>16.5</td>
    </tr>
    <tr>
      <th>five</th>
      <td>2002</td>
      <td>Nevada</td>
      <td>2.9</td>
      <td>16.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame2['debt'] = np.arange(5.)
frame2
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
      <th>year</th>
      <th>state</th>
      <th>pop</th>
      <th>debt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>2000</td>
      <td>Ohio</td>
      <td>1.5</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>two</th>
      <td>2001</td>
      <td>Ohio</td>
      <td>1.7</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>three</th>
      <td>2002</td>
      <td>Ohio</td>
      <td>3.6</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>four</th>
      <td>2001</td>
      <td>Nevada</td>
      <td>2.4</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>five</th>
      <td>2002</td>
      <td>Nevada</td>
      <td>2.9</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>



将列表或数组赋值给某个列时，其长度必须跟DataFrame的长度相匹配。如果赋值的是一个Series，就会精确匹配DataFrame的索引，所有的空位都将ve填上缺失值：


```python
val = Series([-1.2,-1.5,-1.7],index=['two','four','five'])
```


```python
frame2['debt'] = val
frame2
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
      <th>year</th>
      <th>state</th>
      <th>pop</th>
      <th>debt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>2000</td>
      <td>Ohio</td>
      <td>1.5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>two</th>
      <td>2001</td>
      <td>Ohio</td>
      <td>1.7</td>
      <td>-1.2</td>
    </tr>
    <tr>
      <th>three</th>
      <td>2002</td>
      <td>Ohio</td>
      <td>3.6</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>four</th>
      <td>2001</td>
      <td>Nevada</td>
      <td>2.4</td>
      <td>-1.5</td>
    </tr>
    <tr>
      <th>five</th>
      <td>2002</td>
      <td>Nevada</td>
      <td>2.9</td>
      <td>-1.7</td>
    </tr>
  </tbody>
</table>
</div>



为不存在的列赋值会创建出一个新列。关键字del用于删除列：


```python
frame2['eastern'] = frame2.state == 'Ohio'
frame2
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
      <th>year</th>
      <th>state</th>
      <th>pop</th>
      <th>debt</th>
      <th>eastern</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>2000</td>
      <td>Ohio</td>
      <td>1.5</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
    <tr>
      <th>two</th>
      <td>2001</td>
      <td>Ohio</td>
      <td>1.7</td>
      <td>-1.2</td>
      <td>True</td>
    </tr>
    <tr>
      <th>three</th>
      <td>2002</td>
      <td>Ohio</td>
      <td>3.6</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
    <tr>
      <th>four</th>
      <td>2001</td>
      <td>Nevada</td>
      <td>2.4</td>
      <td>-1.5</td>
      <td>False</td>
    </tr>
    <tr>
      <th>five</th>
      <td>2002</td>
      <td>Nevada</td>
      <td>2.9</td>
      <td>-1.7</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
del frame2['eastern']
frame2.columns
```




    Index(['year', 'state', 'pop', 'debt'], dtype='object')



注意，通过索引方式返回的列是相应数据的视图而非副本。因此对返回的Series所做的任何就地修改会全反映到源DataFrame上。

另一种常见的数据形式是嵌套字典(也就是字典的字典)：


```python
pop = {'Nevada':{2001:2.4,2002:2.9},
       'Ohio':{2000:1.5,2001:1.7,2002:3.6}}
pop
```




    {'Nevada': {2001: 2.4, 2002: 2.9}, 'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}




```python
frame3 = DataFrame(pop)
frame3
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
      <th>Nevada</th>
      <th>Ohio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2001</th>
      <td>2.4</td>
      <td>1.7</td>
    </tr>
    <tr>
      <th>2002</th>
      <td>2.9</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>NaN</td>
      <td>1.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 当然你可以对结果进行转置
frame3.T
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
      <th>2001</th>
      <th>2002</th>
      <th>2000</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Nevada</th>
      <td>2.4</td>
      <td>2.9</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Ohio</th>
      <td>1.7</td>
      <td>3.6</td>
      <td>1.5</td>
    </tr>
  </tbody>
</table>
</div>



内层字典的键会被合并，排序以形成最终的索引。如果显示指定了索引，则步这样：


```python
DataFrame(pop,index=[2001,2002,2003])
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
      <th>Nevada</th>
      <th>Ohio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2001</th>
      <td>2.4</td>
      <td>1.7</td>
    </tr>
    <tr>
      <th>2002</th>
      <td>2.9</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>2003</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



和Series组成的字典差不多也是一样的用法：


```python
pdata = {'Ohio':frame3['Ohio'][:-1],    # 从头到尾，不含最后一个
        'Nevade':frame3['Nevada'][:2]}  # 第0 和 第1
DataFrame(pdata)
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
      <th>Ohio</th>
      <th>Nevade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2001</th>
      <td>1.7</td>
      <td>2.4</td>
    </tr>
    <tr>
      <th>2002</th>
      <td>3.6</td>
      <td>2.9</td>
    </tr>
  </tbody>
</table>
</div>



DataFrame构造函数所能接受的各种数据见p123

如果设置了DataFrame的index和columns的name属性，则这些信息也会被显示出来：


```python
frame3.index.name = 'year';frame3.columns.name = 'state'
frame3
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
      <th>state</th>
      <th>Nevada</th>
      <th>Ohio</th>
    </tr>
    <tr>
      <th>year</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2001</th>
      <td>2.4</td>
      <td>1.7</td>
    </tr>
    <tr>
      <th>2002</th>
      <td>2.9</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>NaN</td>
      <td>1.5</td>
    </tr>
  </tbody>
</table>
</div>



跟Series一样，values属性也会以二维ndarray的形式返回DataFrame的数据：


```python
frame3.values
```




    array([[2.4, 1.7],
           [2.9, 3.6],
           [nan, 1.5]])



如果DataFrame各列的数据类型不一样，则值数组的数据类型就会选用能兼容所有列的数据类型：


```python
frame2.values
```




    array([[2000, 'Ohio', 1.5, nan],
           [2001, 'Ohio', 1.7, -1.2],
           [2002, 'Ohio', 3.6, nan],
           [2001, 'Nevada', 2.4, -1.5],
           [2002, 'Nevada', 2.9, -1.7]], dtype=object)



# 索引对象
pandas的索引对象负责管理轴标签和其他元数据(比如轴名称等)。构建Series或DataFrame时，所用到的仍和数组或其他序列的标签都会被转换成一个Index：


```python
obj = Series(range(3),index=['a','b','c'])
obj
```




    a    0
    b    1
    c    2
    dtype: int64




```python
index = obj.index
index
```




    Index(['a', 'b', 'c'], dtype='object')




```python
index[1:]
```




    Index(['b', 'c'], dtype='object')




```python
# Index对象是一个不可修改的(immutable),因此用户不能对其进行修改
index[1] = 'd'
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[55], line 2
          1 # Index对象是一个不可修改的(immutable),因此用户不能对其进行修改
    ----> 2 index[1] = 'd'
    

    File E:\python\anaconda\lib\site-packages\pandas\core\indexes\base.py:5302, in Index.__setitem__(self, key, value)
       5300 @final
       5301 def __setitem__(self, key, value):
    -> 5302     raise TypeError("Index does not support mutable operations")
    

    TypeError: Index does not support mutable operations


不可修改性非常重要，因为这样才能使Index对象在多个数据结构之间安全共享：


```python
pd.Index(np.arange(3))
```


```python
index = pd.Index(np.arange(3))
index
```


```python
obj2 = Series([1.5,-2.5,0],index=index)
obj2
```


```python
obj2.index is index    # is与== 的区别是 is必须要是内存地址等，所以是本质上的等,说明在数据使用它时是共享一篇空间
```

pandas库内置的Index类见p125

除了长得像数组，Index的功能也类似一个固定大小的集合：


```python
frame3
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
      <th>state</th>
      <th>Nevada</th>
      <th>Ohio</th>
    </tr>
    <tr>
      <th>year</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2001</th>
      <td>2.4</td>
      <td>1.7</td>
    </tr>
    <tr>
      <th>2002</th>
      <td>2.9</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>NaN</td>
      <td>1.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame3.columns
```




    Index(['Nevada', 'Ohio'], dtype='object', name='state')




```python
'Ohio' in frame3.columns   # in就是查找在里面有没有
```




    True




```python
frame3.index
```




    Int64Index([2001, 2002, 2000], dtype='int64', name='year')




```python
2003 in frame3.index
```




    False



每个索引都有一些方法和属性，它们可用于设置逻辑并回答有关该索引所包含的数据的常见问题，Index的方法和属性见p126

# Day4

# 2.基本功能
在本节中，我将介绍操作Series和DataFrame中的数据的基本手段。后续章节后更加深入地挖掘pandas在数据分析和处理方面的功能。本书不是pandas库的详尽文档，主要关注的是最重要的功能，那些不大常用的功能(也就是那些更深奥的功能)需要自己探索。

# 重新索引
pandas对象的一个重要方法是reindex，其作用是创建一个适应新索引的新对象。以之前的一个简单示例来说：


```python
obj = Series([4.5,7.2,-5.3,3.6],index=['d','b','a','c'])
obj
```




    d    4.5
    b    7.2
    a   -5.3
    c    3.6
    dtype: float64



调用该Series的reindex将会根据新索引进行重排。如果某个索引值当前不存在，就引入缺失值：


```python
obj2 = obj.reindex(['a','b','c','d','e'])
obj2
```




    a   -5.3
    b    7.2
    c    3.6
    d    4.5
    e    NaN
    dtype: float64




```python
obj.reindex(['a','b','c','d','e'],fill_value=0)
```




    a   -5.3
    b    7.2
    c    3.6
    d    4.5
    e    0.0
    dtype: float64



对于时间序列这样的有序数据，重新索引时可能需要做一些插值处理。method选项即可达到此目的，例如，使用ffill可以实现前向值填充：


```python
obj3 = Series(['blue','purple','yellow'],index=[0,2,4])
obj3
```




    0      blue
    2    purple
    4    yellow
    dtype: object




```python
obj3.reindex(range(6),method='ffill')   # ffill向前值填充
```




    0      blue
    1      blue
    2    purple
    3    purple
    4    yellow
    5    yellow
    dtype: object



ffill或pad是前向填充(或搬运)值，bfill或backfill是后向填充(或搬运)值。其实我们有时需要比前向和后向填充更为精确的插值方式。

对于DataFrame，reindex可以修改(行)索引(即第一列)、列，或两个都改。任何仅传入一个序列，则会重新索引行


```python
frame = DataFrame(np.arange(9).reshape((3,3)),index=['a','c','d'],columns=['Ohio','Texas','California'])
frame
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
      <th>Ohio</th>
      <th>Texas</th>
      <th>California</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>c</th>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>d</th>
      <td>6</td>
      <td>7</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame2 = frame.reindex(['a','b','c','d'])
frame2
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
      <th>Ohio</th>
      <th>Texas</th>
      <th>California</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>c</th>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>d</th>
      <td>6.0</td>
      <td>7.0</td>
      <td>8.0</td>
    </tr>
  </tbody>
</table>
</div>



使用columns关键字即可重新索引列(第一行):


```python
states = ['Texas','Utah','California']
frame.reindex(columns=states)
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
      <th>Texas</th>
      <th>Utah</th>
      <th>California</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>c</th>
      <td>4</td>
      <td>NaN</td>
      <td>5</td>
    </tr>
    <tr>
      <th>d</th>
      <td>7</td>
      <td>NaN</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



也可以同时对行和列进行重新索引，而插值则只能按行应用(即轴0 )(即第一列)：


```python
frame.reindex(index=['a','b','c','d'],columns=states)  # 一加上method = 'ffill'就报错
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
      <th>Texas</th>
      <th>Utah</th>
      <th>California</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>c</th>
      <td>4.0</td>
      <td>NaN</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>d</th>
      <td>7.0</td>
      <td>NaN</td>
      <td>8.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame.loc(['a','b','c','d'],states)        # 复现不出来
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[70], line 1
    ----> 1 frame.loc(['a','b','c','d'],states)
    

    TypeError: _LocationIndexer.__call__() takes from 1 to 2 positional arguments but 3 were given


reindex函数的参数见p129，有index，method，fill_value等

# 丢弃指定轴上的项
丢弃某条轴上的一个或多个项很简单，只要有一个索引数组或列表即可。由于需要执行一些数据整理和集合逻辑，所以drop方法返回的是一个在指定轴上删除了指定值的新对象：


```python
obj = Series(np.arange(5.),index=['a','b','c','d','e'])
obj
```




    a    0.0
    b    1.0
    c    2.0
    d    3.0
    e    4.0
    dtype: float64




```python
new_obj = obj.drop('c')
new_obj
```




    a    0.0
    b    1.0
    d    3.0
    e    4.0
    dtype: float64




```python
obj.drop(['d','c'])
```




    a    0.0
    b    1.0
    e    4.0
    dtype: float64



对于DataFrame，可以删除任意轴上的索引值：


```python
data = DataFrame(np.arange(16).reshape((4,4)),
                index=['Ohio','Colorado','Utah','New York'],
                columns=['one','two','three','four'])
data
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
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ohio</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.drop(['Colorado','Ohio'])
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
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Utah</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.drop('two',axis=1)   # KeyError: "['two'] not found in axis"  若不加axis=1即会出现，1应该对应行，就是位置坐标第二个
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
      <th>one</th>
      <th>three</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ohio</th>
      <td>0</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <td>4</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>8</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>12</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.drop(['two','four'],axis=1)
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
      <th>one</th>
      <th>three</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ohio</th>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <td>4</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>8</td>
      <td>10</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>12</td>
      <td>14</td>
    </tr>
  </tbody>
</table>
</div>



# 索引、选取和过滤
Series索引(obj[...])的工作方式类似于Numpy数组的索引，只不过Series的索引值不只是整数，下为例子：


```python
obj = Series(np.arange(4.),index=['a','b','c','d'])
obj
```




    a    0.0
    b    1.0
    c    2.0
    d    3.0
    dtype: float64




```python
obj['b']
```




    1.0




```python
obj[1]
```




    1.0




```python
obj[2:4]
```




    c    2.0
    d    3.0
    dtype: float64




```python
obj[['b','a','d']]
```




    b    1.0
    a    0.0
    d    3.0
    dtype: float64




```python
obj[[1,3]]
```




    b    1.0
    d    3.0
    dtype: float64




```python
obj[obj<2]
```




    a    0.0
    b    1.0
    dtype: float64



利用标签的切片运算与普通的python切片运算不同，其末端边界是包含的：


```python
obj['b':'c']
```




    b    1.0
    c    2.0
    dtype: float64



设置的方法也很简单：


```python
obj['b':'c']=5
obj
```




    a    0.0
    b    5.0
    c    5.0
    d    3.0
    dtype: float64



###### 如你所见，对DataFrame进行索引其实就是获取一个或多个列：


```python
data = DataFrame(np.arange(16).reshape((4,4)),
                index=['Ohio','Colorado','Utah','New York'],
                columns=['one','two','three','four'])
data
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
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ohio</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
data['two']
```




    Ohio         1
    Colorado     5
    Utah         9
    New York    13
    Name: two, dtype: int32




```python
data[['three','one']]
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
      <th>three</th>
      <th>one</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ohio</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <td>6</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>10</td>
      <td>8</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>14</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>



这种索引方式有几个特殊情况。首先是通过切片或布尔型数组选取行：


```python
data[:2]  # 通过切片选取行
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
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ohio</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
data['three']>5   # 布尔型数组
```




    Ohio        False
    Colorado     True
    Utah         True
    New York     True
    Name: three, dtype: bool




```python
data[data['three']>5]   # 通过布尔型数组选取行
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
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Colorado</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>



这可能有些不合乎逻辑，但这种语法的确来源于实践。

另一种用法是通过布尔型DataFrame(比如下面这个由标量比较运算得出的)进行索引：


```python
data < 5
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
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ohio</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
data[data<5] = 0
data
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
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ohio</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <td>0</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>



这段代码的目的是使DataFrame在语法上更像ndarray

为了在DataFrame的行上进行标签索引，我引入了专门的索引字段ix(loc?)。它使你可以通过Numpy式的标记法以及轴标签从DataFrame中选取行和列子集。之前曾经提到过，这也是一种重新索引的简单手段(未复现)


```python
data.loc['Colorado',['two','three']]
```




    two      5
    three    6
    Name: Colorado, dtype: int32




```python
data.loc[['Colorado','Utah'],['four','one','two']]   # 书上第二个写的[3,0,1]，但会报错
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
      <th>four</th>
      <th>one</th>
      <th>two</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Colorado</th>
      <td>7</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>11</td>
      <td>8</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.loc['Utah']   # 选取第三行  书上用的data.ix[2]
```




    one       8
    two       9
    three    10
    four     11
    Name: Utah, dtype: int32




```python
data.loc[:'Utah','two']   # 切片到Utah，显示第二列的数据
```




    Ohio        0
    Colorado    5
    Utah        9
    Name: two, dtype: int32




```python
data.three>5
```




    Ohio        False
    Colorado     True
    Utah         True
    New York     True
    Name: three, dtype: bool




```python
data.loc[data.three>5,:'three']  # 书上写的  :3  会报错
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
      <th>one</th>
      <th>two</th>
      <th>three</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Colorado</th>
      <td>0</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
    </tr>
  </tbody>
</table>
</div>



也就是说，对pandas对象中的数据的选取和重排方式有很多。见p132表格。在使用层次化索引时还能用到一些别的办法(后面会讲到)  

# 算术运算和数据对齐
pandas最重要的一个功能是，它可以对不同索引的对象进行算术运算。在将对象相加时，如果存在不同的索引对，则结构的索引就是该索引对的并集。


```python
s1 = Series([7.3,-2.5,3.4,1.5],index=['a','c','d','e'])
s1
```




    a    7.3
    c   -2.5
    d    3.4
    e    1.5
    dtype: float64




```python
s2 = Series([-2.1,3.6,-1.5,4,3.1],index=['a','c','e','f','g'])
s2
```




    a   -2.1
    c    3.6
    e   -1.5
    f    4.0
    g    3.1
    dtype: float64




```python
s1 + s2
```




    a    5.2
    c    1.1
    d    NaN
    e    0.0
    f    NaN
    g    NaN
    dtype: float64



自动的数据对齐操作在不重叠的索引处引入了NA值。缺失值会在算术运算过程中传播。

对于DataFrame，对齐操作会同时发生在行和列上：


```python
df1 = DataFrame(np.arange(9.).reshape((3,3)),columns=list('bcd'),index=['Ohio','Texas','Colorado'])   # list('bcd')指将bcd强制转化成bcd
df1
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
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ohio</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <td>6.0</td>
      <td>7.0</td>
      <td>8.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2 = DataFrame(np.arange(12.).reshape((4,3)),columns=list('bde'),index=['Utah','Ohio','Texas','Oregon'])
df2
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
      <th>b</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Utah</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Ohio</th>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>6.0</td>
      <td>7.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>Oregon</th>
      <td>9.0</td>
      <td>10.0</td>
      <td>11.0</td>
    </tr>
  </tbody>
</table>
</div>



把它们相加后会返回一个新的DataFrame，其索引和列为原来的那两个DataFrame的并集：


```python
df1 + df2
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
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Colorado</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Ohio</th>
      <td>3.0</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Oregon</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>9.0</td>
      <td>NaN</td>
      <td>12.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



## 在算术方法中填充值

在对不同索引的对象进行算术运算时，你可能希望当一个对象中某个轴标签在另一个对象中找不到时补充一个特殊值(比如0):


```python
df1 = DataFrame(np.arange(12.).reshape((3,4)),columns=list('abcd')) 
df1
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>11.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2 = DataFrame(np.arange(20.).reshape((4,5)),columns=list('abcde')) 
df2
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.0</td>
      <td>6.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.0</td>
      <td>11.0</td>
      <td>12.0</td>
      <td>13.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15.0</td>
      <td>16.0</td>
      <td>17.0</td>
      <td>18.0</td>
      <td>19.0</td>
    </tr>
  </tbody>
</table>
</div>



将它们相加时，没有重叠的位置就会产生NA值：


```python
df1 + df2
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.0</td>
      <td>11.0</td>
      <td>13.0</td>
      <td>15.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>20.0</td>
      <td>22.0</td>
      <td>24.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



使用df1的add方法，传入df2以及一个fill_value参数：


```python
df1.add(df2,fill_value=0)
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.0</td>
      <td>11.0</td>
      <td>13.0</td>
      <td>15.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>20.0</td>
      <td>22.0</td>
      <td>24.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15.0</td>
      <td>16.0</td>
      <td>17.0</td>
      <td>18.0</td>
      <td>19.0</td>
    </tr>
  </tbody>
</table>
</div>



与此类似，在对Series或DataFrame重新索引时，也可以指定一个填充值：


```python
df1.reindex(columns=df2.columns,fill_value=0)
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>7.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>11.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



除了add(+)，还要sub(-),div(/),mul(*)

## DataFrame和Series之间的运算
跟numpy数组一样，DataFrame和Series之间算术运算也是有明确规定的。先来看一个具有启发性的例子，计算一个二维数组与其某行之间的差：


```python
arr = np.arange(12.).reshape((3,4))
arr
```




    array([[ 0.,  1.,  2.,  3.],
           [ 4.,  5.,  6.,  7.],
           [ 8.,  9., 10., 11.]])




```python
arr[0]
```




    array([0., 1., 2., 3.])




```python
arr - arr[0]
```




    array([[0., 0., 0., 0.],
           [4., 4., 4., 4.],
           [8., 8., 8., 8.]])



这就叫广播(broadcasting)，第12章将对此进行详细讲解。DataFrame和Series之间的运算也差不多如此：


```python
frame = DataFrame(np.arange(12.).reshape((4,3)),columns=list('bde'),index=['Utah','Ohio','Texas','Oregon'])
frame
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
      <th>b</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Utah</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Ohio</th>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>6.0</td>
      <td>7.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>Oregon</th>
      <td>9.0</td>
      <td>10.0</td>
      <td>11.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
series = frame.loc['Utah']
series
```




    b    0.0
    d    1.0
    e    2.0
    Name: Utah, dtype: float64



默认情况下，DataFrame和Series之间的算术运算会将Series的索引匹配到DataFrame的列，然后沿着行一直向下广播：


```python
frame - series
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
      <th>b</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Utah</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Ohio</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>6.0</td>
      <td>6.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>Oregon</th>
      <td>9.0</td>
      <td>9.0</td>
      <td>9.0</td>
    </tr>
  </tbody>
</table>
</div>



如果某个索引值在DataFrame的列或Series的索引中找不到，则参与运算的两个对象就会被重新索引以形成并集：


```python
series2 = Series(range(3),index=['b','e','f'])
series2
```




    b    0
    e    1
    f    2
    dtype: int64




```python
frame + series2
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
      <th>b</th>
      <th>d</th>
      <th>e</th>
      <th>f</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Utah</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Ohio</th>
      <td>3.0</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>6.0</td>
      <td>NaN</td>
      <td>9.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Oregon</th>
      <td>9.0</td>
      <td>NaN</td>
      <td>12.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



如果你希望匹配行且在列上广播，则必须使用算术运算方法。例如：


```python
frame
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
      <th>b</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Utah</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Ohio</th>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>6.0</td>
      <td>7.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>Oregon</th>
      <td>9.0</td>
      <td>10.0</td>
      <td>11.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
series3 = frame['d']
series3
```




    Utah       1.0
    Ohio       4.0
    Texas      7.0
    Oregon    10.0
    Name: d, dtype: float64




```python
frame.sub(series3,axis=0)
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
      <th>b</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Utah</th>
      <td>-1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Ohio</th>
      <td>-1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>-1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Oregon</th>
      <td>-1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



传入的轴号就是希望匹配的轴。在本例中，我们的目的是匹配DataFrame的行索引并进行广播

# 函数应用和映射
Numpy的ufuncs(元素级数组方法)也可用于操作pandas对象：


```python
frame = DataFrame(np.random.randn(4,3),columns=list('bde'),index=['Utah','Ohio','Texas','Oregon'])
frame
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
      <th>b</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Utah</th>
      <td>0.095913</td>
      <td>-0.112126</td>
      <td>-0.613737</td>
    </tr>
    <tr>
      <th>Ohio</th>
      <td>-1.682576</td>
      <td>-0.071856</td>
      <td>-1.148150</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>-0.045547</td>
      <td>1.126318</td>
      <td>0.979281</td>
    </tr>
    <tr>
      <th>Oregon</th>
      <td>0.122314</td>
      <td>-0.677103</td>
      <td>0.700761</td>
    </tr>
  </tbody>
</table>
</div>




```python
np.abs(frame)
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
      <th>b</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Utah</th>
      <td>0.095913</td>
      <td>0.112126</td>
      <td>0.613737</td>
    </tr>
    <tr>
      <th>Ohio</th>
      <td>1.682576</td>
      <td>0.071856</td>
      <td>1.148150</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>0.045547</td>
      <td>1.126318</td>
      <td>0.979281</td>
    </tr>
    <tr>
      <th>Oregon</th>
      <td>0.122314</td>
      <td>0.677103</td>
      <td>0.700761</td>
    </tr>
  </tbody>
</table>
</div>



另一个常见的操作是，将 函数 应用到由各列或行所形成的一维数组上。DataFrame的apply方法即可实现此功能：


```python
f = lambda x:x.max() - x.min()
```


```python
frame.apply(f,axis=0)    # 不输则默认axis=0 
```




    b    1.804890
    d    1.803421
    e    2.127431
    dtype: float64



许多最为常见的数组统计功能都被实现成DataFrame的方法(如sum和mean)，因此无需使用apply方法。



```python
frame.sum()
```




    b   -1.509896
    d    0.265233
    e   -0.081845
    dtype: float64



除标量值外，传递给apply的函数还可以返回由多个值组成的Series：


```python
def f(x):
    return Series([x.min(),x.max()],index=['min','max'])
```


```python
frame.apply(f)
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
      <th>b</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>min</th>
      <td>-1.682576</td>
      <td>-0.677103</td>
      <td>-1.148150</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.122314</td>
      <td>1.126318</td>
      <td>0.979281</td>
    </tr>
  </tbody>
</table>
</div>



此外，元素级的python函数也是可以用的。假如你想得到frame中各个浮点值的格式化字符串，使用applymap即可：


```python
format = lambda x: '%.2f'%x
```


```python
frame.applymap(format)
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
      <th>b</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Utah</th>
      <td>0.10</td>
      <td>-0.11</td>
      <td>-0.61</td>
    </tr>
    <tr>
      <th>Ohio</th>
      <td>-1.68</td>
      <td>-0.07</td>
      <td>-1.15</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>-0.05</td>
      <td>1.13</td>
      <td>0.98</td>
    </tr>
    <tr>
      <th>Oregon</th>
      <td>0.12</td>
      <td>-0.68</td>
      <td>0.70</td>
    </tr>
  </tbody>
</table>
</div>



之所以叫做applymap，是因为Series有一个用于应用元素级函数的map方法：


```python
frame['e']
```




    Utah     -0.613737
    Ohio     -1.148150
    Texas     0.979281
    Oregon    0.700761
    Name: e, dtype: float64




```python
frame['e'].map(format)
```




    Utah      -0.61
    Ohio      -1.15
    Texas      0.98
    Oregon     0.70
    Name: e, dtype: object



# 排序和排名
根据条件对数据集排序(sorting)也是一种重要的内置运算。要对行或列索引进行排序(按字典顺序),可使用sort_index方法，它将返回一个已排序的新对象：


```python
obj = Series(range(4),index=['d','a','b','c'])
obj
```




    d    0
    a    1
    b    2
    c    3
    dtype: int64




```python
obj.sort_index()
```




    a    1
    b    2
    c    3
    d    0
    dtype: int64



而对于DataFrame，则可以根据任意一个轴上的索引进行排序：


```python
frame = DataFrame(np.arange(8).reshape((2,4)),index=['three','one'],columns=['d','a','b','c'])
frame
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
      <th>d</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>three</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>one</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame.sort_index()
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
      <th>d</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>three</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame.sort_index(axis=1)
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>three</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>one</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



数据是默认按升序排列的，但也可以降序排序：


```python
frame.sort_index(axis=1,ascending=False)
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
      <th>d</th>
      <th>c</th>
      <th>b</th>
      <th>a</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>three</th>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>one</th>
      <td>4</td>
      <td>7</td>
      <td>6</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



若要按值对Series进行排序，可使用其order方法： ×
#### 上法不适用了，应用sort_values


```python
obj = Series([4,7,-3,2])
obj
```




    0    4
    1    7
    2   -3
    3    2
    dtype: int64




```python
obj.sort_values()
```




    2   -3
    3    2
    0    4
    1    7
    dtype: int64



在排序时，任何缺失值默认都会被放到Series的末尾


```python
obj = Series([4,np.nan,7,np.nan,-3,2])
obj
```




    0    4.0
    1    NaN
    2    7.0
    3    NaN
    4   -3.0
    5    2.0
    dtype: float64




```python
obj.sort_values()
```




    4   -3.0
    5    2.0
    0    4.0
    2    7.0
    1    NaN
    3    NaN
    dtype: float64



在DataFrame上，你可能希望根据一个或多个列中的值进行排序。将一个或多个列的名字传递给by选项即可达到该目的：


```python
frame = DataFrame({'b':[4,7,-3,2],'a':[0,1,0,1]})
frame
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
      <th>b</th>
      <th>a</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame.sort_values(by='b')  # 书上用的sort_index，现pandas库更新了,其实更容易理解了，因为sort_index应该是依据index来排，而value才是内容
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
      <th>b</th>
      <th>a</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>-3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



要根据多个列进行排序，传入名称的列表即可：


```python
frame.sort_values(by=['a','b'])
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
      <th>b</th>
      <th>a</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>-3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame.sort_values(by=['b','a'])  # 优先按前面那个排，前面那个等再比较后面那个
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
      <th>b</th>
      <th>a</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>-3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



排名(ranking)跟排序关系密切，且它会增设应该排名值(从1开始，一直到数组中有效数据的数量)。它跟numpy.argsort产生的间接排序索引差不多，只不过它可以根据某种规则破坏平级关系。接下来介绍Series和DataFrame的rank方法。默认情况下，rank是通过"为各组分配一个平均排名"的方式破坏平级关系的：


```python
obj = Series([7,-5,7,4,2,0,4])
obj
```




    0    7
    1   -5
    2    7
    3    4
    4    2
    5    0
    6    4
    dtype: int64




```python
obj.rank()   # 就是排名！！！右边写的数字就是obj在这个Series中的排名
```




    0    6.5
    1    1.0
    2    6.5
    3    4.5
    4    3.0
    5    2.0
    6    4.5
    dtype: float64



也可以根据值在原数据中出现的顺序给出排序： 
#### 类似于稳定排序


```python
obj.rank(method='first')
```




    0    6.0
    1    1.0
    2    7.0
    3    4.0
    4    3.0
    5    2.0
    6    5.0
    dtype: float64



当然， 你也可以按照降序进行排名：


```python
obj.rank(ascending=False,method='max')
```




    0    2.0
    1    7.0
    2    2.0
    3    4.0
    4    5.0
    5    6.0
    6    4.0
    dtype: float64



p141表列出了所以用于破坏平级关系的method选项。
如average,min,max,first
DataFrame可以在行或列上计算排名：


```python
frame = DataFrame({'b':[4.3,7,-3,2],'a':[0,1,0,1],'c':[-2,5,8,-2.5]})
frame
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
      <th>b</th>
      <th>a</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.3</td>
      <td>0</td>
      <td>-2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.0</td>
      <td>1</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-3.0</td>
      <td>0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.0</td>
      <td>1</td>
      <td>-2.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame.rank()
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
      <th>b</th>
      <th>a</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.0</td>
      <td>1.5</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>3.5</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>1.5</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.0</td>
      <td>3.5</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame.rank(axis=1)
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
      <th>b</th>
      <th>a</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



# 带有重复值的轴索引
到目前为止所介绍的范例都有唯一的轴标签(索引值)。虽然许多pandas函数(如reindex)都要求标签唯一，但这并不是强制性的。我们来看看下面这个简单的带有重复索引值的Series：


```python
obj = Series(range(5),index=['a','a','b','b','c'])
obj
```




    a    0
    a    1
    b    2
    b    3
    c    4
    dtype: int64



索引的is_unique属性可以告诉你它的值是否唯一：


```python
obj.index.is_unique
```




    False



对于带有重复值的索引，数据选取的行为将会有所不同。如果某个索引对应多个值，则返回一个Series；而对应单个值的，则返回一个标量值。


```python
obj['a']
```




    a    0
    a    1
    dtype: int64




```python
obj['c']
```




    4



对DataFrame的行进行索引时也是如此：


```python
df = DataFrame(np.random.randn(4,3),index=['a','a','b','b'])
df
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>0.438592</td>
      <td>-0.678557</td>
      <td>1.412111</td>
    </tr>
    <tr>
      <th>a</th>
      <td>-0.072805</td>
      <td>-0.235022</td>
      <td>1.482193</td>
    </tr>
    <tr>
      <th>b</th>
      <td>1.290504</td>
      <td>0.682057</td>
      <td>-0.366259</td>
    </tr>
    <tr>
      <th>b</th>
      <td>-0.114901</td>
      <td>-1.511325</td>
      <td>1.324367</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc['b']
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>b</th>
      <td>1.290504</td>
      <td>0.682057</td>
      <td>-0.366259</td>
    </tr>
    <tr>
      <th>b</th>
      <td>-0.114901</td>
      <td>-1.511325</td>
      <td>1.324367</td>
    </tr>
  </tbody>
</table>
</div>



# Day5
# 3.汇总和计算描述统计
pandas对象拥有一组常用的数学和统计方法。它们大部分都属于约简和汇总统计，用于从Series中提取单个值(如sum或mean)或从DataFrame的行或列中提取一个Series。跟对应的NumPY数组方法相比，它们都是基于没有缺失数据的假设而构建的，接下来看一个简单的DataFrame：


```python
df = DataFrame([[1.4,np.nan],[7.1,-4.5],[np.nan,np.nan],[0.75,-1.3]],index=['a','b','c','d'],columns=['one','two'])
df
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
      <th>one</th>
      <th>two</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1.40</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>b</th>
      <td>7.10</td>
      <td>-4.5</td>
    </tr>
    <tr>
      <th>c</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>d</th>
      <td>0.75</td>
      <td>-1.3</td>
    </tr>
  </tbody>
</table>
</div>



调用DataFrame的sum方法将会返回一个含有列小计的Series：


```python
df.sum()
```




    one    9.25
    two   -5.80
    dtype: float64




```python
df.sum(axis=1)   # 传入axis=1将会按行进行求和运算
```




    a    1.40
    b    2.60
    c    0.00
    d   -0.55
    dtype: float64



NA值将会自动被排除(书上c为NA，但我这显示是0)，除非整个切片(这里指的是行或列)都是NA。通过skipna选项可以禁用该功能：


```python
df.mean(axis=1,skipna=False)
```




    a      NaN
    b    1.300
    c      NaN
    d   -0.275
    dtype: float64



p143表列出了这些约简方法的常用选项

有些方法(如idxmax和idxmin)返回的是间接统计(比如达到最小值或最大值的索引)：


```python
df.idxmax()
```




    one    b
    two    d
    dtype: object




```python
df.idxmax(axis=1)
```




    a    one
    b    one
    c    NaN
    d    one
    dtype: object




```python
df.idxmin()
```




    one    d
    two    b
    dtype: object



另一些方法则是累计型的：


```python
df.cumsum()
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
      <th>one</th>
      <th>two</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1.40</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>b</th>
      <td>8.50</td>
      <td>-4.5</td>
    </tr>
    <tr>
      <th>c</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>d</th>
      <td>9.25</td>
      <td>-5.8</td>
    </tr>
  </tbody>
</table>
</div>



还有一种方法，它既不是约简型也不是累计型。descr就是一个例子，它用于一次性产生多个汇总统计：


```python
df.describe()
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
      <th>one</th>
      <th>two</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.083333</td>
      <td>-2.900000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.493685</td>
      <td>2.262742</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.750000</td>
      <td>-4.500000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.075000</td>
      <td>-3.700000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.400000</td>
      <td>-2.900000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.250000</td>
      <td>-2.100000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.100000</td>
      <td>-1.300000</td>
    </tr>
  </tbody>
</table>
</div>



对于非数值型数据，describe会产生另外一种汇总统计：


```python
obj = Series(['a','a','b','c']*4)
obj
```




    0     a
    1     a
    2     b
    3     c
    4     a
    5     a
    6     b
    7     c
    8     a
    9     a
    10    b
    11    c
    12    a
    13    a
    14    b
    15    c
    dtype: object




```python
obj.describe()
```




    count     16
    unique     3
    top        a
    freq       8
    dtype: object



p144-155表列出了所有与描述统计相关的方法。

# 相关系数与协方差
有些汇总统计(如相关系数和协方差)是通过参数对计算出来的。我们来看几个DataFrame，它们的数据来自Yahoo！Finance的股票价格和成交量：
#### pandas-datareader包是一个远程获取金融数据的Python工具，通过它可以方便获得下面公司和机构的数据：


```python
from pandas_datareader import data as web # pip install pandas-datareader
```


```python
all_data = {}
for ticker in ['AAPL','IBM','MSFT','GOOG']:
    all_data[ticker] = web.get_components_yahoo(ticker,'1/1/2000','1/1/2010')

price = DataFrame({tic:data['Adj Close']
                  for tic,data in all_data.iterite()})
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[178], line 3
          1 all_data = {}
          2 for ticker in ['AAPL','IBM','MSFT','GOOG']:
    ----> 3     all_data[ticker] = web.get_components_yahoo(ticker,'1/1/2000','1/1/2010')
          5 price = DataFrame({tic:data['Adj Close']
          6                   for tic,data in all_data.iterite()})
    

    TypeError: _get_data() takes 1 positional argument but 3 were given


#### 由于Yahoo！2017年被收购，已经无法获取书中案例数据，且近期无VPN，暂时先不学习该部分。 

# 唯一值、值计数以及成员资格

还有一类方法可以从一维Series的值中抽取信息。以下这个Series为例：


```python
obj = Series(['c','a','d','a','a','b','b','c','c'])
obj
```




    0    c
    1    a
    2    d
    3    a
    4    a
    5    b
    6    b
    7    c
    8    c
    dtype: object



第一个函数是unique，它可以得到Series中的唯一值数组：


```python
uniques = obj.unique()
uniques
```




    array(['c', 'a', 'd', 'b'], dtype=object)



返回的唯一值是未排序的，如果需要的话，可以对结构再次进行排序(uniques.sort())。value_counts用于计算一个Series中各值出现的频率：


```python
obj.value_counts()    # 返回一个Series，其索引为唯一值，其值为频率，按计数值降序排列
```




    c    3
    a    3
    b    2
    d    1
    dtype: int64



为了方便查看，结果Series是按值频率降序排列的。value_counts还是一个顶级pandas方法，可以用于任何数组或序列：


```python
obj.values   # 将Series中的内容以数组形式展现出来
```




    array(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'], dtype=object)




```python
pd.value_counts(obj.values,sort=False)
```




    c    3
    a    3
    d    1
    b    2
    dtype: int64



最后是isin，它用于判断矢量化集合的成员资格，可用于选取Series中或DataFrame列中数据的子集：


```python
mask = obj.isin(['b','c'])   # 检查obj中的各个值是否包含于[]
mask
```




    0     True
    1    False
    2    False
    3    False
    4    False
    5     True
    6     True
    7     True
    8     True
    dtype: bool




```python
obj[mask]
```




    0    c
    5    b
    6    b
    7    c
    8    c
    dtype: object



有时，你可能希望得到DataFrame中多个相关列的一张柱状图。例如：


```python
data = DataFrame({'Qu1':[1,3,4,3,4],
                 'Qu2':[2,3,1,2,3],
                'Qu3':[1,5,2,4,4]})
data
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
      <th>Qu1</th>
      <th>Qu2</th>
      <th>Qu3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>3</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



将pandas.value_counts传给该DataFrame的apply函数，就会出现：


```python
result = data.apply(pd.value_counts).fillna(0)  # 对各个列中的数字出现次数进行统计
result
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
      <th>Qu1</th>
      <th>Qu2</th>
      <th>Qu3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



# 4.处理缺失数据
缺失数据(missing data)在大部分数据分析应用中很常见。
pandas使用浮点值NaN(not a number)表示浮点和非浮点数组中的缺失数据。它只是一个便于被检测出来的标记而已：


```python
string_data = Series(['aardvark','artichoke',np.nan,'avocado'])
string_data
```




    0     aardvark
    1    artichoke
    2          NaN
    3      avocado
    dtype: object




```python
string_data.isnull()
```




    0    False
    1    False
    2     True
    3    False
    dtype: bool



python内置的None值也会被当作NA值处理：


```python
string_data[0]=None
```


```python
string_data.isnull()
```




    0     True
    1    False
    2     True
    3    False
    dtype: bool



还有一些NA处理方法见p149表

# 滤除缺失数据

过滤缺失数据的办法有很多种。纯手工操作永远是一个办法，但dropna可能会更实用一点。对于一个Series，dropna返回一个仅含有非空数据和索引值的Series：


```python
from numpy import nan as NA
```


```python
data = Series([1,NA,3.5,NA,7])
data
```




    0    1.0
    1    NaN
    2    3.5
    3    NaN
    4    7.0
    dtype: float64




```python
data.dropna()
```




    0    1.0
    2    3.5
    4    7.0
    dtype: float64



当然，也可以通过布尔型索引达到这个目的：


```python
data[data.notnull()]
```




    0    1.0
    2    3.5
    4    7.0
    dtype: float64



而对于DataFrame对象，事情就有点复杂了。你可能希望丢弃全NA或含有NA的行或列。dropna默认丢弃任何含有缺失值的行：


```python
data = DataFrame([[1.,6.5,3.],[1.,NA,NA],[NA,NA,NA],[NA,6.5,3.]])
data
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>6.5</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>6.5</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.dropna()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>6.5</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



传入how = 'all'将只丢弃全为NA的那些行：


```python
data.dropna(how='all')
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>6.5</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>6.5</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



要用这种方式丢弃列，只需传入axis=1即可：  # 即按行来检查NaN，，<-这句话当放屁


```python
data[4] = NA   # DataFrmae默认选取的列
data
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>6.5</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>6.5</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.dropna(axis=1,how='all')
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>6.5</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>6.5</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



另一个过滤DataFrame行的问题涉及实践序列数据。假设你只想留下一部分观测数据，可以用thresh参数实现此目的：


```python
df = DataFrame(np.random.randn(7,3))
df
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.184127</td>
      <td>-1.208341</td>
      <td>-1.128798</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.116466</td>
      <td>-0.721804</td>
      <td>1.456721</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.185698</td>
      <td>-0.315483</td>
      <td>-1.546326</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.930952</td>
      <td>0.516175</td>
      <td>-0.536275</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.523340</td>
      <td>-1.120322</td>
      <td>1.222687</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-1.192023</td>
      <td>-0.551835</td>
      <td>-0.109421</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.846409</td>
      <td>2.564510</td>
      <td>-0.097065</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc[:4,1]=NA
df.loc[:2,2]=NA  # 利用标签切片末端是包含的
```


```python
df
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.184127</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.116466</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.185698</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.930952</td>
      <td>NaN</td>
      <td>-0.536275</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.523340</td>
      <td>NaN</td>
      <td>1.222687</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-1.192023</td>
      <td>-0.551835</td>
      <td>-0.109421</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.846409</td>
      <td>2.564510</td>
      <td>-0.097065</td>
    </tr>
  </tbody>
</table>
</div>



#### 这里的axis=0意思是一列一列检查的意思吗，怎么反了一样


```python
df.dropna(thresh=3)   # thresh参数用法是：保留至少有n个非NaN数据的行/列（用axis=0或1表示） =0是行角度看
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>-1.192023</td>
      <td>-0.551835</td>
      <td>-0.109421</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.846409</td>
      <td>2.564510</td>
      <td>-0.097065</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dropna(thresh=2) 
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>-1.930952</td>
      <td>NaN</td>
      <td>-0.536275</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.523340</td>
      <td>NaN</td>
      <td>1.222687</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-1.192023</td>
      <td>-0.551835</td>
      <td>-0.109421</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.846409</td>
      <td>2.564510</td>
      <td>-0.097065</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dropna(thresh=2,axis=1)  # =1是从列角度看的
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.184127</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.116466</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.185698</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.930952</td>
      <td>NaN</td>
      <td>-0.536275</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.523340</td>
      <td>NaN</td>
      <td>1.222687</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-1.192023</td>
      <td>-0.551835</td>
      <td>-0.109421</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.846409</td>
      <td>2.564510</td>
      <td>-0.097065</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dropna(thresh=4,axis=1)
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
      <th>0</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.184127</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.116466</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.185698</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.930952</td>
      <td>-0.536275</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.523340</td>
      <td>1.222687</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-1.192023</td>
      <td>-0.109421</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.846409</td>
      <td>-0.097065</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dropna(thresh=5,axis=1)
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
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.184127</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.116466</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.185698</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.930952</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.523340</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-1.192023</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.846409</td>
    </tr>
  </tbody>
</table>
</div>



# 填充缺失数据
你可能不想滤除缺失数据(有可能会丢弃跟它有关的其他数据)，而是希望通过其他方式填补那些"空洞"。对于大多数情况而言，fillna方法是最主要的函数。通过一个常数调用fillna就会将缺失值替换为那个常数值：


```python
df
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.184127</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.116466</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.185698</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.930952</td>
      <td>NaN</td>
      <td>-0.536275</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.523340</td>
      <td>NaN</td>
      <td>1.222687</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-1.192023</td>
      <td>-0.551835</td>
      <td>-0.109421</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.846409</td>
      <td>2.564510</td>
      <td>-0.097065</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.fillna(0)
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.184127</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.116466</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.185698</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.930952</td>
      <td>0.000000</td>
      <td>-0.536275</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.523340</td>
      <td>0.000000</td>
      <td>1.222687</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-1.192023</td>
      <td>-0.551835</td>
      <td>-0.109421</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.846409</td>
      <td>2.564510</td>
      <td>-0.097065</td>
    </tr>
  </tbody>
</table>
</div>



若是通过一个字典调用fillna，就可以实现对不同的列填充不同的值：


```python
df.fillna({1:0.5,3:-1})
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.184127</td>
      <td>0.500000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.116466</td>
      <td>0.500000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.185698</td>
      <td>0.500000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.930952</td>
      <td>0.500000</td>
      <td>-0.536275</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.523340</td>
      <td>0.500000</td>
      <td>1.222687</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-1.192023</td>
      <td>-0.551835</td>
      <td>-0.109421</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.846409</td>
      <td>2.564510</td>
      <td>-0.097065</td>
    </tr>
  </tbody>
</table>
</div>



fillna默认会返回新对象，但也可以对现有对象进行就地修改


```python
_=df.fillna(0,inplace=True)
```


```python
df
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.184127</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.116466</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.185698</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.930952</td>
      <td>0.000000</td>
      <td>-0.536275</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.523340</td>
      <td>0.000000</td>
      <td>1.222687</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-1.192023</td>
      <td>-0.551835</td>
      <td>-0.109421</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.846409</td>
      <td>2.564510</td>
      <td>-0.097065</td>
    </tr>
  </tbody>
</table>
</div>



对reindex有效的那些插值方法也可用于fillna：


```python
df = DataFrame(np.random.randn(6,3))
df
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.459690</td>
      <td>0.816203</td>
      <td>-0.207892</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.793340</td>
      <td>-0.397477</td>
      <td>-2.027453</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.009425</td>
      <td>0.668306</td>
      <td>-1.203108</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.618598</td>
      <td>0.246599</td>
      <td>-0.412937</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.199474</td>
      <td>-0.581405</td>
      <td>-0.141921</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.173434</td>
      <td>1.976637</td>
      <td>0.400192</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc[2:,1]=NA
df.loc[4:,2]=NA
df
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.459690</td>
      <td>0.816203</td>
      <td>-0.207892</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.793340</td>
      <td>-0.397477</td>
      <td>-2.027453</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.009425</td>
      <td>NaN</td>
      <td>-1.203108</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.618598</td>
      <td>NaN</td>
      <td>-0.412937</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.199474</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.173434</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.fillna(method='ffill')   # 按照前一个值填充
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.459690</td>
      <td>0.816203</td>
      <td>-0.207892</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.793340</td>
      <td>-0.397477</td>
      <td>-2.027453</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.009425</td>
      <td>-0.397477</td>
      <td>-1.203108</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.618598</td>
      <td>-0.397477</td>
      <td>-0.412937</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.199474</td>
      <td>-0.397477</td>
      <td>-0.412937</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.173434</td>
      <td>-0.397477</td>
      <td>-0.412937</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.fillna(method='ffill',limit=2)  # limit表示限制填充几个
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.459690</td>
      <td>0.816203</td>
      <td>-0.207892</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.793340</td>
      <td>-0.397477</td>
      <td>-2.027453</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.009425</td>
      <td>-0.397477</td>
      <td>-1.203108</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.618598</td>
      <td>-0.397477</td>
      <td>-0.412937</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.199474</td>
      <td>NaN</td>
      <td>-0.412937</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.173434</td>
      <td>NaN</td>
      <td>-0.412937</td>
    </tr>
  </tbody>
</table>
</div>



只要稍动脑子，你就可以利用fillna实现许多别的功能。比如说，你可以传入Series的平均值或中位数：


```python
data = Series([1,NA,3.5,NA,7])
data
```




    0    1.0
    1    NaN
    2    3.5
    3    NaN
    4    7.0
    dtype: float64




```python
data.fillna(data.mean())
```




    0    1.000000
    1    3.833333
    2    3.500000
    3    3.833333
    4    7.000000
    dtype: float64



fillna函数的参数见p152-153表

# 5.层次化索引
层次化索引(hierarchical indexing)是pandas的一项重要功能，它使你能在一个轴上拥有多个(两个以上)索引级别，抽象点说，它能使你以低维度形式处理高维度数据。我们先来看一个简单的例子：创建一个Series，并用一个由列表或数组组成的列表作为索引。


```python
data = Series(np.random.randn(10),
             index=[['a','a','a','b','b','b','c','c','d','d'],
                   [1,2,3,1,2,3,1,2,2,3]])
data
```




    a  1    1.323804
       2    0.049685
       3    0.206964
    b  1    0.921726
       2   -0.564306
       3    0.932567
    c  1    0.995530
       2   -1.824360
    d  2   -0.068809
       3   -0.570397
    dtype: float64



这就是带有MultiIndex索引的Series的格式化输出形式。索引之间的"间隔"表示"直接使用上面的标签"：


```python
data.index
```




    MultiIndex([('a', 1),
                ('a', 2),
                ('a', 3),
                ('b', 1),
                ('b', 2),
                ('b', 3),
                ('c', 1),
                ('c', 2),
                ('d', 2),
                ('d', 3)],
               )



对于一个层次化索引的对象，选取数据子集的操作很简单：


```python
data['b']
```




    1    0.921726
    2   -0.564306
    3    0.932567
    dtype: float64




```python
data['b':'c']
```




    b  1    0.921726
       2   -0.564306
       3    0.932567
    c  1    0.995530
       2   -1.824360
    dtype: float64




```python
data.loc[['b','d']]
```




    b  1    0.921726
       2   -0.564306
       3    0.932567
    d  2   -0.068809
       3   -0.570397
    dtype: float64



有时甚至还可以在"内层"中进行选取：


```python
data[:,2]  # 外层选取所有的层，内层选取标签为2的值
```




    a    0.049685
    b   -0.564306
    c   -1.824360
    d   -0.068809
    dtype: float64



层次化索引在数据重塑和基于分组的操作(如透视表生成)中 扮演着重要的角色。比如说，这段数据可以通过其unstack方法被重新安排到一个DataFrame中：


```python
data.unstack()
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
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1.323804</td>
      <td>0.049685</td>
      <td>0.206964</td>
    </tr>
    <tr>
      <th>b</th>
      <td>0.921726</td>
      <td>-0.564306</td>
      <td>0.932567</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.995530</td>
      <td>-1.824360</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>-0.068809</td>
      <td>-0.570397</td>
    </tr>
  </tbody>
</table>
</div>



unstack的逆运算是stack：


```python
data.unstack().stack()
```




    a  1    1.323804
       2    0.049685
       3    0.206964
    b  1    0.921726
       2   -0.564306
       3    0.932567
    c  1    0.995530
       2   -1.824360
    d  2   -0.068809
       3   -0.570397
    dtype: float64



stack和unstack将在第七章详细讲解。

对于一个DataFrame，每条轴都可以有分层索引：


```python
frame = DataFrame(np.arange(12).reshape((4,3)),
                 index=[['a','a','b','b'],[1,2,1,2]],
                 columns=[['Ohio','Ohio','Colorado'],['Green','Red','Green']])
frame
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="2" halign="left">Ohio</th>
      <th>Colorado</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>Green</th>
      <th>Red</th>
      <th>Green</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">a</th>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">b</th>
      <th>1</th>
      <td>6</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>



各层都可以有名字(可以是字符串，也可以是别的Python对象)。如果指定了名称，它们就会显示在控制台输出中(不要将索引名称跟轴标签混为一谈！)：


```python
frame.index.names = ['key1','key2']
frame.columns.names = ['state','color']
frame
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>state</th>
      <th colspan="2" halign="left">Ohio</th>
      <th>Colorado</th>
    </tr>
    <tr>
      <th></th>
      <th>color</th>
      <th>Green</th>
      <th>Red</th>
      <th>Green</th>
    </tr>
    <tr>
      <th>key1</th>
      <th>key2</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">a</th>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">b</th>
      <th>1</th>
      <td>6</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>



由于有了分部的列索引，因此可以轻松选取列分组：


```python
frame['Ohio']
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
      <th>color</th>
      <th>Green</th>
      <th>Red</th>
    </tr>
    <tr>
      <th>key1</th>
      <th>key2</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">a</th>
      <th>1</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">b</th>
      <th>1</th>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>



可以单独创建MultiIndex然后复用。上面那个DataFrame中的(分级的)列创建：


```python
MultiIndex.from_arrays([['Ohio','Ohio','Colorado'],['Green','Red','Green']],   # 啥玩意，没懂
                      names=['state','color'])
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[259], line 1
    ----> 1 MultiIndex.from_arrays([['Ohio','Ohio','Colorado'],['Green','Red','Green']],
          2                       names=['state','color'])
    

    NameError: name 'MultiIndex' is not defined


# 重排分级顺序
有时，你需要重新调整某条轴上各级别的顺序，或根据指定级别上的值对数据进行排序。swaplevel接受两个级别编号或名称，并返回一个互换了级别的新对象(但数据不会发生变化)：


```python
frame
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>state</th>
      <th colspan="2" halign="left">Ohio</th>
      <th>Colorado</th>
    </tr>
    <tr>
      <th></th>
      <th>color</th>
      <th>Green</th>
      <th>Red</th>
      <th>Green</th>
    </tr>
    <tr>
      <th>key1</th>
      <th>key2</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">a</th>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">b</th>
      <th>1</th>
      <td>6</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame.swaplevel('key1','key2')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>state</th>
      <th colspan="2" halign="left">Ohio</th>
      <th>Colorado</th>
    </tr>
    <tr>
      <th></th>
      <th>color</th>
      <th>Green</th>
      <th>Red</th>
      <th>Green</th>
    </tr>
    <tr>
      <th>key2</th>
      <th>key1</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <th>a</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <th>a</th>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <th>b</th>
      <td>6</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <th>b</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>



而sortleve则根据单个级别中的值对数据进行排序(稳定的)。交换级别时，常常也会用到sortlevel，这样最终结果就是有序的了：

在 pandas 在版本 0.17.0之后进行了更新，不再有sort，而是用 sort_values 替代

用法为：

df.sort_values([‘B’], ascending=True)

表示对 B 列进行排序，True 表示从小到大。


```python
frame.sort_values(['key2'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>state</th>
      <th colspan="2" halign="left">Ohio</th>
      <th>Colorado</th>
    </tr>
    <tr>
      <th></th>
      <th>color</th>
      <th>Green</th>
      <th>Red</th>
      <th>Green</th>
    </tr>
    <tr>
      <th>key1</th>
      <th>key2</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>b</th>
      <th>1</th>
      <td>6</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>a</th>
      <th>2</th>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>b</th>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame.swaplevel(0,1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>state</th>
      <th colspan="2" halign="left">Ohio</th>
      <th>Colorado</th>
    </tr>
    <tr>
      <th></th>
      <th>color</th>
      <th>Green</th>
      <th>Red</th>
      <th>Green</th>
    </tr>
    <tr>
      <th>key2</th>
      <th>key1</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <th>a</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <th>a</th>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <th>b</th>
      <td>6</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <th>b</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame.swaplevel(0,1).sort_values(['key2'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>state</th>
      <th colspan="2" halign="left">Ohio</th>
      <th>Colorado</th>
    </tr>
    <tr>
      <th></th>
      <th>color</th>
      <th>Green</th>
      <th>Red</th>
      <th>Green</th>
    </tr>
    <tr>
      <th>key2</th>
      <th>key1</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">1</th>
      <th>a</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>b</th>
      <td>6</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">2</th>
      <th>a</th>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>b</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>



# 根据级别汇总统计
许多对DataFrame和Series的描述和汇总统计都有一个level选项，它用于指定在某条轴上求和的级别。再以上面那个DataFrame为例，我们可以根据行或列上的级别来进行求和，如下所示：


```python
frame
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>state</th>
      <th colspan="2" halign="left">Ohio</th>
      <th>Colorado</th>
    </tr>
    <tr>
      <th></th>
      <th>color</th>
      <th>Green</th>
      <th>Red</th>
      <th>Green</th>
    </tr>
    <tr>
      <th>key1</th>
      <th>key2</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">a</th>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">b</th>
      <th>1</th>
      <td>6</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
# frame.sum(level='key2')  出现未来警告!
```

    C:\Users\10053_ouzdo63\AppData\Local\Temp\ipykernel_1188\2004046222.py:1: FutureWarning: Using the level keyword in DataFrame and Series aggregations is deprecated and will be removed in a future version. Use groupby instead. df.sum(level=1) should use df.groupby(level=1).sum().
      frame.sum(level='key2')
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>state</th>
      <th colspan="2" halign="left">Ohio</th>
      <th>Colorado</th>
    </tr>
    <tr>
      <th>color</th>
      <th>Green</th>
      <th>Red</th>
      <th>Green</th>
    </tr>
    <tr>
      <th>key2</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>8</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12</td>
      <td>14</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame.groupby(level=1).sum()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>state</th>
      <th colspan="2" halign="left">Ohio</th>
      <th>Colorado</th>
    </tr>
    <tr>
      <th>color</th>
      <th>Green</th>
      <th>Red</th>
      <th>Green</th>
    </tr>
    <tr>
      <th>key2</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>8</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12</td>
      <td>14</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>




```python
# frame.sum(level='color',axis=1) 也出现为未来警告
```

    C:\Users\10053_ouzdo63\AppData\Local\Temp\ipykernel_1188\1534194635.py:1: FutureWarning: Using the level keyword in DataFrame and Series aggregations is deprecated and will be removed in a future version. Use groupby instead. df.sum(level=1) should use df.groupby(level=1).sum().
      frame.sum(level='color',axis=1)
    




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
      <th>color</th>
      <th>Green</th>
      <th>Red</th>
    </tr>
    <tr>
      <th>key1</th>
      <th>key2</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">a</th>
      <th>1</th>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>4</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">b</th>
      <th>1</th>
      <td>14</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>



这其实是利用了pandas的groupby功能，稍后会对其进行详细讲解。

# 使用DataFrame的列
人们经常想要将DataFrame的一个或多个列当作行索引(第一列)来用，或者可能希望将行索引变成DataFrame的列。以下面这个DataFrame为例：


```python
frame = DataFrame({'a':range(7),'b':range(7,0,-1),'c':['one','one','one','two','two','two','two'],'d':[0,1,2,0,1,2,3]})
frame
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>7</td>
      <td>one</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>6</td>
      <td>one</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>5</td>
      <td>one</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>4</td>
      <td>two</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>3</td>
      <td>two</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>2</td>
      <td>two</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>1</td>
      <td>two</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



DataFrame的set_index函数会将其一个或多个列转换为行索引，并创建一个新的DataFrame：


```python
frame2 = frame.set_index(['c','d'])
frame2
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
      <th></th>
      <th>a</th>
      <th>b</th>
    </tr>
    <tr>
      <th>c</th>
      <th>d</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">one</th>
      <th>0</th>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">two</th>
      <th>0</th>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



默认情况下，那些列会从DataFrame中移除，但也可以将其保留下来:


```python
frame.set_index(['c','d'],drop=False)
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
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
    <tr>
      <th>c</th>
      <th>d</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">one</th>
      <th>0</th>
      <td>0</td>
      <td>7</td>
      <td>one</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>6</td>
      <td>one</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>5</td>
      <td>one</td>
      <td>2</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">two</th>
      <th>0</th>
      <td>3</td>
      <td>4</td>
      <td>two</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>3</td>
      <td>two</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>2</td>
      <td>two</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>1</td>
      <td>two</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



reset_index的功能和set_index刚好相反，层次化索引的级别会被转移到列里面：


```python
frame2.reset_index()
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
      <th>c</th>
      <th>d</th>
      <th>a</th>
      <th>b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>one</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>one</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>two</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>two</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>two</td>
      <td>2</td>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>two</td>
      <td>3</td>
      <td>6</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



# 6.其他有关pandas的话题
# 整数索引
操作由整数索引的pandas对象常常会让新手抓狂，因为它们跟内置的python数据结构(如列表和元组)在索引语义上有些不同。例如，你可能认为下面这段代码不会产生一个错误：


```python
ser = Series(np.arange(3.))
ser[-1]
```

在这种情况下，虽然pandas会"求助于"整数索引，但没有哪种方法能够既不引入任何bug又安全有效地解决该问题。这里，我们有一个含有0、1、2的索引，但是很难推断出用户想要什么(基于标签还是基于位置的索引)：


```python
ser
```




    0    0.0
    1    1.0
    2    2.0
    dtype: float64



相反，对于一个非整数索引，就没有这样的歧义：


```python
ser2 = Series(np.arange(3.),index=['a','b','c'])
ser2
```




    a    0.0
    b    1.0
    c    2.0
    dtype: float64




```python
ser2[-1]
```




    2.0



为了保持良好的一致性，如果你的轴索引含有索引器，那么根据整数进行数据选取的操作将总是 面向标签 。这也包括用ix进行切片：


```python
ser.loc[:1]  # 面向标签，所以切片包含末端
```




    0    0.0
    1    1.0
    dtype: float64



如果你需要可靠的、不考虑索引类型的、基于 位置 的索引，可以使用Series的iget_value方法和DataFrame的irow和icol方法
#### row表示行，col表示列。但irow方法和icol方法已经被淘汰，可以使用iloc[:, i]选取列，使用iloc[i, :]选取行

#### iget_value()函数已被iat[]函数替代，将原始代码改为：ser3.iat[2]


```python
ser3 = Series(range(3),index=[-5,1,3])
ser3
```




    -5    0
     1    1
     3    2
    dtype: int64




```python
ser3.iat[2]   # 第3个位置上的数，即按位置索引
```




    2




```python
frame = DataFrame(np.arange(6).reshape(3,2),index=[2,0,1])
frame
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
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame.iloc[0,:]  # 选取第1行
```




    0    0
    1    1
    Name: 2, dtype: int32




```python
frame.iloc[:,1]  # 选取第2列
```




    2    1
    0    3
    1    5
    Name: 1, dtype: int32



# 面板数据(Panel)  不学
