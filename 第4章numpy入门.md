{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "15d436af",
   "metadata": {},
   "source": [
    "# Day1\n",
    "# 1.NumPy的ndarray:一种多维数组对象\n",
    "# 创建ndarray"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0858fc5c",
   "metadata": {},
   "source": [
    "以一个列表的转换为例"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "576cd6ef",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "6df4c38d",
   "metadata": {},
   "outputs": [],
   "source": [
    "data1 = [6,7.5,8,0,1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "a7446ed4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([6. , 7.5, 8. , 0. , 1. ])"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr1 = np.array(data1)\n",
    "arr1"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "297eb653",
   "metadata": {},
   "source": [
    "嵌套序列（比如由一组等长列表组成的列表）将会被转换为一个多维数组"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "499328b9",
   "metadata": {},
   "outputs": [],
   "source": [
    "data2 = [[1,2,3,4],[5,6,7,8]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "bbe7859e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 2, 3, 4],\n",
       "       [5, 6, 7, 8]])"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr2 = np.array(data2)\n",
    "arr2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "71924674",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr2.ndim  # 返回数组维度"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "9c613f43",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(2, 4)"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr2.shape  # 表示各位维度大小的元组。返回的是一个元组"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "8e9a25d8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "dtype('float64')"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr1.dtype"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "a613789b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "dtype('int32')"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr2.dtype"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9e48afc8",
   "metadata": {},
   "source": [
    "除了np.array还有一些函数可以创建数组，如zeros和ones可以创建指定长度或形状全0或全1数组。\n",
    "empty可以创建一个没有任何具体值的数组。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "8c0c9b32",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.zeros(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "1e1896ea",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0., 0., 0., 0., 0., 0.],\n",
       "       [0., 0., 0., 0., 0., 0.],\n",
       "       [0., 0., 0., 0., 0., 0.]])"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.zeros((3,6))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "7b6d7da4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[1.28720595e-311, 2.47032823e-322],\n",
       "        [0.00000000e+000, 0.00000000e+000],\n",
       "        [0.00000000e+000, 8.60952352e-072]],\n",
       "\n",
       "       [[6.81137280e-091, 2.21378155e-052],\n",
       "        [3.60957848e+174, 1.16350629e+165],\n",
       "        [3.99910963e+252, 1.46030983e-319]]])"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.empty((2,3,2))   # 返回的是一些未初始化的垃圾值"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fe98550e",
   "metadata": {},
   "source": [
    "arange是[python内置函数range]的数组版"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "400c3d2f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.arange(15)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "4ecb1d1d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 2, 3],\n",
       "       [2, 3, 4]])"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.array([[1,2,3],[2,3,4]])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3f571581",
   "metadata": {},
   "source": [
    "# ndarray的数据类型\n",
    "dtype(数据类型)是一个特殊的对象，它含有ndarray将一块内存解释为特定数据类型所需的信息:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "17e3ea67",
   "metadata": {},
   "outputs": [],
   "source": [
    "arr1=np.array([1,2,3],dtype=np.float64)\n",
    "arr2=np.array([1,2,3],dtype=np.int32)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "405e7656",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "dtype('float64')"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr1.dtype"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "bfb801a5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "dtype('int32')"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr2.dtype"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "611114f7",
   "metadata": {},
   "source": [
    "可以使用ndarray的astype方法显式的转换其dtype\n",
    "下为例子"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "c1980086",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "dtype('int32')"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr=np.array([1,2,3,4,5])\n",
    "arr.dtype"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "fc0e5165",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 2, 3, 4, 5])"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "bc587154",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "dtype('float64')"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "float_arr = arr.astype(np.float64)\n",
    "float_arr.dtype"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "9d45899b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1., 2., 3., 4., 5.])"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "float_arr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "fd4ce010",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 2, 3, 4, 5])"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr # astype转换类型是创建了一个新的ndarray，需要用变量接收"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fe9a096e",
   "metadata": {},
   "source": [
    "例2，如果字符串数组表示的全是数字，则也可以用astype转换"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "c47b1c56",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1.2 , 2.11, 4.75])"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "numeric_string=np.array(['1.2','2.11','4.75'],dtype=np.string_)\n",
    "numeric_string.astype(float) # np.float64"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4bc6b417",
   "metadata": {},
   "source": [
    "强制转换时，astype（）括号里可以写另一个ndarray.dtype表示转换成他的类型，如："
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "5a6bc5e1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "int_array=np.arange(10)\n",
    "int_array"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "30b6f74e",
   "metadata": {},
   "outputs": [],
   "source": [
    "calibers = np.array([.22,.270,.414],dtype =np.float64)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "0febfbce",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "int_array.astype(calibers.dtype)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "714367df",
   "metadata": {},
   "source": [
    "也可以用简洁的类型代码来表示dtype，如：int32-i4  uint32-u4"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "e5a79f1f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 858993459, 1072902963],\n",
       "       [2920577761, 1073799495],\n",
       "       [         0, 1074987008]], dtype=uint32)"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "empty_uint32 = np.empty((3,2),dtype='u4')\n",
    "empty_uint32"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dd72932c",
   "metadata": {},
   "source": [
    "# 数组和标量之间的运算\n",
    "大小相等的数组之间的任何算术运算都会将运算应用到元素级"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "92f45f13",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1., 2., 3.],\n",
       "       [4., 5., 6.]])"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr = np.array([[1.,2.,3.],[4.,5.,6.]])\n",
    "arr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "f111f07b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 1.,  4.,  9.],\n",
       "       [16., 25., 36.]])"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr*arr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "a0028e2c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0., 0., 0.],\n",
       "       [0., 0., 0.]])"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr-arr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "5a4641bd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0.5, 1. , 1.5],\n",
       "       [2. , 2.5, 3. ]])"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr*0.5"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "4fc752a0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1.        , 0.5       , 0.33333333],\n",
       "       [0.25      , 0.2       , 0.16666667]])"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "1/arr   # 0.5 ，1这种叫做标量"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ecf1709f",
   "metadata": {},
   "source": [
    "不同大小的数组之间的运算叫做广播，第12章中详讲"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "465dfe7f",
   "metadata": {},
   "source": [
    "# 基本的索引和切片\n",
    "Numpy数组的索引内容很多，因为选取数据子集或单个元素的方式有很多。一维数组很简单，从表面上看，他跟python的列表的功能差不多"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "6bcfca1f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr = np.arange(10)\n",
    "arr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "c74d69b4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "5"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr[5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "c7f336b5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([5, 6, 7])"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr[5:8]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "d11df556",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 0,  1,  2,  3,  4, 12, 12, 12,  8,  9])"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr[5:8]=12\n",
    "arr"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "004fd735",
   "metadata": {},
   "source": [
    "可见，把一个标量赋值给一个切片时，该值会自动传播(即广播)到整个选区。\n",
    "### 此外，跟列表最重要的区别在于，数组切片是原始数组的视图。这意味着数据不会被复制，视图上\n",
    "### 任何修改都会直接反映到源数组上:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "17df5349",
   "metadata": {},
   "outputs": [],
   "source": [
    "arr_slice = arr[5:8]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "5a9b3aa2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([   0,    1,    2,    3,    4,   12, 1234,   12,    8,    9])"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr_slice[1]=1234\n",
    "arr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "5fbd206e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 0,  1,  2,  3,  4, 64, 64, 64,  8,  9])"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr_slice[:] = 64\n",
    "arr"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cde31708",
   "metadata": {},
   "source": [
    "Numpy用于处理大数据，所以不把数据复制来复制去\n",
    "waring：如果你想得到的是ndarray切片的一份副本而非视图，就需要显式的进行复制操作\n",
    "如： arr[5:8].copy()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bfa835ba",
   "metadata": {},
   "source": [
    "对于高维度数组中，如二维数组，各索引位置上的元素不再是标量而是一维数组:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "045bb476",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([7, 8, 9])"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])\n",
    "arr2d[2]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0b5987f9",
   "metadata": {},
   "source": [
    "要访问里面元素有两种方式"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "8fc16ddf",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "8"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 方式1\n",
    "arr2d[2][1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "7a22dec8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "8"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 方式2\n",
    "arr2d[2,1]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9d268ad9",
   "metadata": {},
   "source": [
    "在更高维度的数组中同理，如：在2×2×3的arr3d中："
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "53354131",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[ 1,  2,  3],\n",
       "        [ 4,  5,  6]],\n",
       "\n",
       "       [[ 7,  8,  9],\n",
       "        [10, 11, 12]]])"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr3d = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])\n",
    "arr3d"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "f5918ad9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 2, 3],\n",
       "       [4, 5, 6]])"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr3d[0]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6804b4e8",
   "metadata": {},
   "source": [
    "标量值和数组都可以赋值给arr3d[0]:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "9a188622",
   "metadata": {},
   "outputs": [],
   "source": [
    "old_values = arr3d[0].copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "3682e770",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[42, 42, 42],\n",
       "        [42, 42, 42]],\n",
       "\n",
       "       [[ 7,  8,  9],\n",
       "        [10, 11, 12]]])"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr3d[0] = 42\n",
    "arr3d"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "b8348fec",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[ 1,  2,  3],\n",
       "        [ 4,  5,  6]],\n",
       "\n",
       "       [[ 7,  8,  9],\n",
       "        [10, 11, 12]]])"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr3d[0] = old_values\n",
    "arr3d"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "4dd16b69",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([4, 5, 6])"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr3d[0,1]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0b9792ad",
   "metadata": {},
   "source": [
    "### 切片索引\n",
    "ndarray的切片语法跟python列表这样的一维对象差不多："
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "80956691",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 1,  2,  3,  4, 64])"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr[1:6]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "14d50546",
   "metadata": {},
   "source": [
    "高维度对象可以在一个或多个轴上进行切片，也可以跟整数索引混合使用。如："
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "58c5ef9f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 2, 3],\n",
       "       [4, 5, 6],\n",
       "       [7, 8, 9]])"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr2d"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "5cfe4619",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 2, 3],\n",
       "       [4, 5, 6]])"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr2d[:2]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "1a3c8ddd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[2, 3],\n",
       "       [5, 6]])"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 可以一次传入多个切片，就像传入多个索引那样\n",
    "arr2d[:2,1:]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "57237776",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([4, 5])"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 可以整数索引和切片混合\n",
    "arr2d[1,:2]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "61432e4d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([7])"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr2d[2,:1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "68852be1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1],\n",
       "       [4],\n",
       "       [7]])"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 注意，“只有冒号”表示选取整个轴\n",
    "arr2d[:,:1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "1c847ead",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 0, 0],\n",
       "       [4, 0, 0],\n",
       "       [7, 8, 9]])"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 当然，对切片表达式的赋值也会被扩散到整个选区\n",
    "arr2d[:2,1:]=0\n",
    "arr2d"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "17c5a474",
   "metadata": {},
   "source": [
    "# 布尔型索引\n",
    "假设我们有个存储数据的数组和一个存储名字的数组(含重复项)。在这里，我们使用numpy.random中的randn函数生成一些正态分布的随机数据"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "008e8aeb",
   "metadata": {},
   "outputs": [],
   "source": [
    "names = np.array(['Bob','Joe','Will','Bob','Will','Joe','Joe'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "5bb8ce90",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[-0.12137403, -0.21989713, -0.28924979, -0.93689261],\n",
       "       [-0.91754465,  1.16823752, -0.38574666, -0.21475166],\n",
       "       [ 0.17283649, -0.17393293,  0.93879546,  1.2980231 ],\n",
       "       [ 2.9162052 , -0.15326669,  1.17628485,  0.96175703],\n",
       "       [-0.86929264,  1.45663917,  0.76956501, -0.83683981],\n",
       "       [ 1.21612489, -0.10835012, -1.74800688, -0.36114468],\n",
       "       [ 0.07558196,  1.11354748, -1.08176614,  0.40204156]])"
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = np.random.randn(7,4)\n",
    "data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "a369a5e6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ True, False, False,  True, False, False, False])"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 假设每个名字都对应data数组中的一行，而我们想选出对应名字\"Bob\"的所有行\n",
    "# 跟算术运算一样，数组的比较运算(==)也是矢量化的，因此可以通过比较运算得到一个布尔数组\n",
    "names == 'Bob'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "61203acb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[-0.12137403, -0.21989713, -0.28924979, -0.93689261],\n",
       "       [ 2.9162052 , -0.15326669,  1.17628485,  0.96175703]])"
      ]
     },
     "execution_count": 61,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 这个布尔数组可以用于索引，当然布尔数组的长度必须与被索引的轴长度一致\n",
    "data[names == 'Bob']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "9eb888d7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[-0.28924979, -0.93689261],\n",
       "       [ 1.17628485,  0.96175703]])"
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data[names == 'Bob',2:]   # 注意，这里和上面二位数组切片不一样，这里2:是直接在选中所有轴的情况下进行切"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "05ae5e28",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([-0.93689261,  0.96175703])"
      ]
     },
     "execution_count": 63,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data[names == 'Bob',3]   # 注意，这里和上面二位数组切片不一样，这里2:是直接在选中所有轴的情况下进行切"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "55966390",
   "metadata": {},
   "source": [
    "要选择除'Bob'外的其他值，既可以使用 != ，也可以通过负号 - 对条件进行否定"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "ef3e509c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([False,  True,  True, False,  True,  True,  True])"
      ]
     },
     "execution_count": 64,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 使用！=\n",
    "names != 'Bob'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "e8f1c8ee",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[-0.91754465,  1.16823752, -0.38574666, -0.21475166],\n",
       "       [ 0.17283649, -0.17393293,  0.93879546,  1.2980231 ],\n",
       "       [-0.86929264,  1.45663917,  0.76956501, -0.83683981],\n",
       "       [ 1.21612489, -0.10835012, -1.74800688, -0.36114468],\n",
       "       [ 0.07558196,  1.11354748, -1.08176614,  0.40204156]])"
      ]
     },
     "execution_count": 65,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 方式2 使用负号  ？？？变成~了  -用了报错\n",
    "data[~(names == 'Bob')]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "8ef1a00b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([False,  True,  True, False,  True,  True,  True])"
      ]
     },
     "execution_count": 66,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "~(names == 'Bob')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d7342b9d",
   "metadata": {},
   "source": [
    "选取这三个名字中的两个需要组合应用多个布尔条件，使用&（和），|（或）之类的布尔算术运算符即可"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "f53c5f1f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ True, False,  True,  True,  True, False, False])"
      ]
     },
     "execution_count": 67,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mask = (names == 'Bob')|(names == 'Will')\n",
    "mask"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "be542cb5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[-0.12137403, -0.21989713, -0.28924979, -0.93689261],\n",
       "       [ 0.17283649, -0.17393293,  0.93879546,  1.2980231 ],\n",
       "       [ 2.9162052 , -0.15326669,  1.17628485,  0.96175703],\n",
       "       [-0.86929264,  1.45663917,  0.76956501, -0.83683981]])"
      ]
     },
     "execution_count": 68,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data[mask]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0eb0a935",
   "metadata": {},
   "source": [
    "通过布尔型索引选取数组中的数据，将总是创建数据的副本，即使返回一模一样的数组也是如此\n",
    "\n",
    "通过布尔型数组设置值是一种常用的手段，为了将data中所有的负值都设为0，我们只需要："
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "1fb2a976",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0.        , 0.        , 0.        , 0.        ],\n",
       "       [0.        , 1.16823752, 0.        , 0.        ],\n",
       "       [0.17283649, 0.        , 0.93879546, 1.2980231 ],\n",
       "       [2.9162052 , 0.        , 1.17628485, 0.96175703],\n",
       "       [0.        , 1.45663917, 0.76956501, 0.        ],\n",
       "       [1.21612489, 0.        , 0.        , 0.        ],\n",
       "       [0.07558196, 1.11354748, 0.        , 0.40204156]])"
      ]
     },
     "execution_count": 69,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data[data<0]=0\n",
    "data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "2ff386e3",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[7.        , 7.        , 7.        , 7.        ],\n",
       "       [0.        , 1.16823752, 0.        , 0.        ],\n",
       "       [7.        , 7.        , 7.        , 7.        ],\n",
       "       [7.        , 7.        , 7.        , 7.        ],\n",
       "       [7.        , 7.        , 7.        , 7.        ],\n",
       "       [1.21612489, 0.        , 0.        , 0.        ],\n",
       "       [0.07558196, 1.11354748, 0.        , 0.40204156]])"
      ]
     },
     "execution_count": 70,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 也可以通过一维布尔数组设置整行或整列的值\n",
    "data[names != 'Joe'] = 7\n",
    "data"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4354e710",
   "metadata": {},
   "source": [
    "# Day 2\n",
    "# 花式索引\n",
    "花式索引(Fancy indexing)是一个NumPy术语，它指的是利用整数数组进行索引。假设我们有一个8×4数组："
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "3eb5e6eb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1.28720507e-311, 1.06224114e-321, 0.00000000e+000,\n",
       "        0.00000000e+000],\n",
       "       [2.37663529e-312, 5.02034658e+175, 2.25285651e+184,\n",
       "        1.80260843e+185],\n",
       "       [1.47791958e-075, 2.19508539e-056, 6.44357065e-067,\n",
       "        7.75582049e-144],\n",
       "       [3.59751658e+252, 1.46901661e+179, 8.37404147e+242,\n",
       "        2.59027926e-144],\n",
       "       [3.80985069e+180, 1.14428494e+243, 2.59027907e-144,\n",
       "        7.79952704e-143],\n",
       "       [3.42166000e-032, 1.54728069e-075, 1.44071214e+160,\n",
       "        1.79075917e+160],\n",
       "       [2.59027856e-144, 2.59903818e-144, 6.19410688e-091,\n",
       "        3.51046162e-033],\n",
       "       [2.58082100e-057, 1.11475752e+261, 1.16318408e-028,\n",
       "        2.97707521e+296]])"
      ]
     },
     "execution_count": 73,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr = np.empty((8,4))\n",
    "arr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "id": "c9bcc1c8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0., 0., 0., 0.],\n",
       "       [1., 1., 1., 1.],\n",
       "       [2., 2., 2., 2.],\n",
       "       [3., 3., 3., 3.],\n",
       "       [4., 4., 4., 4.],\n",
       "       [5., 5., 5., 5.],\n",
       "       [6., 6., 6., 6.],\n",
       "       [7., 7., 7., 7.]])"
      ]
     },
     "execution_count": 74,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "for i in range(8):\n",
    "    arr[i]=i\n",
    "arr"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d254ea51",
   "metadata": {},
   "source": [
    "想要以特定顺序选取行子集，只需传入一个用于指定顺序的整数列表或ndarray即可："
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "2b5d83e1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[4., 4., 4., 4.],\n",
       "       [3., 3., 3., 3.],\n",
       "       [0., 0., 0., 0.],\n",
       "       [6., 6., 6., 6.]])"
      ]
     },
     "execution_count": 75,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr[[4,3,0,6]]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9128fcaa",
   "metadata": {},
   "source": [
    "同列表一样，使用负数索引会从末尾开始选取行"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "dda4c03b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[5., 5., 5., 5.],\n",
       "       [3., 3., 3., 3.],\n",
       "       [1., 1., 1., 1.]])"
      ]
     },
     "execution_count": 76,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr[[-3,-5,-7]]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1a7e31dc",
   "metadata": {},
   "source": [
    "有点不符预期直觉的是，一次传入多个索引数组返回的是一个以为数组，其中的元素对应各个索引元组:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "id": "111a1cfc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 0,  1,  2,  3],\n",
       "       [ 4,  5,  6,  7],\n",
       "       [ 8,  9, 10, 11],\n",
       "       [12, 13, 14, 15],\n",
       "       [16, 17, 18, 19],\n",
       "       [20, 21, 22, 23],\n",
       "       [24, 25, 26, 27],\n",
       "       [28, 29, 30, 31]])"
      ]
     },
     "execution_count": 78,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr = np.arange(32).reshape((8,4))\n",
    "arr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "id": "a2b8aaa6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 4, 23, 29, 10])"
      ]
     },
     "execution_count": 79,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr[[1,5,7,2],[0,3,1,2]]   # 即分别对应元素(1,0)......"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "44ceed56",
   "metadata": {},
   "source": [
    "想要得到预计的结果(选出对应行对应列的矩阵)，有两种方法"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "id": "de07f013",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 4,  7,  5,  6],\n",
       "       [20, 23, 21, 22],\n",
       "       [28, 31, 29, 30],\n",
       "       [ 8, 11,  9, 10]])"
      ]
     },
     "execution_count": 84,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 方法1\n",
    "arr[[1,5,7,2]][:,[0,3,1,2]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "id": "2beb6d20",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 4,  7,  5,  6],\n",
       "       [20, 23, 21, 22],\n",
       "       [28, 31, 29, 30],\n",
       "       [ 8, 11,  9, 10]])"
      ]
     },
     "execution_count": 85,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 方式2   使用np.ix_函数，该函数作用是将两个一维整数数组转换为一个用于选取方形区域的索引器\n",
    "arr[np.ix_([1,5,7,2],[0,3,1,2])]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e337c56e",
   "metadata": {},
   "source": [
    "记住，花式索引跟切片不一样，它总是将数据赋值到新数组中\n",
    "# 数组转置和轴对换"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a40dbbe0",
   "metadata": {},
   "source": [
    "转置(transpose)是重塑的一种特殊形式，它返回的是源数据的视图(不会进行任何复制操作)。瘫坐不但有transpose方法，还要一个特殊的T属性:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "id": "d8ff10a4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 0,  1,  2,  3,  4],\n",
       "       [ 5,  6,  7,  8,  9],\n",
       "       [10, 11, 12, 13, 14]])"
      ]
     },
     "execution_count": 86,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr = np.arange(15).reshape(3,5)\n",
    "arr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "id": "c0e3335f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 0,  5, 10],\n",
       "       [ 1,  6, 11],\n",
       "       [ 2,  7, 12],\n",
       "       [ 3,  8, 13],\n",
       "       [ 4,  9, 14]])"
      ]
     },
     "execution_count": 87,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr.T"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "350803ec",
   "metadata": {},
   "source": [
    "在进行矩阵计算时，经常需要用到该操作，比如利用np.dot计算矩阵内积X转置X"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "id": "dc4bd987",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[-0.37837149,  0.06849045, -0.74938424],\n",
       "       [ 0.65189709,  0.38934392,  0.71622347],\n",
       "       [-0.6334776 , -1.3998728 , -1.46899174],\n",
       "       [-1.7952196 ,  0.25083811, -0.48422619],\n",
       "       [ 0.17942745, -0.27613012,  0.12383641],\n",
       "       [ 0.23774314, -0.04360816,  2.4257589 ]])"
      ]
     },
     "execution_count": 89,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr = np.random.randn(6,3)\n",
    "arr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "id": "fe15f7a6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[4.28095808, 0.60446304, 3.14924252],\n",
       "       [0.60446304, 2.25699276, 2.02249294],\n",
       "       [3.14924252, 2.02249294, 9.36660623]])"
      ]
     },
     "execution_count": 90,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.dot(arr.T,arr)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2ea61923",
   "metadata": {},
   "source": [
    "对于高维数组，transpose需要得到一个由轴编号组成的元组才能对这些轴进行转置(0,1,2)->(1,0,2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "id": "20fbddac",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[ 0,  1,  2,  3],\n",
       "        [ 4,  5,  6,  7]],\n",
       "\n",
       "       [[ 8,  9, 10, 11],\n",
       "        [12, 13, 14, 15]]])"
      ]
     },
     "execution_count": 105,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr = np.arange(16).reshape((2,2,4))\n",
    "arr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 110,
   "id": "af88e204",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[ 0,  1,  2,  3],\n",
       "        [ 8,  9, 10, 11]],\n",
       "\n",
       "       [[ 4,  5,  6,  7],\n",
       "        [12, 13, 14, 15]]])"
      ]
     },
     "execution_count": 110,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr.transpose((1,0,2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 107,
   "id": "e8f45609",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[ 0,  1,  2,  3],\n",
       "        [ 4,  5,  6,  7]],\n",
       "\n",
       "       [[ 8,  9, 10, 11],\n",
       "        [12, 13, 14, 15]]])"
      ]
     },
     "execution_count": 107,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr  # 迷惑，书上写transpose返回的是源数据的视图而非复制品，但为什么这里源程序没变"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3ffbd07b",
   "metadata": {},
   "source": [
    "对于上面的疑惑的解答:视图不等于原来的就也会变，只不过它也指向源数据，在它里面修改后源数据也会跟着变。即不会进行任何复制操作的意思！\n",
    "简单的转置可以使用.T,它其实就是进行轴对换而已，ndarray还有一个swapaxes方法，他需要接受一对轴编号"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 108,
   "id": "bae5f046",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[ 0,  4],\n",
       "        [ 1,  5],\n",
       "        [ 2,  6],\n",
       "        [ 3,  7]],\n",
       "\n",
       "       [[ 8, 12],\n",
       "        [ 9, 13],\n",
       "        [10, 14],\n",
       "        [11, 15]]])"
      ]
     },
     "execution_count": 108,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr.swapaxes(1,2)# 其实就是弱化版transpose ，括号里输入的是要交换的一堆轴号"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2da62a0d",
   "metadata": {},
   "source": [
    "同上，这里返回的是源数据的视图，但不会进行复制操作！即arr还是原来那样"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6aa2f187",
   "metadata": {},
   "source": [
    "# 2.通用函数：快速的元素级数组函数\n",
    "通用函数(即ufunc)是一种对ndarray中的数据执行元素级运算的函数。可以将其看做简单函数(接受一个或多个标量值，并产生一个或多个标量值)\n",
    "的矢量化包装器\n",
    "许多ufunc都是简单的元素级变体,如sqrt和exp:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 111,
   "id": "f02a962f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.        , 1.        , 1.41421356, 1.73205081, 2.        ,\n",
       "       2.23606798, 2.44948974, 2.64575131, 2.82842712, 3.        ])"
      ]
     },
     "execution_count": 111,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr = np.arange(10)\n",
    "np.sqrt(arr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "id": "f8047475",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1.00000000e+00, 2.71828183e+00, 7.38905610e+00, 2.00855369e+01,\n",
       "       5.45981500e+01, 1.48413159e+02, 4.03428793e+02, 1.09663316e+03,\n",
       "       2.98095799e+03, 8.10308393e+03])"
      ]
     },
     "execution_count": 112,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.exp(arr)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ac78f62a",
   "metadata": {},
   "source": [
    "以上是一元ufunc，另外一些(如add和maximum)接受2个数组(因此也叫二元(binary)ufunc)，并返回一个结果数组:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 115,
   "id": "c5635d45",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 0.94665965, -0.4754774 , -0.15001702, -1.49266338, -1.43456233,\n",
       "       -0.59047708,  0.6526553 , -2.49389592])"
      ]
     },
     "execution_count": 115,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x = np.random.randn(8)\n",
    "x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 116,
   "id": "ea2bf6bf",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([-0.0863673 ,  0.1911106 ,  0.53903421, -1.00703208, -1.20888113,\n",
       "        0.12637546, -0.14495472,  1.93305098])"
      ]
     },
     "execution_count": 116,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y = np.random.randn(8)\n",
    "y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 117,
   "id": "be459b15",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 0.94665965,  0.1911106 ,  0.53903421, -1.00703208, -1.20888113,\n",
       "        0.12637546,  0.6526553 ,  1.93305098])"
      ]
     },
     "execution_count": 117,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.maximum(x,y)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "44f25bf1",
   "metadata": {},
   "source": [
    "也有一些不常见的ufunc可以返回多个数组，如modf可以返回浮点数数组的小数和整数部分"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 118,
   "id": "088f299f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([11.14458519,  1.0766965 ,  1.19843104,  2.51754491,  5.51744021,\n",
       "       -4.93376601,  4.1171672 ])"
      ]
     },
     "execution_count": 118,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr = np.random.randn(7)*5\n",
    "arr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 119,
   "id": "cbf65e27",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(array([ 0.14458519,  0.0766965 ,  0.19843104,  0.51754491,  0.51744021,\n",
       "        -0.93376601,  0.1171672 ]),\n",
       " array([11.,  1.,  1.,  2.,  5., -4.,  4.]))"
      ]
     },
     "execution_count": 119,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.modf(arr)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3e52779b",
   "metadata": {},
   "source": [
    "还有一些一元和二元ufunc可以查表"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "99846fb8",
   "metadata": {},
   "source": [
    "# 3.利用数组进行数据分析\n",
    "Numpy数组可以将许多数据处理任务表述为简洁的数组表达式\n",
    "假设我们想在一组值(网格型)上计算函数sqrt(x方+y方)。np.meshgrid函数接受两个一维数组，并产生两个二维矩阵(对应于两个数组中所有的(x,y)对)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 121,
   "id": "8d7e708f",
   "metadata": {},
   "outputs": [],
   "source": [
    "points = np.arange(-5,5,0.01) # 1000个间隔相等的点"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 122,
   "id": "a20c2951",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[-5.  , -4.99, -4.98, ...,  4.97,  4.98,  4.99],\n",
       "       [-5.  , -4.99, -4.98, ...,  4.97,  4.98,  4.99],\n",
       "       [-5.  , -4.99, -4.98, ...,  4.97,  4.98,  4.99],\n",
       "       ...,\n",
       "       [-5.  , -4.99, -4.98, ...,  4.97,  4.98,  4.99],\n",
       "       [-5.  , -4.99, -4.98, ...,  4.97,  4.98,  4.99],\n",
       "       [-5.  , -4.99, -4.98, ...,  4.97,  4.98,  4.99]])"
      ]
     },
     "execution_count": 122,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "xs,ys = np.meshgrid(points,points)\n",
    "xs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 123,
   "id": "4b6c86de",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[-5.  , -5.  , -5.  , ..., -5.  , -5.  , -5.  ],\n",
       "       [-4.99, -4.99, -4.99, ..., -4.99, -4.99, -4.99],\n",
       "       [-4.98, -4.98, -4.98, ..., -4.98, -4.98, -4.98],\n",
       "       ...,\n",
       "       [ 4.97,  4.97,  4.97, ...,  4.97,  4.97,  4.97],\n",
       "       [ 4.98,  4.98,  4.98, ...,  4.98,  4.98,  4.98],\n",
       "       [ 4.99,  4.99,  4.99, ...,  4.99,  4.99,  4.99]])"
      ]
     },
     "execution_count": 123,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ys"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 124,
   "id": "c79f8bbd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[array([[-5.  , -4.99, -4.98, ...,  4.97,  4.98,  4.99],\n",
       "        [-5.  , -4.99, -4.98, ...,  4.97,  4.98,  4.99],\n",
       "        [-5.  , -4.99, -4.98, ...,  4.97,  4.98,  4.99],\n",
       "        ...,\n",
       "        [-5.  , -4.99, -4.98, ...,  4.97,  4.98,  4.99],\n",
       "        [-5.  , -4.99, -4.98, ...,  4.97,  4.98,  4.99],\n",
       "        [-5.  , -4.99, -4.98, ...,  4.97,  4.98,  4.99]]),\n",
       " array([[-5.  , -5.  , -5.  , ..., -5.  , -5.  , -5.  ],\n",
       "        [-4.99, -4.99, -4.99, ..., -4.99, -4.99, -4.99],\n",
       "        [-4.98, -4.98, -4.98, ..., -4.98, -4.98, -4.98],\n",
       "        ...,\n",
       "        [ 4.97,  4.97,  4.97, ...,  4.97,  4.97,  4.97],\n",
       "        [ 4.98,  4.98,  4.98, ...,  4.98,  4.98,  4.98],\n",
       "        [ 4.99,  4.99,  4.99, ...,  4.99,  4.99,  4.99]])]"
      ]
     },
     "execution_count": 124,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.meshgrid(points,points)  # meshgrid返回的是传入的两种一维数组的所有组合值,对应位置的x，y即一对组合"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "83e0a9cb",
   "metadata": {},
   "source": [
    "现在，对该函数的求值运算只要把这两个数组当作两个[浮点数]编写表达式即可"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 125,
   "id": "4046fbab",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 126,
   "id": "7ded39e1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[7.07106781, 7.06400028, 7.05693985, ..., 7.04988652, 7.05693985,\n",
       "        7.06400028],\n",
       "       [7.06400028, 7.05692568, 7.04985815, ..., 7.04279774, 7.04985815,\n",
       "        7.05692568],\n",
       "       [7.05693985, 7.04985815, 7.04278354, ..., 7.03571603, 7.04278354,\n",
       "        7.04985815],\n",
       "       ...,\n",
       "       [7.04988652, 7.04279774, 7.03571603, ..., 7.0286414 , 7.03571603,\n",
       "        7.04279774],\n",
       "       [7.05693985, 7.04985815, 7.04278354, ..., 7.03571603, 7.04278354,\n",
       "        7.04985815],\n",
       "       [7.06400028, 7.05692568, 7.04985815, ..., 7.04279774, 7.04985815,\n",
       "        7.05692568]])"
      ]
     },
     "execution_count": 126,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "z = np.sqrt(xs**2+ys**2)\n",
    "z"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 130,
   "id": "359b15ea",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, 'Image plot of $\\\\sqrt{x^2+y^2}$ for a grid of values')"
      ]
     },
     "execution_count": 130,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAecAAAG9CAYAAAAvGL7FAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy88F64QAAAACXBIWXMAAA9hAAAPYQGoP6dpAACwTUlEQVR4nO29ebheRZXvv06mk4GcSALJIRAg2GmmgECCYZKhmUTCIN2gjGG494oMEsMgiC2BlgShG0HQ2HjpRI10aC+DgDIkKkGaKQZRCAqiEQI3MYghCRASkuzfH/z2e+vUWWvVWlW1p5P6Ps/7vHtXrVq19n7fd3/2qtp7v21ZlmWQlJSUlJSUVBv1qjqApKSkpKSkpK5KcE5KSkpKSqqZEpyTkpKSkpJqpgTnpKSkpKSkminBOSkpKSkpqWZKcE5KSkpKSqqZEpyTkpKSkpJqpgTnpKSkpKSkminBOSkpKSkpqWZKcE5KSkpKSqqZEpyTkpKSkpJqpgTnpKSkpKSkmqlP1QEkJRWh999/HwYMGFB1GJUq/adNUlJzleCc1CP1r//6r/Dkk0/CPvvsU3UoSUlJSWqlYe2kHqlnnnkGJkyYUHUYSUlJSV5KcE7qcVq2bBl0dnZCW1tb1aEkJSUleSnBOanH6f7774eJEydWHUZSUlKStxKck3qc5s2bB4cffrhX27Vr18JZZ50Fo0aNgo6ODthnn33giSeeiByhn+ocW1JSUlwlOCf1KL333nvQ1tbmfaX2+vXrYfTo0fDf//3f8Pbbb8PnP/95OPbYY+G9996LHGnPii0pKSmu2rJ0v0VSD9J9990Hy5cvh//xP/5HNJ9Dhw6FX/ziF/Cxj30sms9YqnNsSUlJ/kqZc1KP0gMPPBB1vvn3v/89rFmzBj760Y9G8xlLdY4tKSkpTD0ezrNmzYK2tjb41a9+VXUolSvfF3/+85/VbZ944gmYOnUqvP3229HjuvPOO2HXXXeFAQMGQFtbGzz33HNefrIsgzfffBM6OzujxPXee+/B6aefDl/5yldgs802i+Izlnxji7WvmyLNd37q1KnRr/Cvan+H/NaT6qEeD+ekOHriiSfg6quvjg7nN998E04//XT46Ec/Cg899BA8+eST8Pd///devp5++mnYe++9o8T1wQcfwEknnQS77LILfPnLX1a1Pfjgg+HMM8+MEgcm39hi7uum6Oijj4Ynn3wSttpqq9L73hT3d1I8JTgnVaqXX34ZPvjgAzjttNPgoIMOgn322QcGDhzo5ev++++HY445Bq1bunQpbLbZZvDZz362S/kDDzwAffv2hSuvvLJVtnHjRjjjjDOgd+/ecPvttxd+v/QJJ5wA22yzTbfy9evXwx577NHlyvOQ2GLu61x1vRgtj2vLLbeEffbZB9rb20uPoYj9nbTpaJODcz509dvf/hZOPPFEGDJkCAwdOhSmTJkC69evh5deegk++clPwuDBg2H77beH66+/vkv7V155Bc466ywYM2YMDBw4ELbeems45phj4Pnnn+/W149//GPYfffdob29HXbYYQe4+eabyaGzP/zhD3DKKafA8OHDob29HXbeeWf41re+Jd6eX//613DCCSdAR0cHDBkyBE477TR48803Rfvk8ccfh0MPPRQGDx4MAwcOhP322w9+8pOfdOnj0ksvBQCA0aNHQ1tbG7S1tcGjjz4a5PfMM8+EAw44AAAAPvOZz0BbWxscfPDBrM/58+fDUUcdhda98MILsNtuu6F1W221FVx22WXwX//1X7Bw4UIAAHj00UfhxBNPhM9//vNw7bXXtmw/97nPwdKlS+HOO++EPn2Kf8LtgQceCG+88Qa8+uqrXcpvvPFG+P3vfw/f/va3g2Nz7WvXZwXw/75rzz77LPzTP/0TbL755ux8t+a3QknyG+LiooZ3f/KTn8Aee+wB7e3tMHr0aPjXf/1XcUySfaX5bt97773Q1tYGP/vZz7rVzZgxo3W8Agjbp2eeeSZsv/323cqxY5LkePTmm2/C//pf/wtGjRoF7e3tsOWWW8L+++8P8+bNc8aSJFDWwzVz5swMALIFCxZkWZZlV111VQYA2Y477pj9y7/8SzZ37tzssssuywAgu+CCC7Kddtop++Y3v5nNnTs3O+usszIAyO66666Wv/nz52cXX3xx9n/+z//J5s+fn91zzz3Z8ccfnw0YMCD7/e9/37J78MEHs169emUHH3xwds8992Q/+tGPsgkTJmTbb799Zu/2RYsWZUOGDMl222237Pvf/372yCOPZBdffHHWq1evbOrUqez25duz3XbbZZdeemn28MMPZzfeeGM2aNCgbM8998zWrVvXbV8sXry4Vfboo49mffv2zcaNG5fdeeed2b333psdccQRWVtbWzZnzpwsy7JsyZIl2YUXXpgBQHb33XdnTz75ZPbkk09mK1euJOOS+H3llVeyb33rWxkAZNOmTcuefPLJbNGiRaTPp59+OvvsZz+bAUD2yiuvdKn705/+lF100UXsvnr33XezkSNHZoceemj2zDPPZIMHD87OOuusbOPGjS2bP//5zxkAZP37988GDRrUej322GOoz40bN2YffPBBl9eBBx6YnXHGGd3KKS1cuDADgOyOO+7osj0DBw7MrrnmGu/YTHH7WvJZZVnX79qXvvSlbO7cudm9995L9in9rVCS/oa4uLDv/Lx587LevXtnBxxwQHb33XdnP/rRj7K9994723bbbbv9Nm1J95Xmu/3BBx9kw4cPz0499dRudR//+MezvfbaS71Pse2eNGlStt1223XrI99/uaTHoyOPPDLbcssts9tuuy179NFHs3vvvTf76le/2mU/JPlrk4Xzv/3bv3Wx22OPPVrwyfXBBx9kW265ZXbCCSeQ/tevX5+tW7cuGzNmTPbFL36xVb733ntno0aNytauXdsqW716dTZs2LBuB4Ajjzwy22abbbrB7oILLsj69++f/e1vfyP7z7fH7DvLsuyHP/xhBgDZ7Nmzu+0L8we7zz77ZMOHD89Wr17dZZvGjh2bbbPNNi1w3XDDDd3acpL6/cUvfpEBQPajH/1I5DfLsmzHHXfs9vnddNNN2bx585xt/+M//iMDgGzQoEHZSSedlK1fv17cL6Y8fsmL2ncbNmzIOjo6svPPP79VduSRR2Y77rhjl+9PqKh9Lf2s8u/aV7/6Va/+qd8KJelviIsL+85PmDAhGzlyZLZmzZpW2apVq7KhQ4c64SzdV1mm+25PmTIlGzBgQPb222+3yl588cUMALJbbrmFbEft0xA4S49Hm222WTZ58mTntiX5aZMb1s5l326z8847Q1tbW5ch0z59+sDf/d3fdRluXL9+PUybNg122WUX6NevH/Tp0wf69esHf/jDH+B3v/sdAAC8++678Ktf/QqOP/546NevX6vtZptt1m1O9P3334ef/exn8OlPfxoGDhwI69evb70+9alPwfvvvw9PPfWUc3tOPfXULusnnXQS9OnTB37xi1+Qbd599114+umn4Z/+6Z+6XPHbu3dvOP300+H111+Hl156ydl3WX5zHXfccfDjH/+4S9ljjz0GBx54oLNtfkFOW1sbzJo1C3r37u0dBwDAuHHjYMGCBV1ee+21F0ycOLFb+ciRI1EfvXr1gv3226/1tK8f/vCH8PDDD8N3vvOdLt+fIuTzWf3jP/6jyLfkt8LFJf0NaeJ69913YcGCBXDCCSdA//79W+WDBw8m/Zpti/pen3322bBmzRq48847W2UzZ86E9vZ2OOWUU1plIftUIs3x6OMf/zjMmjULvva1r8FTTz0FH3zwQXD/Sf9Pmyychw4d2mW9X79+MHDgwC4/2Lz8/fffb61PmTIF/vmf/xmOP/54uP/+++Hpp5+GBQsWwMc+9jFYs2YNAACsWLECsiyDESNGdOvXLnvrrbdg/fr1cMstt0Dfvn27vD71qU8BAMBf//pX5/bYtw/16dMHhg0bBm+99RbZJo8Tu5I1BwnXvmy/uY477jj47//+75aPlStXwmabbQZ9+/Zl2z333HMwceJE2H///eGdd96B//iP//COIdfgwYNh/PjxXV6DBw+GYcOGdSvnQHvggQfCb3/7W3jttddgypQpMGnSJOf8ewz5fFbSK58lvxVXXJLfkCauFStWwMaNG9Hb7Vy34BX5vd51111h7733hpkzZwIAwIYNG2D27Nlw3HHHdTlWhexTiTTHozvvvBMmTZoE//t//2/Yd999YejQoXDGGWfAsmXLguNISv/nrNbs2bPhjDPOgGnTpnUp/+tf/wof+chHAABg8803h7a2NvjLX/7Srb39xd18881bZ97nn38+2ufo0aOdcS1btgy23nrr1vr69evhrbfegmHDhpFtNt98c+jVqxcsXbq0W93//b//FwAAtthiC2ffZfnNtc8++8CwYcPggQcegEmTJsGDDz4In/zkJ9k2L730Ehx55JGw7777wo9//GM48cQTYerUqXDaaafBkCFDvGOJpQMPPBA2bNgAEydOhA0bNqguUAqRz2clvUJc8lvh4pL+hjRx5X4xHy6oFP29Puuss+C8886D3/3ud/CnP/0Jli5dCmeddVYXm5B92r9/f1i7dm23cvPkX3M82mKLLeCmm26Cm266CV577TW477774PLLL4fly5fDQw89JNnkJEabbObsq7a2tm63ZfzkJz+BN954o7U+aNAgGD9+PNx7772wbt26Vvk777wDDzzwQJe2AwcOhEMOOQR+/etfw+67794t2xo/fjwL2Fw//OEPu6z/13/9F6xfv57NvgYNGgQTJkyAu+++u8tZ98aNG2H27NmwzTbbtIaB822WnJ1r/PqoV69ecMwxx7SGth988EHyCm4AgD//+c9w2GGHwY477gh33XUX9O3bF6677jpYsWJFt4NcVdp7771hwIAB8Pzzz8MNN9wQdJDXqMjPSvJb4eKS/oY0GjRoEHz84x+Hu+++u8uI2OrVq+H+++93ti3ye33yySdD//79YdasWTBr1izYeuut4YgjjuhiE7JPt99+e1i+fHmXE55169bBww8/3Fr3PR5tu+22cMEFF8Dhhx8Ozz77rHbTkxClzFmpiRMnwqxZs2CnnXaC3XffHRYuXAg33HBDt/tUr7nmGjj66KPhyCOPhIsuugg2bNgAN9xwA2y22Wbwt7/9rYvtzTffDAcccAB84hOfgM9//vOw/fbbw+rVq+GVV16B+++/H37+858747r77ruhT58+cPjhh8OiRYvgn//5n+FjH/sYnHTSSWy76dOnw+GHHw6HHHIIXHLJJdCvXz/49re/DS+88AL853/+ZysbyW9Ruvnmm2HSpEnQt29f2HHHHWHw4MFBfn113HHHwcknnwzvvPMOvPfee2TWsHTpUjjssMNg+PDh8MADD7T+EGOnnXaCs88+G26++ebWPq9SvXr1gs033xzGjx9f6ANMMBX1WUl/K5Q0vyGN/uVf/gU++clPwuGHHw4XX3wxbNiwAb7+9a/DoEGDnH6L/F5/5CMfgU9/+tMwa9YsePvtt+GSSy6BXr265k8h+/Qzn/kMfPWrX4XPfvazcOmll8L7778P3/zmN2HDhg1d7CTHo5UrV8IhhxwCp5xyCuy0004wePBgWLBgATz00ENwwgkneO+DJENVXo1Whqirtd98880udpMmTcoGDRrUrf1BBx2U7brrrq31FStWZOecc042fPjwbODAgdkBBxyQ/fKXv8wOOuig7KCDDurS9p577sl22223rF+/ftm2226bXXfdddkXvvCFbPPNN+/Wz+LFi7Ozzz4723rrrbO+fftmW265ZbbffvtlX/va19jty7dn4cKF2THHHJNtttlm2eDBg7OTTz45+8tf/oLuC/uq4V/+8pfZP/zDP2SDBg3KBgwYkO2zzz7Z/fff362vK664Ihs5cmTWq1evDACyX/ziF2xsEr8+V2tnWZa999572cCBA7MpU6ZkN910k6ptHXXDDTdk/fr1y1588cXC+uD2teSzon47lDS/FUqS3xAXF/Wdv++++7Ldd9+9i1/7qmVK0t+Lz3f7kUceaV3d//LLL3erl+5Tart/+tOfZnvssUc2YMCAbIcddshuvfVWdLtdx6P3338/O/fcc7Pdd9896+joyAYMGJDtuOOO2VVXXZW9++674u1NopX+lapEffDBB7DHHnvA1ltvDY888kgUn1OnToWrr74a3nzzzdKGQuui448/Hn7yk5/Ayy+/LJqXr5vee+89+M1vfgMLFiyASy+9FK699lq45JJLqg6r1iriN5SUVEelYe0Cdc4558Dhhx8OW221FSxbtgy+853vwO9+9zu4+eabqw6tR+i4446DP/7xj40EMwDAI488Ap/+9Kehs7MTvvzlLycwI0q/oaRNVQnOBWr16tVwySWXwJtvvgl9+/aFvfbaC37605/CYYcdVnVoPULHHHOM8/apOuv444+HNHDFK/2GkjZVpWHtpKSkpKSkmqnSW6m+/e1vw+jRo6F///4wbtw4+OUvf1llOElJSUlJSbVQZXC+8847YfLkyXDllVfCr3/9a/jEJz4BRx11FLz22mtVhZSUlJSUlFQLVTasPWHCBNhrr71gxowZrbKdd94Zjj/+eJg+fXoVISUlJSUlJdVClVwQtm7dOli4cCFcfvnlXcqPOOKI1sP/Ta1du7bLY+c2btwIf/vb32DYsGHBD7NISkpKSipfWZbB6tWrYeTIkd0ethJT77//fpenzPmqX79+3f57oUhVAue//vWvsGHDhm4PsB8xYgT6fNvp06fD1VdfXVZ4SUlJSUklacmSJeKnxmn1/vvvw+jRo6P8GUdnZycsXrxYBOjtt9++y78Z5jrvvPPgW9/6lqi/Sm+lsrPeLMvQTPiKK66AKVOmtNZXrlwJ2267Lbz66qutx0dmH/43dZdlex3gw6zb7M+0ocoo/1Q7uw+Xf6pPzt58N/tz9cn5od5dvnPZcVB2XCzcckgdVSa1wbYtRDFnk2KOHlEZjLQP245bl9Zpljn/AP9v+1xtXTa+72YcmjZYHdfObGtvM9WW6peyofpw2efLq1atgu222458BHAMrVu3DpYtWwavvfYadHR0ePtZtWoVbLvttrBu3ToRnBcsWNDlsagvvPACHH744XDiiSeK+6wEzltssQX07t2729nM8uXL0b+Da29v7/awd4AP/66vo6MjCpg1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ZZddYMmSJYUNiTRdq1atglGjRqV95FDaTzKl/SRT2k8yZVkGq1evhpEjRxbWR//+/WHx4sWwbt26YF9Z1v1aFSxrNmXbYz4oNRLOvXr1gq233hoAADo6OtIPwKG0j2RK+0mmtJ9kSvvJraIyZlP9+/eH/v37F96PqS222AJ69+7dLUtevnx5t2yaUqNvpUpKSkpKSqqb+vXrB+PGjYO5c+d2KZ87dy7st99+Ih+NzJyTkpKSkpLqrClTpsDpp58O48ePh3333Rduu+02eO211+Dcc88VtW8snNvb2+Gqq65yjvlvykr7SKa0n2RK+0mmtJ+SAAA+85nPwFtvvQXXXHMNLF26FMaOHQs//elPYbvtthO1b+R9zklJSUlJST1Zac45KSkpKSmpZkpwTkpKSkpKqpkSnJOSkpKSkmqmBOekpKSkpKSaKcE5KSkpKSmpZmoknEP+I7Ppmj59Ouy9994wePBgGD58OBx//PHw0ksvdbHJsgymTp0KI0eOhAEDBsDBBx8MixYt6mKzdu1auPDCC2GLLbaAQYMGwbHHHguvv/56mZtSqqZPnw5tbW0wefLkVlnaTx/qjTfegNNOOw2GDRsGAwcOhD322AMWLlzYqk/7CWD9+vXwla98BUaPHg0DBgyAHXbYAa655ppuf26zqe+npIjKGqY5c+Zkffv2zb773e9mL774YnbRRRdlgwYNyl599dWqQytFRx55ZDZz5szshRdeyJ577rns6KOPzrbddtvsnXfeadlcd9112eDBg7O77rore/7557PPfOYz2VZbbZWtWrWqZXPuuedmW2+9dTZ37tzs2WefzQ455JDsYx/7WLZ+/foqNqtQPfPMM9n222+f7b777tlFF13UKk/7Kcv+9re/Zdttt1125plnZk8//XS2ePHibN68edkrr7zSskn7Kcu+9rWvZcOGDcseeOCBbPHixdmPfvSjbLPNNstuuummlk3aT0kx1Tg4f/zjH8/OPffcLmU77bRTdvnll1cUUbVavnx5BgDZ/PnzsyzLso0bN2adnZ3Zdddd17J5//33syFDhmTf+c53sizLsrfffjvr27dvNmfOnJbNG2+8kfXq1St76KGHyt2AgrV69epszJgx2dy5c7ODDjqoBee0nz7Ul770peyAAw4g69N++lBHH310dvbZZ3cpO+GEE7LTTjsty7K0n5Liq1HD2vl/ZB5xxBFdyjX/kdnTtHLlSgAAGDp0KAAALF68GJYtW9ZlH7W3t8NBBx3U2kcLFy6EDz74oIvNyJEjYezYsT1uP55//vlw9NFHw2GHHdalPO2nD3XffffB+PHj4cQTT4Thw4fDnnvuCd/97ndb9Wk/fagDDjgAfvazn8HLL78MAAC/+c1v4PHHH4dPfepTAJD2U1J8NerxnTH+I7MnKcsymDJlChxwwAEwduxYAIDWfsD20auvvtqy6devH2y++ebdbHrSfpwzZw48++yzsGDBgm51aT99qD/96U8wY8YMmDJlCnz5y1+GZ555Br7whS9Ae3s7nHHGGWk//f/60pe+BCtXroSddtoJevfuDRs2bIBrr70WTj75ZABI36ek+GoUnHOF/EdmT9IFF1wAv/3tb+Hxxx/vVuezj3rSflyyZAlcdNFF8Mgjj7B/F7ep76eNGzfC+PHjYdq0aQAAsOeee8KiRYtgxowZcMYZZ7TsNvX9dOedd8Ls2bPhjjvugF133RWee+45mDx5MowcORImTZrUstvU91NSPDVqWDvGf2T2FF144YVw3333wS9+8QvYZpttWuWdnZ0AAOw+6uzshHXr1sGKFStIm6Zr4cKFsHz5chg3bhz06dMH+vTpA/Pnz4dvfvOb0KdPn9Z2bur7aauttoJddtmlS9nOO+8Mr732GgCk71OuSy+9FC6//HL47Gc/C7vtthucfvrp8MUvfhGmT58OAGk/JcVXo+Ac4z8ym64sy+CCCy6Au+++G37+85/D6NGju9SPHj0aOjs7u+yjdevWwfz581v7aNy4cdC3b98uNkuXLoUXXnihx+zHQw89FJ5//nl47rnnWq/x48fDqaeeCs899xzssMMOaT8BwP7779/tVryXX3659c856fv0od577z3o1avr4bJ3796tW6nSfkqKroouRPNWfivV7bffnr344ovZ5MmTs0GDBmV//vOfqw6tFH3+85/PhgwZkj366KPZ0qVLW6/33nuvZXPddddlQ4YMye6+++7s+eefz04++WT0lo5tttkmmzdvXvbss89m//AP/9Djb+kwr9bOsrSfsuzD28z69OmTXXvttdkf/vCH7Ic//GE2cODAbPbs2S2btJ+ybNKkSdnWW2/dupXq7rvvzrbYYovssssua9mk/ZQUU42Dc5Zl2be+9a1su+22y/r165fttdderduINgUBAPqaOXNmy2bjxo3ZVVddlXV2dmbt7e3ZgQcemD3//PNd/KxZsya74IILsqFDh2YDBgzIJk6cmL322mslb025suGc9tOHuv/++7OxY8dm7e3t2U477ZTddtttXerTfsqyVatWZRdddFG27bbbZv3798922GGH7Morr8zWrl3bskn7KSmm0v85JyUlJSUl1UyNmnNOSkpKSkraFJTgnJSUlJSUVDMlOCclJSUlJdVMCc5JSUlJSUk1U4JzUlJSUlJSzZTgnJSUlJSUVDMlOCclJSUlJdVMCc5JSUlJSUk1U4JzUlJSUlJSzZTgnJSUlJSUVDMlOCclJSUlJdVM/x9/CGwf8kJFEwAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.imshow(z,cmap=plt.cm.gray);plt.colorbar()\n",
    "plt.title(\"Image plot of $\\sqrt{x^2+y^2}$ for a grid of values\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f3da58f0",
   "metadata": {},
   "source": [
    "# 将条件逻辑表述为数组运算\n",
    "numpy.where函数是三元表达式 x if condition else y 的矢量化版本，假设我们有一个布尔数组和两个值数组:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 131,
   "id": "8207aa0c",
   "metadata": {},
   "outputs": [],
   "source": [
    "xarr = np.array([1.1,1.2,1.3,1.4,1.5])\n",
    "yarr = np.array([2.1,2.2,2.3,2.4,2.5])\n",
    "cond = np.array([True,False,True,True,False])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3f49c3f7",
   "metadata": {},
   "source": [
    "假设我们想要根据cond中的值选中xarr和yarr的值:当cond中的值为True时，选取xarr的值，否则从yarr中选取。列表推导式写法如下"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 132,
   "id": "0ec236e0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[1.1, 2.2, 1.3, 1.4, 2.5]"
      ]
     },
     "execution_count": 132,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "result = [(x if c else y) for x,y,c in zip(xarr,yarr,cond)] # zip()将迭代对象作为参数，将对象中对应方法打包成一个元组，然后返回\n",
    "result                                                       # 由这些元组组成的list"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "da214ac8",
   "metadata": {},
   "source": [
    "这种纯python的写法处理速度慢，且无法用于多维数组，若用np.where就可以把功能写的很简洁"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 133,
   "id": "258af777",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1.1, 2.2, 1.3, 1.4, 2.5])"
      ]
     },
     "execution_count": 133,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "result = np.where(cond,xarr,yarr)\n",
    "result"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "048ac952",
   "metadata": {},
   "source": [
    "np.where的第一个参数为判断条件，而第二第三参数不必是数组，可以是标量值。\n",
    "在使用中np.where经常用于根据一个数组产生一个新的数组：\n",
    "假设有一个由随机数据组成的矩阵，将所有正值替换为2，所有负值替换成-2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 134,
   "id": "4d6eb443",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 1.39595466, -0.06786897, -0.38880612,  0.65788331],\n",
       "       [ 0.45516478, -0.02129167, -1.13867192, -1.27983256],\n",
       "       [-0.90151097,  1.33607766, -0.46125103, -0.51132908],\n",
       "       [ 0.45729048, -0.6141165 , -0.80421038, -1.60341814]])"
      ]
     },
     "execution_count": 134,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr = np.random.randn(4,4)\n",
    "arr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 135,
   "id": "6366ab35",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 2, -2, -2,  2],\n",
       "       [ 2, -2, -2, -2],\n",
       "       [-2,  2, -2, -2],\n",
       "       [ 2, -2, -2, -2]])"
      ]
     },
     "execution_count": 135,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.where(arr>0,2,-2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 136,
   "id": "d167fbf8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 1.39595466, -0.06786897, -0.38880612,  0.65788331],\n",
       "       [ 0.45516478, -0.02129167, -1.13867192, -1.27983256],\n",
       "       [-0.90151097,  1.33607766, -0.46125103, -0.51132908],\n",
       "       [ 0.45729048, -0.6141165 , -0.80421038, -1.60341814]])"
      ]
     },
     "execution_count": 136,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr # 原来的不变"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 138,
   "id": "f17a0fcb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 2.        , -0.06786897, -0.38880612,  2.        ],\n",
       "       [ 2.        , -0.02129167, -1.13867192, -1.27983256],\n",
       "       [-0.90151097,  2.        , -0.46125103, -0.51132908],\n",
       "       [ 2.        , -0.6141165 , -0.80421038, -1.60341814]])"
      ]
     },
     "execution_count": 138,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 也可以只把正值设置为2，其余不变\n",
    "np.where(arr>0,2,arr)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cafb1ee2",
   "metadata": {},
   "source": [
    "传递给where的数据大小可以不相等，甚至可以是标量值。\n",
    "使用where可以表达出更复杂的逻辑，比如我们有两个布尔型数组cond1和cond2"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4eb240e3",
   "metadata": {},
   "source": [
    "### 逻辑\n",
    "result = []\n",
    "for i in range(n):\n",
    "    if cond1[i] and cond2[i]:\n",
    "        result.append(0)\n",
    "    elif cond1[i]:\n",
    "        result.append(1)\n",
    "    elif cond2[i]:\n",
    "        result.append(2)\n",
    "    else:\n",
    "        result.append(3)\n",
    "这个for循环可以改写成一个嵌套的where表达式:"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a21f0330",
   "metadata": {},
   "source": [
    "np.where(cond1 & cond2,0,\n",
    "        np.where(cond1,1,\n",
    "                np.where(cond2,2,3)))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "90cf14ad",
   "metadata": {},
   "source": [
    "我们还可以用\"布尔值在计算过程中可以被当作0或1处理\"改写成一下算术运算（6）"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "36236da0",
   "metadata": {},
   "source": [
    "result = 1*(cond1 - cond2) + 2*(cond2 & -cond1) + 3* -(cond1|cond2)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f0178800",
   "metadata": {},
   "source": [
    "# 数学和统计方法\n",
    "可以通过数组上的一组数学函数对整个数组或某个轴向的数据进行统计计算。sum、mean以及标准差std等聚合计算(直接得到最终结果，aggregation，通常叫做约简(reduction))既可以当作数组的实例方法使用，也可以当作顶级Numpy函数使用"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 141,
   "id": "4cc01015",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 0.68920602, -0.53510218,  1.1939524 , -1.43757147],\n",
       "       [-0.52695821, -0.02314714,  0.25601309, -0.53765451],\n",
       "       [-1.06257951, -2.08250046,  0.14740113, -1.0689935 ],\n",
       "       [ 0.78087519, -0.02727014,  0.85753957,  0.02663162],\n",
       "       [ 1.16680306, -0.58284562,  1.02442472, -0.71781284]])"
      ]
     },
     "execution_count": 141,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr = np.random.randn(5,4)# 正态分布数据\n",
    "arr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 142,
   "id": "ace416e7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "-0.12297943940984805"
      ]
     },
     "execution_count": 142,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr.mean() # 数组实例方法"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 143,
   "id": "a2d42bed",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "-0.12297943940984805"
      ]
     },
     "execution_count": 143,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.mean(arr)  # numpy函数"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 144,
   "id": "fd441219",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "-2.459588788196961"
      ]
     },
     "execution_count": 144,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr.sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1ac69b9f",
   "metadata": {},
   "source": [
    "mena和sum这类函数可以接受一个axis参数(用于计算该轴上的统计值)，最终结果是一个少一维的数组:\n",
    "axis = 0时表示按列，=1时表示按行(原理即把每个数字的元组组成写出来(0,1)，对照第几列就是轴几)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 145,
   "id": "357337b3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([-0.02237881, -0.20793669, -1.01666809,  0.40944406,  0.22264233])"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr.mean(axis = 1) # 按行"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 146,
   "id": "89d9c23c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 1.04734655, -3.25086553,  3.47933091, -3.73540071])"
      ]
     },
     "execution_count": 146,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr.sum(0) # 按列"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fdc9b828",
   "metadata": {},
   "source": [
    "其他如cumsum和cumprod之类的方法不聚合，而是产生一个由中间结果组成的数组:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 147,
   "id": "0dbf7ea6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0, 1, 2],\n",
       "       [3, 4, 5],\n",
       "       [6, 7, 8]])"
      ]
     },
     "execution_count": 147,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr = np.array([[0,1,2],[3,4,5],[6,7,8]])\n",
    "arr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 148,
   "id": "26e6ac58",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 0,  1,  3,  6, 10, 15, 21, 28, 36])"
      ]
     },
     "execution_count": 148,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr.cumsum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 149,
   "id": "ad546ae9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 0,  1,  2],\n",
       "       [ 3,  5,  7],\n",
       "       [ 9, 12, 15]])"
      ]
     },
     "execution_count": 149,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr.cumsum(0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 151,
   "id": "78fe063b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[  0,   0,   0],\n",
       "       [  3,  12,  60],\n",
       "       [  6,  42, 336]])"
      ]
     },
     "execution_count": 151,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr.cumprod(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 153,
   "id": "18711c7c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([  0,  60, 336])"
      ]
     },
     "execution_count": 153,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr.prod(1)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "005d1cd3",
   "metadata": {},
   "source": [
    "还有一些数组统计方法见p104表\n",
    "# 用于布尔型数组的方法\n",
    "在上述这些方法中，布尔值会被强制转换为1，(True)和0(False)。因此，sum经常被用来对布尔型数组中的True值计数:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 154,
   "id": "81ff26b9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "51"
      ]
     },
     "execution_count": 154,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr = np.random.randn(100)\n",
    "(arr>0).sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "256b9718",
   "metadata": {},
   "source": [
    "还有两个方法any和all，他们对布尔型数组非常有用，当然也可以用于非布尔型数组，所有非0元素将被当作True。\n",
    "any用于检测数组中是否存在一个或多个True，而all则检查数组中所有值是否都是True"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 155,
   "id": "3019b0ff",
   "metadata": {},
   "outputs": [],
   "source": [
    "bools = np.array([False,False,True,False])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 156,
   "id": "49333480",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 156,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "bools.any()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 157,
   "id": "30388453",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "False"
      ]
     },
     "execution_count": 157,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "bools.all()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bc671d20",
   "metadata": {},
   "source": [
    "# 排序\n",
    "跟python内置的列表一样，NumPy数组也可以通过sort方法排序:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 158,
   "id": "8c2dc48b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 0.78360551,  0.33184397,  0.72934639, -1.08776835, -0.09152173,\n",
       "       -0.41032779, -0.37258074, -0.06877516])"
      ]
     },
     "execution_count": 158,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr = np.random.randn(8)\n",
    "arr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 160,
   "id": "f425203a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([-1.08776835, -0.41032779, -0.37258074, -0.09152173, -0.06877516,\n",
       "        0.33184397,  0.72934639,  0.78360551])"
      ]
     },
     "execution_count": 160,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr.sort()\n",
    "arr"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4b61a03b",
   "metadata": {},
   "source": [
    "多维数组可以在任何一个轴上进行排序，只需将轴编号传给sort即可(原理即把每个数字的元组组成写出来如(0,1,2)，对照第几列就是轴几)\n",
    "比如说\n",
    "0.58(0,1)\n",
    "1.14(0,2)\n",
    "-0.41(0,3)\n",
    "由于轴线是1，就是对着取元组右半边排序，从效果上来看就是按行排序"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 161,
   "id": "dc7f156b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 0.5888034 ,  1.14279987, -0.41475084],\n",
       "       [-0.52047509, -1.61858872,  1.36510202],\n",
       "       [ 0.03819903, -0.60642371,  0.87989419],\n",
       "       [ 1.03046213,  1.30044014,  2.24327605],\n",
       "       [-0.19006715,  0.0345155 , -0.30115184]])"
      ]
     },
     "execution_count": 161,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr = np.random.randn(5,3)\n",
    "arr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 162,
   "id": "d690f6c8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[-0.41475084,  0.5888034 ,  1.14279987],\n",
       "       [-1.61858872, -0.52047509,  1.36510202],\n",
       "       [-0.60642371,  0.03819903,  0.87989419],\n",
       "       [ 1.03046213,  1.30044014,  2.24327605],\n",
       "       [-0.30115184, -0.19006715,  0.0345155 ]])"
      ]
     },
     "execution_count": 162,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr.sort(1)\n",
    "arr"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2ad5ca6f",
   "metadata": {},
   "source": [
    "顶级方法np.sort返回的是数组的已排序副本，而就地排序则会修改数组本身。计算数组分位数最简单的办法是对其进行排序，然后选取特定位置的值:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 163,
   "id": "69d85e60",
   "metadata": {},
   "outputs": [],
   "source": [
    "large_arr = np.random.randn(1000)\n",
    "large_arr.sort()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 164,
   "id": "3f6b8216",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "-1.6702712860098508"
      ]
     },
     "execution_count": 164,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "large_arr[int(0.05*len(large_arr))]    # 5%分位数"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 165,
   "id": "c1bccdd5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.04077811570603626"
      ]
     },
     "execution_count": 165,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "large_arr[int(0.5*len(large_arr))]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6ebd9068",
   "metadata": {},
   "source": [
    "更多关于numpy排序方法以及注入间接排序之类的高级技术在第12章，在pandas中还可以找到一些其他跟排序有关的数据操作(比如根据一列或多列对表格型数据进行排序)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c3f0d3fa",
   "metadata": {},
   "source": [
    "# 唯一化以及其他的集合逻辑\n",
    "numpy提供了一些针对一维数组ndarray的基本集合运算。最常用的是np.unique，它用于找出数组中的唯一值并返回已排序的结果:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 166,
   "id": "253f7734",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['Bob', 'Joe', 'Will'], dtype='<U4')"
      ]
     },
     "execution_count": 166,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 例1\n",
    "names = np.array(['Bob','Joe','Will','Bob','Will','Joe','Joe'])\n",
    "np.unique(names)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 168,
   "id": "3173513d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'], dtype='<U4')"
      ]
     },
     "execution_count": 168,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "names"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 169,
   "id": "59dd2f67",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 2, 3, 4])"
      ]
     },
     "execution_count": 169,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 例2   注意返回的是已排好序的结果\n",
    "ints = np.array([3,3,3,2,2,1,1,4,4])\n",
    "np.unique(ints)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 170,
   "id": "70deb2cb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['Bob', 'Joe', 'Will']"
      ]
     },
     "execution_count": 170,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# python纯代码实现\n",
    "sorted(set(names))        # set集合具有去重效果，然后再排序"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "55c6f914",
   "metadata": {},
   "source": [
    "另一个函数比较常用的函数是np.in1d(x,y)，得到一个\"x的元素是否包含于y\"的布尔型数组"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 171,
   "id": "0cf13260",
   "metadata": {},
   "outputs": [],
   "source": [
    "values = np.array([6,0,0,3,2,5,6])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 173,
   "id": "6f4fefe1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ True, False, False,  True,  True, False,  True])"
      ]
     },
     "execution_count": 173,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.in1d(values,[2,3,6])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4c25b396",
   "metadata": {},
   "source": [
    "还有一些集合函数见p107"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8d665f16",
   "metadata": {},
   "source": [
    "# 4.用于数组的文件输入输出\n",
    "NumPy能够读写磁盘上的文本数据或二进制数据。后面会介绍一些pandas中用于将表格型数据读取到内存的工具。"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5f737938",
   "metadata": {},
   "source": [
    "# 将数组以二进制格式保存到磁盘\n",
    "np.save和np.load是读写磁盘数组数据的两个主要函数。默认情况下，数组是以未压缩的原始二进制格式保存在扩展名为.npy的文件中"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 174,
   "id": "63b683cb",
   "metadata": {},
   "outputs": [],
   "source": [
    "arr = np.arange(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 175,
   "id": "4b14be97",
   "metadata": {},
   "outputs": [],
   "source": [
    "np.save('some_array',arr)# 默认存在jupyer打开的路径里"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f08bba0e",
   "metadata": {},
   "source": [
    "如果文件路径末尾没有扩展名.npy，则该扩展名会被自动加上。然后就可以通过np.load读取磁盘上的数组。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 177,
   "id": "024d7665",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])"
      ]
     },
     "execution_count": 177,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.load('some_array.npy')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c3868ef6",
   "metadata": {},
   "source": [
    "通过np.savez可以将多个数组保存到一个压缩文件中，将数组以关键字参数的形式传入即可"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 178,
   "id": "6ebb0888",
   "metadata": {},
   "outputs": [],
   "source": [
    "np.savez('array_archive.npz',a=arr,b=arr)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0a2259a9",
   "metadata": {},
   "source": [
    "加载.npz文件时，可以得到一个类似字典的对象，该对象会对各个数组进行延迟加载"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 179,
   "id": "309ccfe6",
   "metadata": {},
   "outputs": [],
   "source": [
    "arch = np.load('array_archive.npz')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 180,
   "id": "0a048461",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])"
      ]
     },
     "execution_count": 180,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arch['b']  # 类似于字典"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ac8fae5f",
   "metadata": {},
   "source": [
    "# 存取文本文件\n",
    "从文件中加载文本是一件非常标准的任务，python的文件读写函数的格式容易晕，所有主要使用pandas中的read_csv和read_table。\n",
    "有时需要用np.loadtxt或更为专门化的np.genfromtxt将数据加载到普通的numpy数组中。\n",
    "这些函数都有很多想选可供使用：指定各种分隔符、针对特定列的转换器函数、需要跳过的行数等。以一个简单的逗号分隔文件(csv)为例"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4a598013",
   "metadata": {},
   "source": [
    "'E:\\data_analysis\\资料\\pydata-book-3rd-edition\\examples/array_ex.txt'\n",
    "将上地址的文件加载到一个二维数组中："
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 183,
   "id": "5a7f83e5",
   "metadata": {},
   "outputs": [],
   "source": [
    "arr = np.loadtxt('E:/data_analysis/资料/pydata-book-3rd-edition/examples/array_ex.txt',delimiter=',')  # 以逗号分隔文件(CSV)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 184,
   "id": "f6ad2200",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 0.580052,  0.18673 ,  1.040717,  1.134411],\n",
       "       [ 0.194163, -0.636917, -0.938659,  0.124094],\n",
       "       [-0.12641 ,  0.268607, -0.695724,  0.047428],\n",
       "       [-1.484413,  0.004176, -0.744203,  0.005487],\n",
       "       [ 2.302869,  0.200131,  1.670238, -1.88109 ],\n",
       "       [-0.19323 ,  1.047233,  0.482803,  0.960334]])"
      ]
     },
     "execution_count": 184,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "851ddf33",
   "metadata": {},
   "source": [
    "np.savetxt执行的是相反的操作：将数组写到以某种分隔符隔开的文本文件中。\n",
    "genfromtxt跟loadtxt差不多，只不过它面向的是结构化数组和缺失数据处理。更多相关结构化数组知识看第12章\n",
    "更多有关文件读写(尤其是表格型数据)知识名字后面pandas和DataFrame对象章节"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "059dcb83",
   "metadata": {},
   "source": [
    "# 5.线性代数\n",
    "线性代数(如矩阵乘法、矩阵分解、行列式以及其他方阵数学等)是任何数组库的重要组成部分。以为通过 * 对两个二维数组相乘得到的是一个元素级的积，而不是一个矩阵点积。因此，numpy提供了一个用于矩阵乘法的dot函数(即是一个数组方法也是numpy命名空间中的一个函数)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 220,
   "id": "f832fa6e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1., 2., 3.],\n",
       "       [4., 5., 6.]])"
      ]
     },
     "execution_count": 220,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x = np.array([[1.,2.,3.],[4.,5.,6.]])\n",
    "x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 221,
   "id": "f0fb72c5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 6., 23.],\n",
       "       [-1.,  7.],\n",
       "       [ 8.,  9.]])"
      ]
     },
     "execution_count": 221,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y = np.array([[6.,23.],[-1,7],[8,9]])\n",
    "y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 222,
   "id": "077a1f95",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 28.,  64.],\n",
       "       [ 67., 181.]])"
      ]
     },
     "execution_count": 222,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x.dot(y)   # 相当于np.dot(x,y)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c723f0b3",
   "metadata": {},
   "source": [
    "一个二位数组跟一个大小合适的以为数组的矩阵点积运算之后得到的是一个一维数组"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 223,
   "id": "febcce26",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 6., 15.])"
      ]
     },
     "execution_count": 223,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.dot(x,np.ones(3))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "66202c0c",
   "metadata": {},
   "source": [
    "numpy.linalg中有一组标准的矩阵分解运算以及诸如 求逆 和 行列式 之类的东西"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 224,
   "id": "b26fcbcf",
   "metadata": {},
   "outputs": [],
   "source": [
    "from numpy.linalg import inv,qr  # inv求逆，qr用于QR分解"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 226,
   "id": "4000c75e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 2.89286303, -0.65673005,  0.66105493,  0.90273529, -0.74807308],\n",
       "       [-0.92062622,  0.04199541,  0.10603415,  0.83816582, -0.17104726],\n",
       "       [-0.1928912 ,  1.00254358,  0.81995827,  0.83242996, -0.77163792],\n",
       "       [ 0.20182507, -1.02403026, -0.53053515, -0.00724483, -1.01890923],\n",
       "       [-0.37888763,  0.43815606,  1.15563841, -0.32768204,  1.11999838]])"
      ]
     },
     "execution_count": 226,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X = np.random.randn(5,5)\n",
    "X"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 227,
   "id": "6791fe17",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 9.43770533, -2.50456087,  1.11162841,  1.8019762 , -2.48775515],\n",
       "       [-2.50456087,  2.67877031,  1.44199623,  0.1407361 ,  1.24462618],\n",
       "       [ 1.11162841,  1.44199623,  2.7375361 ,  0.99335133,  0.68951517],\n",
       "       [ 1.8019762 ,  0.1407361 ,  0.99335133,  2.3178206 , -1.820634  ],\n",
       "       [-2.48775515,  1.24462618,  0.68951517, -1.820634  ,  3.47686799]])"
      ]
     },
     "execution_count": 227,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mat = X.T.dot(X)\n",
    "mat"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 228,
   "id": "3aa5840d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 0.29689758,  0.40996611, -0.3954729 ,  0.04581219,  0.16809546],\n",
       "       [ 0.40996611,  1.17580697, -0.74434218, -0.09406032, -0.02920992],\n",
       "       [-0.3954729 , -0.74434218,  1.38719519, -0.7999547 , -0.71050406],\n",
       "       [ 0.04581219, -0.09406032, -0.7999547 ,  1.56481962,  1.04449871],\n",
       "       [ 0.16809546, -0.02920992, -0.71050406,  1.04449871,  1.10619355]])"
      ]
     },
     "execution_count": 228,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "inv(mat)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 229,
   "id": "9bd833b6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 1.00000000e+00, -8.43770576e-17, -2.21732134e-16,\n",
       "        -8.72072324e-17,  5.03091062e-17],\n",
       "       [ 2.57269986e-17,  1.00000000e+00,  1.57772094e-16,\n",
       "         4.48459175e-17, -1.01227168e-16],\n",
       "       [ 4.69994023e-17,  2.69934857e-16,  1.00000000e+00,\n",
       "        -2.66611985e-16,  1.43496216e-16],\n",
       "       [-2.59928616e-18,  9.24664952e-17, -5.59677972e-17,\n",
       "         1.00000000e+00, -1.06848902e-16],\n",
       "       [-4.46024702e-17,  2.36684481e-16, -1.77451021e-16,\n",
       "        -4.10613245e-16,  1.00000000e+00]])"
      ]
     },
     "execution_count": 229,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mat.dot(inv(mat))   #AA逆=E"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 230,
   "id": "d59bcf4a",
   "metadata": {},
   "outputs": [],
   "source": [
    "q,r = qr(mat)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 231,
   "id": "bd54c675",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[-0.91660669, -0.11085139,  0.15628852,  0.33645005,  0.0995962 ],\n",
       "       [ 0.2432474 , -0.6970772 ,  0.66142261,  0.13089951, -0.01730681],\n",
       "       [-0.10796332, -0.63959508, -0.62785024, -0.08859236, -0.42097213],\n",
       "       [-0.17501113, -0.24441848, -0.03357167, -0.72492203,  0.61886324],\n",
       "       [ 0.2416152 , -0.18158491, -0.37785057,  0.57991907,  0.6554173 ]])"
      ]
     },
     "execution_count": 231,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "q # 正交矩阵Q"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 232,
   "id": "3f3a84e6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[-10.2963522 ,   3.06770867,  -0.97096791,  -2.5702525 ,\n",
       "          3.66729809],\n",
       "       [  0.        ,  -2.7723732 ,  -3.24732185,  -1.16911664,\n",
       "         -1.21919011],\n",
       "       [  0.        ,   0.        ,  -0.88514119,   0.36115285,\n",
       "         -1.25111077],\n",
       "       [  0.        ,   0.        ,   0.        ,  -2.19936564,\n",
       "          2.60094957],\n",
       "       [  0.        ,   0.        ,   0.        ,   0.        ,\n",
       "          0.59249785]])"
      ]
     },
     "execution_count": 232,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "r # 上三角矩阵R"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0a41e405",
   "metadata": {},
   "source": [
    "常用numpy.linalg函数见p110"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3f40393e",
   "metadata": {},
   "source": [
    "# 6.随机数生成\n",
    "numpy.random对python内置的random进行了补充,增加了一些用于高效生成多种概率分布的样本值的函数。例如，你可以用normal来得到一个标准正态分布的4×4样本数组"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 233,
   "id": "96743c74",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 7.98615231e-01, -4.87054898e-01, -7.03793088e-01,\n",
       "        -4.13481389e-01],\n",
       "       [-1.20188500e-03, -5.94326808e-01, -7.49469336e-01,\n",
       "        -5.46399378e-02],\n",
       "       [-3.58513700e-05, -2.31094158e+00, -5.56742217e-01,\n",
       "        -1.15091429e+00],\n",
       "       [ 1.70448563e+00,  8.07285798e-02,  8.12735969e-01,\n",
       "         5.77782646e-01]])"
      ]
     },
     "execution_count": 233,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "samples = np.random.normal(size=(4,4))\n",
    "samples"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2f100812",
   "metadata": {},
   "source": [
    "pyhton内置的random模块只能一次生成一个样本值。如果需要大量样本值，numpy.random快了不止一个数量级"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 234,
   "id": "205d866b",
   "metadata": {},
   "outputs": [],
   "source": [
    "from random import normalvariate  #python内置模块"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 235,
   "id": "876eed16",
   "metadata": {},
   "outputs": [],
   "source": [
    "N=1000000"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 237,
   "id": "31aad9b6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "786 ms ± 28.5 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)\n"
     ]
    }
   ],
   "source": [
    "%timeit samples = [normalvariate(0,1) for _ in range(N)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 238,
   "id": "e6cef771",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "23.9 ms ± 300 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)\n"
     ]
    }
   ],
   "source": [
    "%timeit np.random.normal(size=N)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7b370d61",
   "metadata": {},
   "source": [
    "部分numpy.random函数可见p111"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "de503156",
   "metadata": {},
   "source": [
    "# 7.范例：随机漫步\n",
    "通过模拟随机漫步说明如何运用数组运算。先看一个简单的随机漫步的例子:从0开始，步长1和-1出现的概率相等。我们通过内置的random模块以纯python方式实现1000步的随机漫步："
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 243,
   "id": "080bfeb8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "-4"
      ]
     },
     "execution_count": 243,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 纯python\n",
    "import random\n",
    "position = 0\n",
    "walk = [position]\n",
    "steps = 1000\n",
    "for i in range(steps):\n",
    "    step = 1 if random.randint(0,1) else -1   # randint(0,1)表示取范围内的整数，边界也算，所以就是0和1\n",
    "    position += step\n",
    "    walk.append(position)\n",
    "position"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dd0b931a",
   "metadata": {},
   "source": [
    "不难看出随机漫步就是将各步的累计和，可以用一个数组运算来实现。因此，我用np.random模块一次性随机产生1000个\"抛硬币\"结果(即两个数中选一个)\n",
    "然后将其分别设置为1或-1，然后计算累计和:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 250,
   "id": "4ac4097d",
   "metadata": {},
   "outputs": [],
   "source": [
    "nsteps = 1000"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 251,
   "id": "a2ac6b21",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0,\n",
       "       1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1,\n",
       "       0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0,\n",
       "       1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1,\n",
       "       0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,\n",
       "       1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0,\n",
       "       1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1,\n",
       "       0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1,\n",
       "       0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0,\n",
       "       0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0,\n",
       "       1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,\n",
       "       1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0,\n",
       "       1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0,\n",
       "       0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0,\n",
       "       1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1,\n",
       "       1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1,\n",
       "       1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1,\n",
       "       1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0,\n",
       "       0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1,\n",
       "       0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1,\n",
       "       1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1,\n",
       "       0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0,\n",
       "       0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1,\n",
       "       1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0,\n",
       "       1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0,\n",
       "       0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "       0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,\n",
       "       0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1,\n",
       "       0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1,\n",
       "       0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1,\n",
       "       1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0,\n",
       "       1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0,\n",
       "       1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0,\n",
       "       0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1,\n",
       "       0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0,\n",
       "       1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0,\n",
       "       0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0,\n",
       "       1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1,\n",
       "       0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1,\n",
       "       0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0,\n",
       "       0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0,\n",
       "       1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1,\n",
       "       1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0,\n",
       "       1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0,\n",
       "       0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,\n",
       "       0, 0, 0, 1, 1, 1, 0, 1, 1, 1])"
      ]
     },
     "execution_count": 251,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "draws = np.random.randint(0,2,size=nsteps)  # 利用np内置的random取随机数，左闭右开所以只有0和1\n",
    "draws"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 253,
   "id": "4c8d0b81",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,\n",
       "       -1,  1, -1, -1, -1,  1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1, -1,\n",
       "       -1, -1, -1, -1,  1,  1, -1, -1, -1,  1, -1,  1,  1, -1,  1, -1,  1,\n",
       "       -1,  1,  1,  1, -1,  1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1,\n",
       "       -1,  1, -1,  1,  1,  1, -1, -1, -1,  1, -1, -1, -1,  1, -1, -1,  1,\n",
       "        1,  1,  1, -1,  1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1, -1,  1,\n",
       "       -1, -1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1,  1, -1,  1, -1,  1,\n",
       "        1, -1, -1,  1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,\n",
       "       -1, -1,  1,  1,  1,  1,  1,  1, -1,  1,  1,  1, -1, -1, -1, -1, -1,\n",
       "        1, -1, -1, -1,  1, -1, -1, -1,  1,  1, -1,  1,  1,  1,  1, -1,  1,\n",
       "        1, -1, -1,  1, -1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,\n",
       "       -1, -1,  1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1, -1,  1, -1, -1,\n",
       "        1, -1,  1, -1, -1,  1,  1,  1,  1, -1, -1,  1,  1,  1, -1, -1,  1,\n",
       "        1, -1,  1, -1,  1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,\n",
       "       -1, -1,  1, -1,  1,  1, -1,  1, -1, -1, -1, -1, -1,  1, -1, -1, -1,\n",
       "       -1,  1, -1, -1,  1,  1,  1,  1, -1,  1,  1, -1, -1,  1, -1,  1,  1,\n",
       "        1,  1, -1,  1, -1,  1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,\n",
       "        1,  1,  1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1, -1,\n",
       "        1, -1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1,\n",
       "       -1,  1,  1, -1, -1, -1,  1,  1, -1, -1,  1,  1, -1,  1,  1, -1, -1,\n",
       "       -1, -1,  1,  1, -1, -1, -1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1,\n",
       "       -1,  1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1, -1,  1,\n",
       "        1,  1,  1, -1,  1,  1,  1,  1,  1,  1, -1,  1,  1, -1,  1, -1, -1,\n",
       "        1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1,  1, -1,  1,  1,\n",
       "       -1, -1,  1, -1,  1, -1, -1,  1,  1,  1, -1,  1, -1, -1,  1,  1, -1,\n",
       "        1,  1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1, -1,\n",
       "       -1, -1,  1, -1,  1, -1, -1,  1, -1, -1, -1,  1, -1, -1, -1, -1,  1,\n",
       "       -1, -1,  1, -1, -1, -1,  1,  1,  1,  1, -1,  1, -1,  1, -1,  1, -1,\n",
       "       -1, -1,  1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,\n",
       "       -1, -1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1,  1, -1, -1, -1,\n",
       "        1,  1, -1,  1,  1, -1,  1, -1, -1, -1, -1, -1,  1,  1, -1, -1,  1,\n",
       "       -1,  1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1,  1,  1,  1,  1, -1,\n",
       "       -1,  1,  1,  1, -1, -1, -1,  1, -1, -1, -1, -1, -1,  1, -1, -1,  1,\n",
       "       -1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1, -1,  1,\n",
       "       -1, -1,  1,  1, -1,  1, -1, -1, -1, -1, -1, -1,  1, -1,  1, -1, -1,\n",
       "       -1,  1,  1,  1, -1,  1,  1,  1, -1, -1,  1, -1,  1, -1, -1,  1,  1,\n",
       "        1,  1, -1,  1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1, -1,  1,  1,\n",
       "       -1,  1, -1,  1, -1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1,  1,  1,\n",
       "       -1,  1, -1,  1,  1, -1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1, -1,\n",
       "       -1,  1, -1, -1, -1, -1,  1, -1, -1,  1,  1, -1,  1, -1, -1, -1,  1,\n",
       "       -1, -1,  1,  1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1,  1, -1,\n",
       "       -1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1,  1,\n",
       "       -1,  1,  1,  1,  1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1, -1, -1,\n",
       "       -1, -1, -1,  1,  1, -1,  1,  1, -1,  1, -1, -1,  1,  1,  1,  1,  1,\n",
       "       -1,  1, -1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,\n",
       "        1,  1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1,  1, -1,  1,  1,  1,\n",
       "        1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1, -1,  1,\n",
       "        1,  1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1, -1, -1, -1,  1,  1,\n",
       "        1, -1,  1, -1, -1,  1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1,\n",
       "        1,  1,  1, -1,  1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1,  1,\n",
       "       -1,  1, -1, -1,  1, -1, -1,  1, -1, -1, -1, -1,  1,  1,  1,  1,  1,\n",
       "       -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1, -1,  1,\n",
       "        1,  1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1,  1,  1,  1, -1,\n",
       "       -1,  1, -1,  1, -1, -1, -1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,\n",
       "        1, -1, -1,  1,  1,  1,  1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,\n",
       "        1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1, -1, -1,\n",
       "       -1, -1,  1, -1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1,  1, -1, -1,\n",
       "        1, -1,  1,  1, -1,  1, -1, -1, -1,  1,  1,  1,  1,  1,  1, -1,  1,\n",
       "        1,  1, -1,  1, -1, -1, -1,  1,  1,  1, -1,  1,  1,  1])"
      ]
     },
     "execution_count": 253,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "steps = np.where(draws>0,1,-1) # 利用where将随机数分配  诶为什么不一步到位用np.random.randint(-1,1)，噢不行中间有个0\n",
    "steps"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 254,
   "id": "05f688d6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([  1,   0,  -1,  -2,  -3,  -4,  -5,  -4,  -3,  -2,  -1,  -2,  -1,\n",
       "        -2,  -3,  -2,  -1,  -2,  -1,  -2,  -3,  -4,  -3,  -4,  -3,  -2,\n",
       "        -1,   0,   1,   0,  -1,  -2,  -1,  -2,  -3,  -4,  -5,  -6,  -5,\n",
       "        -4,  -5,  -6,  -7,  -6,  -7,  -6,  -5,  -6,  -5,  -6,  -5,  -6,\n",
       "        -5,  -4,  -3,  -4,  -3,  -2,  -1,  -2,  -1,   0,  -1,   0,   1,\n",
       "         0,   1,   2,   1,   2,   1,   2,   3,   4,   3,   2,   1,   2,\n",
       "         1,   0,  -1,   0,  -1,  -2,  -1,   0,   1,   2,   1,   2,   3,\n",
       "         2,   1,   0,  -1,   0,   1,   0,   1,   0,  -1,   0,  -1,  -2,\n",
       "        -1,   0,  -1,  -2,  -1,  -2,  -1,   0,   1,   2,   3,   2,   3,\n",
       "         2,   3,   4,   3,   2,   3,   2,   3,   2,   1,   2,   3,   2,\n",
       "         1,   0,   1,   0,  -1,   0,  -1,  -2,  -1,   0,   1,   2,   3,\n",
       "         4,   3,   4,   5,   6,   5,   4,   3,   2,   1,   2,   1,   0,\n",
       "        -1,   0,  -1,  -2,  -3,  -2,  -1,  -2,  -1,   0,   1,   2,   1,\n",
       "         2,   3,   2,   1,   2,   1,   2,   1,   2,   3,   4,   5,   4,\n",
       "         3,   4,   3,   4,   5,   4,   3,   4,   3,   4,   5,   4,   3,\n",
       "         4,   3,   2,   1,   2,   1,   2,   1,   0,   1,   0,   1,   0,\n",
       "        -1,   0,   1,   2,   3,   2,   1,   2,   3,   4,   3,   2,   3,\n",
       "         4,   3,   4,   3,   4,   3,   2,   1,   2,   3,   4,   3,   2,\n",
       "         1,   0,  -1,   0,  -1,  -2,  -1,  -2,  -1,   0,  -1,   0,  -1,\n",
       "        -2,  -3,  -4,  -5,  -4,  -5,  -6,  -7,  -8,  -7,  -8,  -9,  -8,\n",
       "        -7,  -6,  -5,  -6,  -5,  -4,  -5,  -6,  -5,  -6,  -5,  -4,  -3,\n",
       "        -2,  -3,  -2,  -3,  -2,  -3,  -2,  -1,   0,  -1,  -2,  -3,  -4,\n",
       "        -5,  -4,  -3,  -2,  -1,   0,   1,   2,   1,   0,   1,   2,   1,\n",
       "         0,   1,   2,   3,   2,   3,   2,   3,   2,   3,   2,   3,   4,\n",
       "         5,   6,   5,   4,   5,   4,   5,   4,   5,   6,   5,   4,   5,\n",
       "         6,   5,   4,   3,   4,   5,   4,   3,   4,   5,   4,   5,   6,\n",
       "         5,   4,   3,   2,   3,   4,   3,   2,   1,   2,   1,   2,   1,\n",
       "         2,   3,   2,   1,   2,   1,   0,   1,   0,   1,   2,   1,   0,\n",
       "         1,   0,  -1,  -2,  -1,   0,  -1,  -2,  -3,  -2,  -1,   0,   1,\n",
       "         0,   1,   2,   3,   4,   5,   6,   5,   6,   7,   6,   7,   6,\n",
       "         5,   6,   5,   4,   3,   2,   1,   0,   1,   0,  -1,  -2,  -3,\n",
       "        -4,  -3,  -4,  -3,  -2,  -3,  -4,  -3,  -4,  -3,  -4,  -5,  -4,\n",
       "        -3,  -2,  -3,  -2,  -3,  -4,  -3,  -2,  -3,  -2,  -1,   0,   1,\n",
       "         0,  -1,   0,   1,   2,   1,   2,   3,   2,   3,   4,   5,   4,\n",
       "         3,   2,   3,   2,   3,   2,   1,   2,   1,   0,  -1,   0,  -1,\n",
       "        -2,  -3,  -4,  -3,  -4,  -5,  -4,  -5,  -6,  -7,  -6,  -5,  -4,\n",
       "        -3,  -4,  -3,  -4,  -3,  -4,  -3,  -4,  -5,  -6,  -5,  -6,  -7,\n",
       "        -6,  -5,  -6,  -7,  -6,  -7,  -8,  -9, -10, -11, -12, -13, -14,\n",
       "       -15, -14, -13, -12, -13, -14, -15, -14, -13, -12, -11, -10,  -9,\n",
       "       -10, -11, -12, -11, -10, -11, -10,  -9, -10,  -9, -10, -11, -12,\n",
       "       -13, -14, -13, -12, -13, -14, -13, -14, -13, -12, -11, -12, -11,\n",
       "       -12, -13, -14, -15, -14, -13, -12, -11, -10,  -9, -10, -11, -10,\n",
       "        -9,  -8,  -9, -10, -11, -10, -11, -12, -13, -14, -15, -14, -15,\n",
       "       -16, -15, -16, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24,\n",
       "       -25, -26, -25, -24, -25, -24, -25, -26, -25, -24, -25, -24, -25,\n",
       "       -26, -27, -28, -29, -30, -29, -30, -29, -30, -31, -32, -31, -30,\n",
       "       -29, -30, -29, -28, -27, -28, -29, -28, -29, -28, -29, -30, -29,\n",
       "       -28, -27, -26, -27, -26, -27, -26, -25, -26, -27, -26, -25, -26,\n",
       "       -27, -26, -27, -26, -25, -26, -25, -26, -25, -26, -25, -24, -25,\n",
       "       -24, -25, -24, -25, -24, -23, -22, -21, -20, -21, -20, -21, -20,\n",
       "       -19, -20, -19, -18, -19, -20, -21, -20, -19, -18, -17, -16, -17,\n",
       "       -18, -17, -18, -19, -20, -21, -20, -21, -22, -21, -20, -21, -20,\n",
       "       -21, -22, -23, -22, -23, -24, -23, -22, -21, -20, -21, -22, -21,\n",
       "       -22, -23, -24, -23, -22, -23, -22, -23, -24, -25, -24, -25, -26,\n",
       "       -25, -26, -25, -26, -27, -26, -27, -28, -27, -28, -27, -26, -27,\n",
       "       -26, -25, -24, -23, -22, -23, -22, -23, -22, -21, -22, -23, -22,\n",
       "       -23, -24, -25, -26, -27, -28, -27, -26, -27, -26, -25, -26, -25,\n",
       "       -26, -27, -26, -25, -24, -23, -22, -23, -22, -23, -22, -21, -20,\n",
       "       -19, -20, -21, -22, -23, -24, -25, -26, -25, -24, -23, -22, -21,\n",
       "       -20, -19, -20, -19, -18, -19, -18, -19, -18, -17, -16, -17, -16,\n",
       "       -15, -14, -13, -14, -15, -14, -15, -14, -13, -12, -13, -14, -15,\n",
       "       -16, -17, -16, -15, -16, -15, -14, -13, -14, -13, -12, -13, -12,\n",
       "       -13, -12, -13, -14, -13, -14, -15, -16, -15, -14, -13, -14, -13,\n",
       "       -14, -15, -14, -15, -14, -13, -14, -15, -14, -13, -14, -15, -14,\n",
       "       -13, -12, -11, -10, -11, -10, -11, -10,  -9,  -8,  -9, -10,  -9,\n",
       "       -10, -11, -10,  -9,  -8,  -9,  -8,  -9, -10,  -9, -10, -11, -10,\n",
       "       -11, -12, -13, -14, -13, -12, -11, -10,  -9, -10,  -9,  -8,  -7,\n",
       "        -6,  -7,  -8,  -7,  -8,  -7,  -6,  -5,  -6,  -7,  -8,  -9,  -8,\n",
       "        -7,  -6,  -7,  -6,  -5,  -6,  -7,  -6,  -5,  -6,  -7,  -6,  -5,\n",
       "        -4,  -3,  -2,  -3,  -4,  -3,  -4,  -3,  -4,  -5,  -6,  -5,  -4,\n",
       "        -3,  -4,  -5,  -6,  -5,  -4,  -3,  -2,  -1,  -2,  -3,  -2,  -1,\n",
       "         0,   1,   2,   3,   4,   5,   4,   5,   6,   5,   6,   7,   8,\n",
       "         7,   6,   7,   6,   5,   4,   5,   6,   5,   4,   5,   6,   5,\n",
       "         4,   3,   2,   1,   0,   1,   0,   1,   0,  -1,   0,  -1,   0,\n",
       "        -1,   0,   1,   0,   1,   0,  -1,   0,  -1,   0,   1,   0,   1,\n",
       "         0,  -1,  -2,  -1,   0,   1,   2,   3,   4,   3,   4,   5,   6,\n",
       "         5,   6,   5,   4,   3,   4,   5,   6,   5,   6,   7,   8])"
      ]
     },
     "execution_count": 254,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "walk = steps.cumsum()# 然后利用累加展现每一步到的坐标点\n",
    "walk"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cdf5f767",
   "metadata": {},
   "source": [
    "有了这些数据之后就可以做一些统计工作了如最大最小值"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 255,
   "id": "fe4b947a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "8"
      ]
     },
     "execution_count": 255,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "walk.max()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 256,
   "id": "650d891d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "-32"
      ]
     },
     "execution_count": 256,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "walk.min()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "04053734",
   "metadata": {},
   "source": [
    "现在来看一个复杂点的统计任务--首次穿越时间，就是随机漫步过程中第一次达到某个特定值的时间。假设我们想要知道本次随机漫步需要多久才能距离初始0点至少10步远(任意方向)np.abs(walk)>=10可以得到一个布尔型数组，它表示的是距离是否达到或超过10，而我们想知道的是第一个10或-10的索引。可以用argmax来解决这个问题，它返回的是该布尔型数组第一个最大值的索引(True就是最大值):"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 261,
   "id": "ba2505e4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True, False,  True,  True,  True,  True,  True,  True,\n",
       "        True, False,  True, False,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True, False,  True,  True,  True, False, False, False,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True, False, False, False,  True, False,  True,\n",
       "        True,  True, False, False, False, False, False,  True, False,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True, False,  True, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False, False, False, False, False, False, False, False, False,\n",
       "       False])"
      ]
     },
     "execution_count": 261,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.abs(walk)>=10   # np.abs()是元素级的，加上判断得到的是一个布尔型数组"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 257,
   "id": "30ec0197",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "489"
      ]
     },
     "execution_count": 257,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "(np.abs(walk)>=10).argmax()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "50942464",
   "metadata": {},
   "source": [
    "注，这里使用argmax并不是很高效，以为无论如何都会对数组进行完全扫描。在本例中，只要发现了一个True，那我们就真的它是个最大值了"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d9344ce1",
   "metadata": {},
   "source": [
    "# 一次模拟多个随机漫步\n",
    "如果你希望模拟多个随机漫步过程(如5000个)，只需对上面代码做一点点修改即可生成所以的随机漫步过程。只要给numpy.random的函数传入一个 二元元组 \n",
    "就可以产生一个二维数组，然后我们就可以一次性计算5000个随机漫步过程(一行一个)的累计和了:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 266,
   "id": "0e61d94e",
   "metadata": {},
   "outputs": [],
   "source": [
    "nwalks = 5000\n",
    "nsteps = 1000"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 267,
   "id": "fbaf2a30",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 1, 1, ..., 1, 0, 0],\n",
       "       [1, 0, 0, ..., 1, 1, 1],\n",
       "       [1, 1, 0, ..., 1, 1, 1],\n",
       "       ...,\n",
       "       [0, 0, 0, ..., 0, 1, 1],\n",
       "       [0, 0, 1, ..., 0, 1, 1],\n",
       "       [0, 1, 1, ..., 0, 0, 0]])"
      ]
     },
     "execution_count": 267,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "draws = np.random.randint(0,2,size=(nwalks,nsteps))  # 0或1\n",
    "draws"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 270,
   "id": "1236a3e9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 1,  1,  1, ...,  1, -1, -1],\n",
       "       [ 1, -1, -1, ...,  1,  1,  1],\n",
       "       [ 1,  1, -1, ...,  1,  1,  1],\n",
       "       ...,\n",
       "       [-1, -1, -1, ..., -1,  1,  1],\n",
       "       [-1, -1,  1, ..., -1,  1,  1],\n",
       "       [-1,  1,  1, ..., -1, -1, -1]])"
      ]
     },
     "execution_count": 270,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "steps = np.where(draws>0,1,-1)\n",
    "steps"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 271,
   "id": "3d14ec9c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[  1,   2,   3, ..., -10, -11, -12],\n",
       "       [  1,   0,  -1, ...,   4,   5,   6],\n",
       "       [  1,   2,   1, ..., -48, -47, -46],\n",
       "       ...,\n",
       "       [ -1,  -2,  -3, ..., -20, -19, -18],\n",
       "       [ -1,  -2,  -1, ..., -24, -23, -22],\n",
       "       [ -1,   0,   1, ..., -58, -59, -60]])"
      ]
     },
     "execution_count": 271,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "walks = steps.cumsum(1)  # 1表示按行累加\n",
    "walks"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4d63a07b",
   "metadata": {},
   "source": [
    "现在，我们来计算所有随机漫步过程的最大值和最小值:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 272,
   "id": "76bd6290",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "120"
      ]
     },
     "execution_count": 272,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "walks.max()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 273,
   "id": "62bd92c0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "-120"
      ]
     },
     "execution_count": 273,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "walks.min()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2715a3db",
   "metadata": {},
   "source": [
    "得到这些数据后，我们来计算30或-30的最小穿越时间。\n",
    "注意！这里需要思考一下，因为不是5000个过程都到达了30。我们可以用any方法(如果有一个或多个True就返回True)来对此进行检查:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 274,
   "id": "1bd08aac",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([False,  True,  True, ...,  True,  True,  True])"
      ]
     },
     "execution_count": 274,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "hits30 = (np.abs(walks)>=30).any(1)  # 以行来检查\n",
    "hits30 # 里面True即表示这次随机漫步有大于30的步数"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 275,
   "id": "2ff73f10",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "3379"
      ]
     },
     "execution_count": 275,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "hits30.sum()  # 返回的是True的数量即到达30或-30的数量"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b8483937",
   "metadata": {},
   "source": [
    "然后我们利用这个布尔型数组选出那些超越了30(绝对值)的随机漫步(行)，并调用argmax在轴1上获取穿越时间"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 276,
   "id": "906d1517",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[  1,   0,  -1, ...,   4,   5,   6],\n",
       "       [  1,   2,   1, ..., -48, -47, -46],\n",
       "       [ -1,   0,  -1, ...,  -4,  -5,  -4],\n",
       "       ...,\n",
       "       [ -1,  -2,  -3, ..., -20, -19, -18],\n",
       "       [ -1,  -2,  -1, ..., -24, -23, -22],\n",
       "       [ -1,   0,   1, ..., -58, -59, -60]])"
      ]
     },
     "execution_count": 276,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "walks[hits30] # 有超过30的行"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 278,
   "id": "777afa4a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[False, False, False, ..., False, False, False],\n",
       "       [False, False, False, ...,  True,  True,  True],\n",
       "       [False, False, False, ..., False, False, False],\n",
       "       ...,\n",
       "       [False, False, False, ..., False, False, False],\n",
       "       [False, False, False, ..., False, False, False],\n",
       "       [False, False, False, ...,  True,  True,  True]])"
      ]
     },
     "execution_count": 278,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.abs(walks[hits30])>=30  # 将这些行转换成布尔型，然后用argmax()判断"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 281,
   "id": "03b47863",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([251, 133, 685, ..., 231, 867, 487], dtype=int64)"
      ]
     },
     "execution_count": 281,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "crossing_times = (np.abs(walks[hits30])>=30).argmax(1)\n",
    "crossing_times  # 记录的是满足的行里分别是在第几个索引位置达到30"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 282,
   "id": "3c110467",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "500.7650192364605"
      ]
     },
     "execution_count": 282,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "crossing_times.mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2df5e49c",
   "metadata": {},
   "source": [
    "当然我们也可以同其他分布方式得到漫步数据，只需要使用不同的随机数生成函数，如mormal用于生成指定均值和标准差的正态分布数据：\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 283,
   "id": "2ec63442",
   "metadata": {},
   "outputs": [],
   "source": [
    "steps = np.random.normal(loc=0,scale=0.25,size=(nwalks,nsteps))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4a5d7c5b",
   "metadata": {},
   "source": [
    "..."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
