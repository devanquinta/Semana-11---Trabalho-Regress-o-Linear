# Semana-11---Trabalho-Regress-o-Linear
Nome: Vander Quintanilha - Mat: 0050013538
from sklearn.datasets import load_boston
from sklearn import datasets
#Crie um cabeçalho com nome e matrícula
#Mostre os primeiros registros da tabela
import pandas as pd
dados_boston = datasets.load_boston()
dataset_boston = pd.DataFrame(dados_boston.data,columns=dados_boston.feature_names)
dataset_boston['MEDIDA'] = pd.Series(dados_boston.target)
dataset_boston.head()
CRIM	ZN	INDUS	CHAS	NOX	RM	AGE	DIS	RAD	TAX	PTRATIO	B	LSTAT	MEDIDA
0	0.00632	18.0	2.31	0.0	0.538	6.575	65.2	4.0900	1.0	296.0	15.3	396.90	4.98	24.0
1	0.02731	0.0	7.07	0.0	0.469	6.421	78.9	4.9671	2.0	242.0	17.8	396.90	9.14	21.6
2	0.02729	0.0	7.07	0.0	0.469	7.185	61.1	4.9671	2.0	242.0	17.8	392.83	4.03	34.7
3	0.03237	0.0	2.18	0.0	0.458	6.998	45.8	6.0622	3.0	222.0	18.7	394.63	2.94	33.4
4	0.06905	0.0	2.18	0.0	0.458	7.147	54.2	6.0622	3.0	222.0	18.7	396.90	5.33	36.2
# Linhas, colunas e tributos
print(dados_boston.data.shape)
#506 linhas e 13 colunas 
print(dados_boston.feature_names)
#13 features
#print(dados_boston.target)
(506, 13)
['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
 'B' 'LSTAT']
# Target e dados
# TARGET
print('\n################# Conjunto de dados ###########################\n')
print(dataset_boston)
print('\n---------------------------------------------------------------------------------------------------------\n')
print('\n################# Totalidade dos dados ###########################\n')
#DADOS
print(dados_boston)
################# Conjunto de dados ###########################

         CRIM    ZN  INDUS  CHAS    NOX     RM    AGE     DIS   RAD    TAX  \
0     0.00632  18.0   2.31   0.0  0.538  6.575   65.2  4.0900   1.0  296.0   
1     0.02731   0.0   7.07   0.0  0.469  6.421   78.9  4.9671   2.0  242.0   
2     0.02729   0.0   7.07   0.0  0.469  7.185   61.1  4.9671   2.0  242.0   
3     0.03237   0.0   2.18   0.0  0.458  6.998   45.8  6.0622   3.0  222.0   
4     0.06905   0.0   2.18   0.0  0.458  7.147   54.2  6.0622   3.0  222.0   
5     0.02985   0.0   2.18   0.0  0.458  6.430   58.7  6.0622   3.0  222.0   
6     0.08829  12.5   7.87   0.0  0.524  6.012   66.6  5.5605   5.0  311.0   
7     0.14455  12.5   7.87   0.0  0.524  6.172   96.1  5.9505   5.0  311.0   
8     0.21124  12.5   7.87   0.0  0.524  5.631  100.0  6.0821   5.0  311.0   
9     0.17004  12.5   7.87   0.0  0.524  6.004   85.9  6.5921   5.0  311.0   
10    0.22489  12.5   7.87   0.0  0.524  6.377   94.3  6.3467   5.0  311.0   
11    0.11747  12.5   7.87   0.0  0.524  6.009   82.9  6.2267   5.0  311.0   
12    0.09378  12.5   7.87   0.0  0.524  5.889   39.0  5.4509   5.0  311.0   
13    0.62976   0.0   8.14   0.0  0.538  5.949   61.8  4.7075   4.0  307.0   
14    0.63796   0.0   8.14   0.0  0.538  6.096   84.5  4.4619   4.0  307.0   
15    0.62739   0.0   8.14   0.0  0.538  5.834   56.5  4.4986   4.0  307.0   
16    1.05393   0.0   8.14   0.0  0.538  5.935   29.3  4.4986   4.0  307.0   
17    0.78420   0.0   8.14   0.0  0.538  5.990   81.7  4.2579   4.0  307.0   
18    0.80271   0.0   8.14   0.0  0.538  5.456   36.6  3.7965   4.0  307.0   
19    0.72580   0.0   8.14   0.0  0.538  5.727   69.5  3.7965   4.0  307.0   
20    1.25179   0.0   8.14   0.0  0.538  5.570   98.1  3.7979   4.0  307.0   
21    0.85204   0.0   8.14   0.0  0.538  5.965   89.2  4.0123   4.0  307.0   
22    1.23247   0.0   8.14   0.0  0.538  6.142   91.7  3.9769   4.0  307.0   
23    0.98843   0.0   8.14   0.0  0.538  5.813  100.0  4.0952   4.0  307.0   
24    0.75026   0.0   8.14   0.0  0.538  5.924   94.1  4.3996   4.0  307.0   
25    0.84054   0.0   8.14   0.0  0.538  5.599   85.7  4.4546   4.0  307.0   
26    0.67191   0.0   8.14   0.0  0.538  5.813   90.3  4.6820   4.0  307.0   
27    0.95577   0.0   8.14   0.0  0.538  6.047   88.8  4.4534   4.0  307.0   
28    0.77299   0.0   8.14   0.0  0.538  6.495   94.4  4.4547   4.0  307.0   
29    1.00245   0.0   8.14   0.0  0.538  6.674   87.3  4.2390   4.0  307.0   
..        ...   ...    ...   ...    ...    ...    ...     ...   ...    ...   
476   4.87141   0.0  18.10   0.0  0.614  6.484   93.6  2.3053  24.0  666.0   
477  15.02340   0.0  18.10   0.0  0.614  5.304   97.3  2.1007  24.0  666.0   
478  10.23300   0.0  18.10   0.0  0.614  6.185   96.7  2.1705  24.0  666.0   
479  14.33370   0.0  18.10   0.0  0.614  6.229   88.0  1.9512  24.0  666.0   
480   5.82401   0.0  18.10   0.0  0.532  6.242   64.7  3.4242  24.0  666.0   
481   5.70818   0.0  18.10   0.0  0.532  6.750   74.9  3.3317  24.0  666.0   
482   5.73116   0.0  18.10   0.0  0.532  7.061   77.0  3.4106  24.0  666.0   
483   2.81838   0.0  18.10   0.0  0.532  5.762   40.3  4.0983  24.0  666.0   
484   2.37857   0.0  18.10   0.0  0.583  5.871   41.9  3.7240  24.0  666.0   
485   3.67367   0.0  18.10   0.0  0.583  6.312   51.9  3.9917  24.0  666.0   
486   5.69175   0.0  18.10   0.0  0.583  6.114   79.8  3.5459  24.0  666.0   
487   4.83567   0.0  18.10   0.0  0.583  5.905   53.2  3.1523  24.0  666.0   
488   0.15086   0.0  27.74   0.0  0.609  5.454   92.7  1.8209   4.0  711.0   
489   0.18337   0.0  27.74   0.0  0.609  5.414   98.3  1.7554   4.0  711.0   
490   0.20746   0.0  27.74   0.0  0.609  5.093   98.0  1.8226   4.0  711.0   
491   0.10574   0.0  27.74   0.0  0.609  5.983   98.8  1.8681   4.0  711.0   
492   0.11132   0.0  27.74   0.0  0.609  5.983   83.5  2.1099   4.0  711.0   
493   0.17331   0.0   9.69   0.0  0.585  5.707   54.0  2.3817   6.0  391.0   
494   0.27957   0.0   9.69   0.0  0.585  5.926   42.6  2.3817   6.0  391.0   
495   0.17899   0.0   9.69   0.0  0.585  5.670   28.8  2.7986   6.0  391.0   
496   0.28960   0.0   9.69   0.0  0.585  5.390   72.9  2.7986   6.0  391.0   
497   0.26838   0.0   9.69   0.0  0.585  5.794   70.6  2.8927   6.0  391.0   
498   0.23912   0.0   9.69   0.0  0.585  6.019   65.3  2.4091   6.0  391.0   
499   0.17783   0.0   9.69   0.0  0.585  5.569   73.5  2.3999   6.0  391.0   
500   0.22438   0.0   9.69   0.0  0.585  6.027   79.7  2.4982   6.0  391.0   
501   0.06263   0.0  11.93   0.0  0.573  6.593   69.1  2.4786   1.0  273.0   
502   0.04527   0.0  11.93   0.0  0.573  6.120   76.7  2.2875   1.0  273.0   
503   0.06076   0.0  11.93   0.0  0.573  6.976   91.0  2.1675   1.0  273.0   
504   0.10959   0.0  11.93   0.0  0.573  6.794   89.3  2.3889   1.0  273.0   
505   0.04741   0.0  11.93   0.0  0.573  6.030   80.8  2.5050   1.0  273.0   

     PTRATIO       B  LSTAT  MEDIDA  
0       15.3  396.90   4.98    24.0  
1       17.8  396.90   9.14    21.6  
2       17.8  392.83   4.03    34.7  
3       18.7  394.63   2.94    33.4  
4       18.7  396.90   5.33    36.2  
5       18.7  394.12   5.21    28.7  
6       15.2  395.60  12.43    22.9  
7       15.2  396.90  19.15    27.1  
8       15.2  386.63  29.93    16.5  
9       15.2  386.71  17.10    18.9  
10      15.2  392.52  20.45    15.0  
11      15.2  396.90  13.27    18.9  
12      15.2  390.50  15.71    21.7  
13      21.0  396.90   8.26    20.4  
14      21.0  380.02  10.26    18.2  
15      21.0  395.62   8.47    19.9  
16      21.0  386.85   6.58    23.1  
17      21.0  386.75  14.67    17.5  
18      21.0  288.99  11.69    20.2  
19      21.0  390.95  11.28    18.2  
20      21.0  376.57  21.02    13.6  
21      21.0  392.53  13.83    19.6  
22      21.0  396.90  18.72    15.2  
23      21.0  394.54  19.88    14.5  
24      21.0  394.33  16.30    15.6  
25      21.0  303.42  16.51    13.9  
26      21.0  376.88  14.81    16.6  
27      21.0  306.38  17.28    14.8  
28      21.0  387.94  12.80    18.4  
29      21.0  380.23  11.98    21.0  
..       ...     ...    ...     ...  
476     20.2  396.21  18.68    16.7  
477     20.2  349.48  24.91    12.0  
478     20.2  379.70  18.03    14.6  
479     20.2  383.32  13.11    21.4  
480     20.2  396.90  10.74    23.0  
481     20.2  393.07   7.74    23.7  
482     20.2  395.28   7.01    25.0  
483     20.2  392.92  10.42    21.8  
484     20.2  370.73  13.34    20.6  
485     20.2  388.62  10.58    21.2  
486     20.2  392.68  14.98    19.1  
487     20.2  388.22  11.45    20.6  
488     20.1  395.09  18.06    15.2  
489     20.1  344.05  23.97     7.0  
490     20.1  318.43  29.68     8.1  
491     20.1  390.11  18.07    13.6  
492     20.1  396.90  13.35    20.1  
493     19.2  396.90  12.01    21.8  
494     19.2  396.90  13.59    24.5  
495     19.2  393.29  17.60    23.1  
496     19.2  396.90  21.14    19.7  
497     19.2  396.90  14.10    18.3  
498     19.2  396.90  12.92    21.2  
499     19.2  395.77  15.10    17.5  
500     19.2  396.90  14.33    16.8  
501     21.0  391.99   9.67    22.4  
502     21.0  396.90   9.08    20.6  
503     21.0  396.90   5.64    23.9  
504     21.0  393.45   6.48    22.0  
505     21.0  396.90   7.88    11.9  

[506 rows x 14 columns]

---------------------------------------------------------------------------------------------------------


################# Totalidade dos dados ###########################

{'target': array([24. , 21.6, 34.7, 33.4, 36.2, 28.7, 22.9, 27.1, 16.5, 18.9, 15. ,
       18.9, 21.7, 20.4, 18.2, 19.9, 23.1, 17.5, 20.2, 18.2, 13.6, 19.6,
       15.2, 14.5, 15.6, 13.9, 16.6, 14.8, 18.4, 21. , 12.7, 14.5, 13.2,
       13.1, 13.5, 18.9, 20. , 21. , 24.7, 30.8, 34.9, 26.6, 25.3, 24.7,
       21.2, 19.3, 20. , 16.6, 14.4, 19.4, 19.7, 20.5, 25. , 23.4, 18.9,
       35.4, 24.7, 31.6, 23.3, 19.6, 18.7, 16. , 22.2, 25. , 33. , 23.5,
       19.4, 22. , 17.4, 20.9, 24.2, 21.7, 22.8, 23.4, 24.1, 21.4, 20. ,
       20.8, 21.2, 20.3, 28. , 23.9, 24.8, 22.9, 23.9, 26.6, 22.5, 22.2,
       23.6, 28.7, 22.6, 22. , 22.9, 25. , 20.6, 28.4, 21.4, 38.7, 43.8,
       33.2, 27.5, 26.5, 18.6, 19.3, 20.1, 19.5, 19.5, 20.4, 19.8, 19.4,
       21.7, 22.8, 18.8, 18.7, 18.5, 18.3, 21.2, 19.2, 20.4, 19.3, 22. ,
       20.3, 20.5, 17.3, 18.8, 21.4, 15.7, 16.2, 18. , 14.3, 19.2, 19.6,
       23. , 18.4, 15.6, 18.1, 17.4, 17.1, 13.3, 17.8, 14. , 14.4, 13.4,
       15.6, 11.8, 13.8, 15.6, 14.6, 17.8, 15.4, 21.5, 19.6, 15.3, 19.4,
       17. , 15.6, 13.1, 41.3, 24.3, 23.3, 27. , 50. , 50. , 50. , 22.7,
       25. , 50. , 23.8, 23.8, 22.3, 17.4, 19.1, 23.1, 23.6, 22.6, 29.4,
       23.2, 24.6, 29.9, 37.2, 39.8, 36.2, 37.9, 32.5, 26.4, 29.6, 50. ,
       32. , 29.8, 34.9, 37. , 30.5, 36.4, 31.1, 29.1, 50. , 33.3, 30.3,
       34.6, 34.9, 32.9, 24.1, 42.3, 48.5, 50. , 22.6, 24.4, 22.5, 24.4,
       20. , 21.7, 19.3, 22.4, 28.1, 23.7, 25. , 23.3, 28.7, 21.5, 23. ,
       26.7, 21.7, 27.5, 30.1, 44.8, 50. , 37.6, 31.6, 46.7, 31.5, 24.3,
       31.7, 41.7, 48.3, 29. , 24. , 25.1, 31.5, 23.7, 23.3, 22. , 20.1,
       22.2, 23.7, 17.6, 18.5, 24.3, 20.5, 24.5, 26.2, 24.4, 24.8, 29.6,
       42.8, 21.9, 20.9, 44. , 50. , 36. , 30.1, 33.8, 43.1, 48.8, 31. ,
       36.5, 22.8, 30.7, 50. , 43.5, 20.7, 21.1, 25.2, 24.4, 35.2, 32.4,
       32. , 33.2, 33.1, 29.1, 35.1, 45.4, 35.4, 46. , 50. , 32.2, 22. ,
       20.1, 23.2, 22.3, 24.8, 28.5, 37.3, 27.9, 23.9, 21.7, 28.6, 27.1,
       20.3, 22.5, 29. , 24.8, 22. , 26.4, 33.1, 36.1, 28.4, 33.4, 28.2,
       22.8, 20.3, 16.1, 22.1, 19.4, 21.6, 23.8, 16.2, 17.8, 19.8, 23.1,
       21. , 23.8, 23.1, 20.4, 18.5, 25. , 24.6, 23. , 22.2, 19.3, 22.6,
       19.8, 17.1, 19.4, 22.2, 20.7, 21.1, 19.5, 18.5, 20.6, 19. , 18.7,
       32.7, 16.5, 23.9, 31.2, 17.5, 17.2, 23.1, 24.5, 26.6, 22.9, 24.1,
       18.6, 30.1, 18.2, 20.6, 17.8, 21.7, 22.7, 22.6, 25. , 19.9, 20.8,
       16.8, 21.9, 27.5, 21.9, 23.1, 50. , 50. , 50. , 50. , 50. , 13.8,
       13.8, 15. , 13.9, 13.3, 13.1, 10.2, 10.4, 10.9, 11.3, 12.3,  8.8,
        7.2, 10.5,  7.4, 10.2, 11.5, 15.1, 23.2,  9.7, 13.8, 12.7, 13.1,
       12.5,  8.5,  5. ,  6.3,  5.6,  7.2, 12.1,  8.3,  8.5,  5. , 11.9,
       27.9, 17.2, 27.5, 15. , 17.2, 17.9, 16.3,  7. ,  7.2,  7.5, 10.4,
        8.8,  8.4, 16.7, 14.2, 20.8, 13.4, 11.7,  8.3, 10.2, 10.9, 11. ,
        9.5, 14.5, 14.1, 16.1, 14.3, 11.7, 13.4,  9.6,  8.7,  8.4, 12.8,
       10.5, 17.1, 18.4, 15.4, 10.8, 11.8, 14.9, 12.6, 14.1, 13. , 13.4,
       15.2, 16.1, 17.8, 14.9, 14.1, 12.7, 13.5, 14.9, 20. , 16.4, 17.7,
       19.5, 20.2, 21.4, 19.9, 19. , 19.1, 19.1, 20.1, 19.9, 19.6, 23.2,
       29.8, 13.8, 13.3, 16.7, 12. , 14.6, 21.4, 23. , 23.7, 25. , 21.8,
       20.6, 21.2, 19.1, 20.6, 15.2,  7. ,  8.1, 13.6, 20.1, 21.8, 24.5,
       23.1, 19.7, 18.3, 21.2, 17.5, 16.8, 22.4, 20.6, 23.9, 22. , 11.9]), 'feature_names': array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
       'TAX', 'PTRATIO', 'B', 'LSTAT'], dtype='<U7'), 'data': array([[6.3200e-03, 1.8000e+01, 2.3100e+00, ..., 1.5300e+01, 3.9690e+02,
        4.9800e+00],
       [2.7310e-02, 0.0000e+00, 7.0700e+00, ..., 1.7800e+01, 3.9690e+02,
        9.1400e+00],
       [2.7290e-02, 0.0000e+00, 7.0700e+00, ..., 1.7800e+01, 3.9283e+02,
        4.0300e+00],
       ...,
       [6.0760e-02, 0.0000e+00, 1.1930e+01, ..., 2.1000e+01, 3.9690e+02,
        5.6400e+00],
       [1.0959e-01, 0.0000e+00, 1.1930e+01, ..., 2.1000e+01, 3.9345e+02,
        6.4800e+00],
       [4.7410e-02, 0.0000e+00, 1.1930e+01, ..., 2.1000e+01, 3.9690e+02,
        7.8800e+00]]), 'DESCR': "Boston House Prices dataset\n===========================\n\nNotes\n------\nData Set Characteristics:  \n\n    :Number of Instances: 506 \n\n    :Number of Attributes: 13 numeric/categorical predictive\n    \n    :Median Value (attribute 14) is usually the target\n\n    :Attribute Information (in order):\n        - CRIM     per capita crime rate by town\n        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.\n        - INDUS    proportion of non-retail business acres per town\n        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)\n        - NOX      nitric oxides concentration (parts per 10 million)\n        - RM       average number of rooms per dwelling\n        - AGE      proportion of owner-occupied units built prior to 1940\n        - DIS      weighted distances to five Boston employment centres\n        - RAD      index of accessibility to radial highways\n        - TAX      full-value property-tax rate per $10,000\n        - PTRATIO  pupil-teacher ratio by town\n        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town\n        - LSTAT    % lower status of the population\n        - MEDV     Median value of owner-occupied homes in $1000's\n\n    :Missing Attribute Values: None\n\n    :Creator: Harrison, D. and Rubinfeld, D.L.\n\nThis is a copy of UCI ML housing dataset.\nhttp://archive.ics.uci.edu/ml/datasets/Housing\n\n\nThis dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.\n\nThe Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic\nprices and the demand for clean air', J. Environ. Economics & Management,\nvol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics\n...', Wiley, 1980.   N.B. Various transformations are used in the table on\npages 244-261 of the latter.\n\nThe Boston house-price data has been used in many machine learning papers that address regression\nproblems.   \n     \n**References**\n\n   - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.\n   - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.\n   - many more! (see http://archive.ics.uci.edu/ml/datasets/Housing)\n"}
# gráfico que mostra a relação
# conventional way to import seaborn
import seaborn as sns

# allow plots to appear within the notebook
%matplotlib inline
# visualize the relationship between the features and the response using scatterplots
sns.pairplot(dataset_boston, x_vars=['CRIM','ZN','INDUS','CHAS'], y_vars='MEDIDA',size=7, aspect=0.7, kind='reg')
sns.pairplot(dataset_boston, x_vars=['NOX','RM','AGE','DIS'], y_vars='MEDIDA',size=7, aspect=0.7, kind='reg')
sns.pairplot(dataset_boston, x_vars=['RAD','TAX','PTRATIO','B','LSTAT'], y_vars='MEDIDA',size=6, aspect=0.7, kind='reg')
/home/nbuser/anaconda3_420/lib/python3.5/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
<seaborn.axisgrid.PairGrid at 0x7f8815a50278>



#Preparando X y Usando o Pandas
# create a Python list of feature names
# criando um cabeçalho - dataset
feature_cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']

# use the list to select a subset of the original DataFrame
X = dataset_boston[feature_cols]

# equivalent command to do this in one line
X = dataset_boston[['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']]

# print the first 5 rows
X.head()
CRIM	ZN	INDUS	CHAS	NOX	RM	AGE	DIS	RAD	TAX	PTRATIO	B	LSTAT
0	0.00632	18.0	2.31	0.0	0.538	6.575	65.2	4.0900	1.0	296.0	15.3	396.90	4.98
1	0.02731	0.0	7.07	0.0	0.469	6.421	78.9	4.9671	2.0	242.0	17.8	396.90	9.14
2	0.02729	0.0	7.07	0.0	0.469	7.185	61.1	4.9671	2.0	242.0	17.8	392.83	4.03
3	0.03237	0.0	2.18	0.0	0.458	6.998	45.8	6.0622	3.0	222.0	18.7	394.63	2.94
4	0.06905	0.0	2.18	0.0	0.458	7.147	54.2	6.0622	3.0	222.0	18.7	396.90	5.33
# Tipos de dados X y
# check the type and shape of X
print(type(X))
print(X.shape)
#linhas e colunas a partir deste cabeçalho
<class 'pandas.core.frame.DataFrame'>
(506, 13)
# série do conjunto de dados - dataset - Medida
# select a Series from the DataFrame
y = dataset_boston['MEDIDA']

# equivalent command that works if there are no spaces in the column name
y = dataset_boston.MEDIDA

# print the first 5 values
y.head()
0    24.0
1    21.6
2    34.7
3    33.4
4    36.2
Name: MEDIDA, dtype: float64
# Splitting X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
# default split is 75% for training and 25% for testing
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
(379, 13)
(379,)
(127, 13)
(127,)
#Linear regression in scikit-learn
# import model
from sklearn.linear_model import LinearRegression

# instantiate
linreg = LinearRegression()

# fit the model to the training data (learn the coefficients)
linreg.fit(X_train, y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
# Splitting X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
# Train e test
# default split is 75% for training and 25% for testing
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
(379, 13)
(379,)
(127, 13)
(127,)
############################################
# make predictions on the testing set
y_pred = linreg.predict(X_test)
# define true and predicted response values
true = [100, 50, 30, 20]
pred = [90, 50, 50, 30]
#MAE
# calculate MAE by hand
print((10 + 0 + 20 + 10)/4.)

# calculate MAE using scikit-learn
from sklearn import metrics
print(metrics.mean_absolute_error(true, pred))
10.0
10.0
#MSE
# calculate MSE by hand
print((10**2 + 0**2 + 20**2 + 10**2)/4.)

# calculate MSE using scikit-learn
print(metrics.mean_squared_error(true, pred))
150.0
150.0
# calculate RMSE by hand
import numpy as np
print(np.sqrt((10**2 + 0**2 + 20**2 + 10**2)/4.))

# calculate RMSE using scikit-learn
print(np.sqrt(metrics.mean_squared_error(true, pred)))
12.24744871391589
12.24744871391589
# intercept and coefficients
# Interpreting model coefficients¶
# print the intercept and coefficients
print(linreg.intercept_)
print(linreg.coef_)
45.236415846056985
[-1.13256952e-01  5.70869807e-02  3.87621062e-02  2.43279795e+00
 -2.12706290e+01  2.86930027e+00  7.02105327e-03 -1.47118312e+00
  3.05187368e-01 -1.06649888e-02 -9.97404179e-01  6.39833822e-03
 -5.58425480e-01]
# feature names with the coefficients
# pair the feature names with the coefficients
list(zip(feature_cols, linreg.coef_))
# Retirar o ('B', 0.006398338224938453)
[('CRIM', -0.11325695150325484),
 ('ZN', 0.057086980673967454),
 ('INDUS', 0.0387621061705701),
 ('CHAS', 2.4327979454811017),
 ('NOX', -21.270629005497163),
 ('RM', 2.8693002671025525),
 ('AGE', 0.007021053271749427),
 ('DIS', -1.4711831191291065),
 ('RAD', 0.3051873675841825),
 ('TAX', -0.01066498878281223),
 ('PTRATIO', -0.9974041787728967),
 ('B', 0.006398338224938453),
 ('LSTAT', -0.5584254800083318)]
# make predictions on the testing set
y_pred = linreg.predict(X_test)
# Model evaluation metrics for regression
# define true and predicted response values
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
4.678607638226911
# Margin de erro RMSE
# create a Python list of feature names
feature_cols= ['CRIM','ZN','INDUS','CHAS','RM','AGE','DIS','RAD','TAX','B','PTRATIO','LSTAT'] # 'B' Retirado das features

# use the list to select a subset of the original DataFrame
X = dataset_boston[feature_cols]

# select a Series from the DataFrame
y = dataset_boston.MEDIDA

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# fit the model to the training data (learn the coefficients)
linreg.fit(X_train, y_train)

# make predictions on the testing set
y_pred = linreg.predict(X_test)

# compute the RMSE of our predictions
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
4.604452983843312
# CRIM — Taxa de crime per capita por cidade,
# ZN — Proporção de lotes residenciais num raio de 25,000 pés quadrados,
# INDUS — Proporção de acres para negócios que não são varejo por cidade,
# CHAS — Proximidade com o rio (1 se o lote toca o rio, 0 se não),
# NOX — Concentração de óxidos nítricos (partes por 10 milhões),
# RM — Numero médio de quartos por residência,
# AGE — Proporção de unidades construídas antes de 1940,
# DIS — Distância ponderada dos cinco grandes centros de trabalho,
# RAD — Índice de acessibilidade à estradas radiais,
# TAX — Razão de imposto sobre valor total da propriedade,
# PTRATIO — Razão de pupilos por professor na cidade,
# B 1000(Bk — 0.63)² — onde Bk é a proporção de negros por cidade,
# LSTAT — % porcentagem de status inferior da população,
# MEDIDA — Valor mediano das casas ocupadas em milhares,
# Usamos o RMSE . Dividimos o conjunto em train e test. O NOX — Concentração de óxidos nítricos (partes por 10 milhões) foi retirado, da matriz xy, dos atributos, pois tem uma correlação fraca com o valor mediano - 'MEDIDA'.
# Assim, temos uma melhor precisão diminuindo a taxa de erro.
