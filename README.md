# Classification of Fermi LAT Gamma-ray Sources from the FL8Y Catalog using Machine Learning Techniques

To use the code, follow the instructions below

## Prerequisites

1. Clone all the file to your local computer.

2. Make sure you have the libraries installed for your Python 3 ([NumPy](http://www.numpy.org/), [Pandas](https://pandas.pydata.org/), [scikit-learn](https://scikit-learn.org/) and pickle [pickle is installed for Python 3 by default]).

## AGN-PSR Classification
1. For AGN-PSR Classification, run

```
python3 AGN_PSR_Training.py
```

2. The overall training results will be stored in AGN_PSR_result.txt.

3. To classify the catalog, run

```
python3 AGN_PSR_Result.py
```

## YNG-MSP Classification
1. (Optional) If you don't have the file psr_list.txt for the list of YNG/MSP, run 

```
(Constructing)
```

2. For YNG-MSP Classification, run

```
python3 YNG_MSP_Training.py
```

3. The overall results will be stored in YNG_MSP_result.txt.

4. To classify the catalog, run

```
python3 YNG_MSP_Result.py
```
