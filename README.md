# Classification of Fermi LAT Gamma-ray Sources from the FL8Y Catalog using Machine Learning Techniques

To use the code, follow the instructions below.

## Prerequisites

1. Clone all the file to your local computer.

2. Make sure you have the required libraries installed for your Python 3 ([NumPy](http://www.numpy.org/), [Pandas](https://pandas.pydata.org/), [scikit-learn](https://scikit-learn.org/) and pickle [pickle is installed for Python 3 by default]).

## AGN-PSR Classification
1. For AGN-PSR Classification, run

```
python3 AGN_PSR_Training.py
```
The overall training results will be stored in AGN_PSR_result.txt.

2. To classify the whole catalog, run

```
python3 AGN_PSR_Result.py
```

The classified catalog will be stored in results.csv. The list of PSR for the two different models (90.0%-up-model and top-3-model) will be stored in psr_candidates_90_up_model.txt and psr_candidates_top_3_model.txt.

## YNG-MSP Classification
1. (Optional) If you don't have the file psr_list.txt for the list of YNG/MSP, run 

```
g++ gen_psr_list.cpp -o gen_psr_list
./gen_psr_list
```

2. For YNG-MSP Classification, run

```
python3 YNG_MSP_Training.py
```

The overall training results will be stored in YNG_MSP_result.txt.

3. To classify the PSR candidates (Make sure you have run the AGN-PSR Classification for the list of PSR candidates), run

```
python3 YNG_MSP_Result.py
```

The classified catalog will be stored in results_msp.csv. The list of MSP for the two different models (90.0%-up-model and top-3-model) will be stored in msp_candidates_90_up_model.txt and msp_candidates_top_3_model.txt.