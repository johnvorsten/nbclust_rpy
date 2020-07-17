# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 21:21:52 2019

@author: z003vrzk
"""
"""The NbClust package must be installed in the R working directory
If it is not then this module wont work
This checks to see if NbClust is installed in this environment

With anaconda, R is installed at
C:\ProgramData\Anaconda3\envs\<environment name>\Lib\R\bin\R.exe
Modules are installed at
C:\ProgramData\Anaconda3\envs\tf\Lib\R\library

To install R in an anaconda environment
conda activate <environment name>
conda insatll r-essentials

To install NbClust in an anaconda environment
conda activate <environment name>
r # Enter R command prompt
install.packages("NbClust","C:\\ProgramData\\Anaconda3\\envs\\<environment name>\\Lib\\R\\library")

Check if the package is installed
r
> library() # Check if NbClust is listed, or
> library("NbClust")

OR Alternatively install from python with the R utils package
# import rpy2's package module
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector

# import R's utility package
utils = rpackages.importr('utils')

# select a mirror for R packages
utils.chooseCRANmirror(ind=1) # select the first mirror in the list

utils.install_packages(StrVector(['NbClust']))
"""

# Python imports
from collections import namedtuple

# Third party import
import rpy2
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
import pandas as pd
import numpy as np

# Local import

# Global declarations
module_name = 'NbClust'
if not rpy2.robjects.packages.isinstalled(module_name):
    raise ImportError('{} is not installed'.format(module_name))
nbclust = importr('NbClust')  #Import the NbClust package
numpy2ri.activate()
NbClustResults = namedtuple('NbClustResults', ['index_df', 'best_nc_df'])

#%%


def r_matrix_to_df(r_matrix):
    """Convert an rpy2.robjects.Matrix to a pandas dataframe.
    This method assumed the matrix is 2D and has named dimensions
    for each of the clustering indexes and number of test clusters
    inputs
    -------
    r_matrix : (rpy2.robjects.Matrix) output of NbClust module with named
    dimensions
    outputs
    -------
    df : (pandas.DataFrame) results matrix with
        values :
        columns :
        index : """

    assert type(r_matrix) is rpy2.robjects.Matrix, 'r_matrix argument is not\
    type rpy2.robjects.Matrix'

    values = np.array(r_matrix)
    row_names = list(r_matrix.rownames)
    col_names = list(r_matrix.colnames)
    df = pd.DataFrame(data=values, index=row_names, columns=col_names)
    return df


def r_listvector_to_df(r_list):
    """Method for converting the NbClust ListVector to a pandas dataframe.
    This method is used when only one clustering index is returned from NbClust.
    Usually an R Matrix is returned, but if only a single index is selected
    then a ListVector is returned instead
    This method assumes the ListVector has named objects
    ['All.index', 'Best.nc', 'Best.partition'].

    inputs
    -------
    r_list : (rpy2.robjects.vectors.ListVector) R List vector with names
    ['All.index', 'Best.nc', 'Best.partition']
    All.index is a list of the calculated clustering index values
    Best.nc is a rpy2.robjects.vectors.FloatVector which should have 2 values
    [best_n_clusters, index_value].
    The names of All.index FloatVector names should be
    ['Number_clusters', 'Value_Index']

    Best.partition is the best number of clusters at each of the test
    number of clusters. For example, given min_nc=2, max_nc=5, Best.Partion
    would give the best number of clusters at each of 2,3,4,5 test number of
    clusters

    outputs
    --------
    df : (pandas.DataFrame) Best number of cluster dictionary with
        values :
        columns :
        index : """

    msg = 'r_list argument is not type rpy2.robjects.vectors.ListVector'
    assert isinstance(r_list, rpy2.robjects.vectors.ListVector), msg

    values = np.array(r_list.__getitem__(0))
    row_names = list(r_list.names)
    col_names = list(r_list.colnames)
    df = pd.DataFrame(data=values, index=row_names, columns=col_names)
    return df


def r_floatvector_to_df(r_vector, col_names=None, row_names=None):
    """Method for converting the NbClust FloatVector to a pandas dataframe.
    This method is used when only one clustering index is returned from NbClust.
    Usually an R Matrix is returned, but if only a single index is selected
    then a FloatVector is returned instead
    The NbClust results is a ListVector with named objects
    ['All.index', 'Best.nc', 'Best.partition']
    Pass a single object in the list to this method like
    index_df = r_floatvector_to_df(result.__getitem__(0))

    inputs
    -------
    r_vector : (rpy2.robjects.vectors.FloatVector) R Float vector
    All.index is a list of the calculated clustering index values
    Best.nc is a rpy2.robjects.vectors.FloatVector which should have 2 values
    [best_n_clusters, index_value].
    The names of All.index FloatVector names should be
    ['Number_clusters', 'Value_Index']

    Best.partition is the best number of clusters at each of the test
    number of clusters. For example, given min_nc=2, max_nc=5, Best.Partion
    would give the best number of clusters at each of 2,3,4,5 test number of
    clusters
    col_names : (list) of str objects to name columns of the passed FloatVector
    row_names : (list) of str or int to name the dataframe index

    outputs
    --------
    df : (pandas.DataFrame) Best number of cluster dictionary with
        values :
        columns :
        index : """

    msg = 'r_vector argument is not type rpy2.robjects.vectors.FloatVector'
    assert isinstance(r_vector, rpy2.robjects.vectors.FloatVector), msg

    values = np.array(r_vector)

    if row_names is None:
        row_names = list(r_vector.names)

    if col_names is None:
        col_names = list(r_vector.colnames)

    df = pd.DataFrame(data=values, index=row_names, columns=col_names)
    return df


def nbclust_calc(data,
                 min_nc,
                 max_nc,
                 distance='euclidean',
                 clusterer='kmeans',
                 index='all'):
    """Uses the R package NbClust to find the optimal number of clusters
    in a dataset.  Returns the results in a pandas dataframe
    inputs
    -------
    data : (np.array) numpy array of your data of shape (n,p) where n is the
    number of instances and p is the number of features
    min_nc : (int) NbClust parameter, minimum number of clusters to test
    max_nc : (int) NbClust parameter, maximum number of clusters to test
    distance : (str) distance measurement between instances. See NbClust inputs
    method : (str) clustering method to use. See NbClust inputs
    index : (str) indicies to return from NbClust. See NbClust inputs
    see https://www.rdocumentation.org/packages/NbClust/versions/3.0/topics/NbClust
    for information on parameters
    outputs
    -------
    NbClustResults : (namedtuple) tuple of index_df and best_nc_df
    index_df : (pd.DataFrame) Number of clusters and the best cluster prediction
    metric
    best_nc_df : (pd.DataFrame)  The best number of clusters for each indicy in
    index_df
    see the NbClust documentation for more information"""
    """Result is a ListVector (robjects.ListVector)
    It has named objects All.index, All.CriticalValues, Best.nc Best.partition
    if multiple index arguments are passed
    If only one index argument is passed then the ListVector contains
    ['All.index', 'Best.nc', 'Best.partition']"""
    result = nbclust.NbClust(data,
                             distance=distance,
                             min_nc=min_nc,
                             max_nc=max_nc,
                             method=clusterer,
                             index=index)
    """The first item in in result is a rpy2.robjects.vectors.FloatVector if
    only one index was passed
    Named elements include ['All.index', 'Best.nc', 'Best.partition']"""
    if isinstance(result.__getitem__(0), rpy2.robjects.vectors.FloatVector):
        # All.index
        row_names = list(result.__getitem__(0).names)
        col_names = [index]  # index, cant access from result.__getitem__(0)
        index_df = r_floatvector_to_df(result.__getitem__(0),
                                       col_names=col_names,
                                       row_names=row_names)

        # Best.nc
        row_names = result.__getitem__(1).names
        col_names = [index]  # index, cant access from result.__getitem__(0)
        best_nc_df = r_floatvector_to_df(result.__getitem__(1),
                                         col_names=col_names,
                                         row_names=row_names)
    """The first item in in result is a rpy2.robjects.vectors.Matrix if
    multiple index were passed
    Named elements include ['All.index', 'Best.nc', 'Best.partition']"""
    if isinstance(result.__getitem__(0), rpy2.robjects.Matrix):
        # All.index
        index_df = r_matrix_to_df(result.__getitem__(0))

        # Best.nc
        best_nc_df = r_matrix_to_df(result.__getitem__(2))

    return NbClustResults(index_df, best_nc_df)
