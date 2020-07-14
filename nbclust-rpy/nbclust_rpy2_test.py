# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 20:30:34 2019

@author: z003vrzk
"""

# Python imports

# Third party imports
from rpy2 import robjects
from rpy2.robjects.numpy2ri import numpy2ri
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import rpy2
import numpy as np

# Local imports
import nbclust_rpy2

# Local declarations
rpy2.robjects.numpy2ri.activate()

#%%

def test_import_R_pi():
    """Import pi object from R. The R pi object is a float vector
    """

    # Float vector object
    pi_object = robjects.r.pi
    # Or get an object by name
    pi_object2 = robjects.r['pi']

    pi_val = pi_object[0] # The returned object is indexed
    pi_val = pi_object.__getitem__(0) # Get the first item in the vector

    return pi_object, pi_val

def test_R_function():
    """How to call R functions in python"""

    base = importr('base')
    stats = importr('stats')
    graphics = importr('graphics')

    plot = graphics.plot
    rnorm = stats.rnorm
    plot(rnorm(100), ylab="random")

    return None

def test_NbClust_installed():
    """How to import packages from R into python callables
    Test to see if NbClust is installed"""

    module_name = 'NbClust'
    nbclust_installed = rpy2.robjects.packages.isinstalled(module_name)
    if nbclust_installed:
        print('{} is already Installed'.format(module_name))
    else:
        print('{} needs to be installed'.format(module_name))
        msg=('install.packages("NbClust","C:\\ProgramData\\Anaconda3\\envs\\'+
             '<environment name>\\Lib\\R\\library")')
        print('Try this command in the R command prompt :\n' + msg)

        """Alternatively install from python with the R utils package
        # import rpy2's package module
        import rpy2.robjects.packages as rpackages
        from rpy2.robjects.vectors import StrVector

        # import R's utility package
        utils = rpackages.importr('utils')

        # select a mirror for R packages
        utils.chooseCRANmirror(ind=1) # select the first mirror in the list

        utils.install_packages(StrVector(['NbClust']))
        """

    return nbclust_installed

def test_NbClust_import():
    """Attempt to import the NbClust package"""

    # Import the NbClust package with importr module and implicit path
    nbclust = importr('NbClust')

    # Import the NbClust package with importr module and R module path
    module_path = rpy2.robjects.packages.get_packagepath('NbClust')
    nbclust2 = importr('NbClust', lib_loc=module_path)

    return nbclust

def test_NbClust_call():
    """Import and call the NbClust package with some random data

    Note on passing parameters to NbClust package"""

    # Create test data
    data = np.random.rand(32).reshape(-1,2)

    # Get nbclust object
    nbclust = test_NbClust_import()

    #Print the modules directory
    print('NbClust object callables : ')
    for key, val in nbclust.__dict__.items():
        print(key)

    try:
        answer = nbclust.NbClust(data, diss='NULL', distance='euclidean',
                        method='kmeans', index='all')
    except NotImplementedError as nie:
        print('OOPS! We cant pass a numpy array or dataframe straight to' +
              ' an R package without converting a numpy arraty to an R array'+
              ' first')
        pass


    # Convert numpy object to R matrix
    # from rpy2.robjects.numpy2ri import numpy2ri
    answer = nbclust.NbClust(data, distance='euclidean',
                    min_nc=2, max_nc=10,
                    method='kmeans', index='all')

    # The nbclust package returns an R ListVector
    print('The R package returned an object of type : {}'.format(type(answer)))

    # The ListVector contains the R objects
    print('\nThe ListVector contains R objects : ')
    for key, value in answer.items():
        print('{} | {}'.format(key, type(value)))

    # The following labels are assigned to the returned ListVector
    print('The ListVector contains the names assigned to each object in the list')
    for name in answer.names:
        print(name)

    # The first item in the list is All.index (the clustering index)
    print('The first item in ListVector is named All.index')
    print('All.index is an R data type {}'.format(answer[0].rclass))

    # All.index is an R matrix (not DataFrame)
    # An R matrix is an R array with dimension attributes
    print('All.index is an rpy2 data type of {}'.format(answer[0].__class__))

    # Just like R Vectors, the R Matrix dimensions can be named
    print('All.index contains the column names : ')
    print(answer[0].colnames[:5], 'etc...')
    print('All.index contains the index : ')
    print(answer[0].rownames[:5], 'etc...')

    return answer

def test_rmatrix_to_df():
    """Convert an R matrix to a dataframe"""

    # Generate values, column names, row names, and create R matrix
    values = np.random.rand(30)
    # Resulting shape is 5 x 6
    col_names = robjects.vectors.StrVector(['Col1','Col2','Col3','Col4','Col5','Col6'])
    row_names = robjects.vectors.StrVector(['Row1','Row2','Row3','Row4','Row5'])
    r_matrix = robjects.r.matrix(robjects.FloatVector(values), nrow=5)
    r_matrix._Matrix__rownames_set(row_names)
    r_matrix._Matrix__colnames_set(col_names)
    # r_matrix.colnames
    # r_matrix.rownames

    # Construct the dataframe
    df_values = np.array(r_matrix)
    df_rows_names = list(r_matrix.rownames)
    df_cols_names = list(r_matrix.colnames)
    df = pd.DataFrame(data=df_values,
                      index=df_rows_names,
                      columns=df_cols_names)

    return df

#%% Test my module

def test_multiple_index():
    """Map NbClust calls to output dataframes"""
    data = np.random.rand(50)
    data = data.reshape(-1,2)
    index='all'
    min_nc = 4
    max_nc = 12
    distance='euclidean'
    clusterer='kmeans'
    index='all'

    result = nbclust_rpy2.nbclust_calc(data,
                                       min_nc,
                                       max_nc,
                                       distance=distance,
                                       clusterer=clusterer,
                                       index=index)
    # Field names from namedtuple
    result._fields
    index_df = result.index_df
    best_nc_df = result.best_nc_df

    return result


def test_single_index():
    """Map NbClust calls to output dataframes"""
    data = np.random.rand(50)
    data = data.reshape(-1,2)
    index='KL'
    min_nc = 4
    max_nc = 12
    distance='euclidean'
    clusterer='kmeans'
    index=['kl']

    result = nbclust_rpy2.nbclust_calc(data,
                                       min_nc,
                                       max_nc,
                                       distance=distance,
                                       clusterer=clusterer,
                                       index=index)

    # Field names from namedtuple
    result._fields
    index_df = result.index_df
    best_nc_df = result.best_nc_df

    return result


def test_jvcustom_index():
    """Map NbClust calls to output dataframes"""
    data = np.random.rand(50)
    data = data.reshape(-1,2)
    index='KL'
    min_nc = 4
    max_nc = 12
    distance='euclidean'
    clusterer='kmeans'
    index='jv_custom'

    result = nbclust_rpy2.nbclust_calc(data,
                                       min_nc,
                                       max_nc,
                                       distance=distance,
                                       clusterer=clusterer,
                                       index=index)

    # Field names from namedtuple
    result._fields
    index_df = result.index_df
    best_nc_df = result.best_nc_df

    return result


