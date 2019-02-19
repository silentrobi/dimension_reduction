
from tkFont import names
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
import time


#spliting features

def splitFeature(X, size):
    data = X
    split= size
   # np.random.shuffle(data) #shuffling data by row
    train_data= data[:split, :]
    return train_data





#PCA
def pca(X, n_components): # input data

    pca = PCA(copy=True, iterated_power='auto', n_components=n_components, random_state=None,
    svd_solver='full', tol=0.0, whiten=False)
    principalComponents = pca.fit_transform(X)
    #calculating total information parcentage in n_component
    varienceArray= pca.explained_variance_ratio_ * 100
    sum = 0
    for i in varienceArray:
        sum +=i
    print "Total containing informanation in the ",n_components, " is :", sum
    print "varience ratio of each components: "
    print pca.explained_variance_ratio_
    return principalComponents





def scatterplot(X, x_label="", y_label="", title="", color = "r", yscale_log=False):

    # Create the plot object
    if (X.shape[1] ==2):
        _, ax = plt.subplots()

        # Plot the data, set the size (s), color and transparency (alpha)
        # of the points
        ax.scatter(X[:,0], X[:,1], s = 10, color = color, alpha = 0.75)

        if yscale_log == True:
            ax.set_yscale('log')

    # Label the axes and provide a title
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.show()
    else:
        print "Array diamention should be 2D"

#MDS
def mds(X, n_components):
    mds = MDS( n_components= n_components)
    principalComponents = mds.fit_transform(X)
    # calculating total information parcentage in n_component

    print "ji"
    return principalComponents


#Isomap
def isomap(X,n_components,n_neighbors ):
    iso = Isomap(n_neighbors=n_neighbors,
                          n_components=n_components)  # We will fit a manifold using 6 nearest neighbours and our aim is to reduce down to 2 components.
    principleComponents = iso.fit_transform(X)
    return principleComponents

#LLE
def lle(X,n_components,n_neighbors):
    lle= LocallyLinearEmbedding(n_components=n_components,n_neighbors=n_neighbors)
    principleComponents = lle.fit_transform(X)
    return principleComponents
#choosong number of component
def component(dataset):
    pca = PCA().fit(dataset)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()


def main():
    start_time = time.time()


    # load dataset into Pandas DataFrame
    df = pd.read_csv('53727641925dat.txt', skipinitialspace=True, sep=" ", low_memory=False,
                     names=["USAF", "WBAN", "YR--MODAHRMN", "DIR", "SPD", 'GUS', "CLG", "SKC",
                            "L", "M", "H", "VSB", "MW1", 'MW2', 'MW3', 'MW4', 'AW1', 'AW2', 'AW3',
                            'AW4', 'W', 'TEMP', 'DEWP', 'SLP', 'ALT', 'STP', 'MAX', 'MIN', 'PCP01',
                            'PCP06', 'PCP24', 'PCPXX', 'SD'])
    # SKC = SKY COVER -- CLR-CLEAR, SCT-SCATTERED-1/8 TO 4/8,   BKN-BROKEN-5/8 TO 7/8, OVC-OVERCAST,
    df['SKC'] = df['SKC'].map({'CLR': 0, 'SCT': 1, 'BKN': 2, 'OVC': 3, 'SKC': 'SKC'})

    df = df[["DIR", "SPD", 'GUS', "CLG", "SKC", "L",
             "M", "H", "VSB", "MW1", 'MW2', 'MW3', 'MW4', 'AW1', 'AW2', 'AW3', 'AW4', 'W',
             'TEMP', 'DEWP', 'SLP', 'ALT', 'STP', 'MAX', 'MIN', 'PCP01', 'PCP06',
             'PCP24', 'PCPXX', 'SD']]
    df[["DIR", "SPD", 'GUS', "CLG", "SKC", "L",
        "M", "H", "VSB", "MW1", 'MW2', 'MW3', 'MW4', 'AW1', 'AW2', 'AW3', 'AW4', 'W',
        'TEMP', 'DEWP', 'SLP', 'ALT', 'STP', 'MAX', 'MIN', 'PCP01', 'PCP06',
        'PCP24', 'PCPXX', 'SD']] = df[["DIR", "SPD", 'GUS', "CLG", "SKC",
                                       "L", "M", "H", "VSB", "MW1", 'MW2', 'MW3', 'MW4', 'AW1', 'AW2',
                                       'AW3', 'AW4', 'W', 'TEMP', 'DEWP', 'SLP', 'ALT', 'STP', 'MAX', 'MIN',
                                       'PCP01', 'PCP06', 'PCP24', 'PCPXX', 'SD']].apply(pd.to_numeric, errors='coerce')
    X = df.as_matrix()
    X = np.delete(X, 0, 0)

    # dealing with NaN values

    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    X = imputer.fit_transform(X)  # colums containing all NaN will be removed(column)

    # feature scaling
    X = MinMaxScaler().fit_transform(X)
    # training data
    training_data = splitFeature(X, 1500)

    # calling PCA
    # principleComponentPCA= pca(training_data,10)




    components=lle(training_data,2,6)


    print("--- %s seconds ---" % (time.time() - start_time))
    scatterplot(components)
main()






"""
Comments:
(i) Execution time:
     PCA takes : 4.05700016022 seconds
     MDS takes : 126.655999899 seconds
     Isomap takes : 6.09299993515 seconds
     LLE takes : 4.81200003624 seconds

(ii) PCA:
    Total containing informanation in the  2 principle Component is : 76.7898771945 %
    varience ratio of each components: 
    [0.56771579 0.20018298]
    To get over 90% total varience at least 5 principle Components is needed.
    PCA yields the directions (principal components) that maximize the variance of the data.
    Provides good performance.
    Computationally fast.
 (iii) MDS:
        MDS is actually done by transforming distances into similarities and performing PCA (eigen-decomposition or singular-value-decomposition) on those.
        To get over 90% total varience at least 5 principle Components is needed.
        It is computationally slowest.    
 (iv) Isomap:
        Isomap is a non-linear dimensionality reduction method based on the spectral theory which tries to preserve the geodesic distances in the lower dimension.
        Data fromed into diffenent cluster. Giving good performance.
 (V) LLE:
         Here , LLE is taking advantage of the local geometry  and pieces it together to preserve the global geometry ona lower diamensioal space. I have used standard method of LLE.   
        LLE had may add little bit value compared to PCA.
        Computationally slower than PCA 

Scatter plot is attached with the zip file and number of component graph
"""