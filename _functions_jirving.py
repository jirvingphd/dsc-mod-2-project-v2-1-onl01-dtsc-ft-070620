"""Collection of functions for Linear Regression analysis
James M Irving, Ph.D.
https://github.com/jirvingphd
james.irving.phd@gmail.com
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
inline_rc = dict(mpl.rcParams)
from IPython.display import display
# MULTIPLOT
from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# plt.style.use('seaborn-talk')

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import + altest as normtest # D'Agostino and Pearson's omnibus test

from collections import Counter
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
# Check columns returns the datatype, null values and unique values of input series 

def value_counts(column,dropna=False,normalize=True):
    """Modified default params for pandas value_counts. 
    Includes null values and displays %'s by default.'"""
    return column.value_counts(dropna=dropna,normalize=normalize)


def check_column(series,nlargest='all'):
    """Displays column name, data type, null value counts + % null.
    Also displays the normalized value counts with Null values included."""
    print(f"Column: df['{series.name}']':")
    print(f"dtype: {series.dtype}")
    print(f"isna: {series.isna().sum()} out of {len(series)} - {round(series.isna().sum()/len(series)*100,3)}%")
        
    print(f'\nNormalized Value Counts:') #,df['waterfront'].unique())
    if nlargest =='all':
        display(value_counts(series))#series.value_counts())
    else:
        display(value_counts(series).nlargest(nlargest))
        
# define log + z-score
def log_z(col):
    """Logs and z-scores input series"""
    col = np.array(col)
    logcol = np.log(col)
    
    zlogcol = (logcol-np.mean(logcol))/np.sqrt(np.var(logcol))
    
    return zlogcol


def rem_out_z(col_name):
    """Slices out outliers with an  absolute zscore > 3"""
    col = np.array(col_name)
    z_col = (col - np.mean(col)) / np.sqrt(np.var(col))
    z_col[abs(z_col)>3]=np.nan
    return z_col




def multiplot(df,figsize=(16, 16)):
    """Plots correlation matrix as a seaborn heamap with upper triangle removed."""
    sns.set(style="white")

    # Compute the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=figsize)

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, annot=True, cmap=cmap, center=0,
                
    square=True, linewidths=.5, cbar_kws={"shrink": .5},ax=ax) #
    
    
    # Plots histogram and scatter (vs price) side by side
    
def plot_hist_scat(df,target='price',columns=[],style='seaborn-talk',stats=False,
                   hist_kwds={},scatter_kwds={},figsize=(8,3),fig_filepath=None):
    from scipy.stats import normaltest
        
    if len(columns)==0:
        columns = df.describe().columns
    with plt.style.context(style):

        results = [['column','K_square','p-val']]

        for column in columns:

            fig = plt.figure(figsize=figsize)#plt.figaspect(0.5))#(5,4))
            
            ax1 = fig.add_subplot(121)
            ax1.hist(df[column],density=True,label = column+' histogram',bins=20,**hist_kwds)
            ax1.set_title(column.capitalize())

            ax1.legend()
            
            ax2 = fig.add_subplot(122)
            ax2.scatter(x=df[column], y=df[target],label = column+' vs price',marker='.',**scatter_kwds)
            ax2.set_title(column.capitalize())
            ax2.legend()

            fig.tight_layout()
            if stats==True:
                stat, p = normtest(df[column])
    #             print(f'Normality test for {column}:K_square = {stat}, p-value = {p}')

                results.append([column,stat, p])
                
        # if fig_filepath is None:

        return pd.DataFrame(results)



#SEABORN
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt

# Plots histogram and scatter (vs price) side by side
def plot_hist_scat_sns(df,target='price',style='seaborn-talk',stats=False,
                   hist_kwds={},scatter_kwds={},figsize=(8,3),fig_filepath=None):
    """Plots 2 subplots: a seaborn distplot of each column, and a regplot of each column vs target.
    
    Args:
        df (df): Dataframe of data
        target (str, optional): Name of target column. Defaults to 'price'.
        style (str, optional): Matplotlib style to use. Defaults to 'seaborn-talk'.
        stats (bool, optional): Show normality test. Defaults to False.
        hist_kwds (dict, optional): Plotting **kwargs for seaborn hist. Defaults to {}.
        scatter_kwds (dict, optional): Plotting **kwargs for seaborn scatter. Defaults to {}.
        figsize (tuple, optional): Figure size Defaults to (8,3).
        fig_filepath ([type], optional): To save, give filename to saveas. Defaults to None.
    """
    
    with plt.style.context(style):

        
        ## ----------- DEFINE AESTHETIC CUSTOMIZATIONS ----------- ##
        # Axis Label fonts
        fontTitle = {'fontsize': 16,
                'fontweight': 'bold',
                    'fontfamily':'serif'}

        fontAxis = {'fontsize': 14,
                'fontweight': 'bold',
                    'fontfamily':'serif'}

        fontTicks = {'fontsize': 12,
                'fontweight':'bold',
                    'fontfamily':'serif'}

        # Formatting dollar sign labels
        fmtPrice = '${x:,.0f}'
        tickPrice = mtick.StrMethodFormatter(fmtPrice)
        

        ## ----------- PLOTTING ----------- ##
        
        ## Loop through dataframe to plot
        for column in df.describe():
        
            # Create figure with subplots for current column
            # Note: in order to use identical syntax for large # of subplots (ax[i,j]), 
            #  declare an extra row of subplots to be removed later
            fig, ax = plt.subplots(figsize=figsize, ncols=2, nrows=2)

            ## ----- SUBPLOT 1 -----##
            i,j = 0,0
            ax[i,j].set_title(column.capitalize(),fontdict=fontTitle)
            
            # Define graphing keyword dictionaries for distplot (Subplot 1)
            hist_kws = {"linewidth": 1, "alpha": 1, "color": 'blue','edgecolor':'w'}
            kde_kws = {"color": "white", "linewidth": 1, "label": "KDE"}
            
            # Plot distplot on ax[i,j] using hist_kws and kde_kws
            sns.distplot(df[column], norm_hist=True, kde=True,
                        hist_kws = hist_kws, kde_kws = kde_kws,
                        label=column+' histogram', ax=ax[i,j])
    

            # Set x axis label
            ax[i,j].set_xlabel(column.title(),fontdict=fontAxis)
        
            # Get x-ticks, rotate labels, and return
            xticklab1 = ax[i,j].get_xticklabels(which = 'both')
            ax[i,j].set_xticklabels(labels=xticklab1, fontdict=fontTicks, rotation=45)
            ax[i,j].xaxis.set_major_formatter(mtick.ScalarFormatter())

            
            # Set y-label 
            ax[i,j].set_ylabel('Density',fontdict=fontAxis)
            yticklab1=ax[i,j].get_yticklabels(which='both')
            ax[i,j].set_yticklabels(labels=yticklab1,fontdict=fontTicks)
            ax[i,j].yaxis.set_major_formatter(mtick.ScalarFormatter())
            
            
            # Set y-grid
            ax[i, j].set_axisbelow(True)
            ax[i, j].grid(axis='y',ls='--')

            
            ## ----- SUBPLOT 2-----  ##
            i,j = 0,1
            ax[i,j].set_title(column.capitalize(),fontdict=fontTitle)

            # Define the ketword dictionaries for  scatter plot and regression line (subplot 2)
            line_kws={"color":"white","alpha":0.5,"lw":4,"ls":":"}
            scatter_kws={'s': 2, 'alpha': 0.5,'marker':'.','color':'blue'}

            # Plot regplot on ax[i,j] using line_kws and scatter_kws
            sns.regplot(df[column], df[target], 
                        line_kws = line_kws,
                        scatter_kws = scatter_kws,
                        ax=ax[i,j])
            
            # Set x-axis label
            ax[i,j].set_xlabel(column.title(),fontdict=fontAxis)

            # Get x ticks, rotate labels, and return
            xticklab2=ax[i,j].get_xticklabels(which='both')
            ax[i,j].set_xticklabels(labels=xticklab2,fontdict=fontTicks, rotation=45)
            ax[i,j].xaxis.set_major_formatter(mtick.ScalarFormatter())

            # Set  y-axis label
            ax[i,j].set_ylabel(target,fontdict=fontAxis)
            
            # Get, set, and format y-axis Price labels
            yticklab = ax[i,j].get_yticklabels()
            ax[i,j].set_yticklabels(yticklab,fontdict=fontTicks)
            ax[i,j].get_yaxis().set_major_formatter(tickPrice) 

            # Set y-grid
            ax[i, j].set_axisbelow(True)
            ax[i, j].grid(axis='y',ls='--')       
            
            ## ---------- Final layout adjustments ----------- ##
            # Deleted unused subplots 
            fig.delaxes(ax[1,1])
            fig.delaxes(ax[1,0])

            # Optimizing spatial layout
            fig.tight_layout()
            figtitle=column+'_dist_regr_plots.png'
            
            if fig_filepath is not None:
                plt.savefig(fig_filepath+figtitle)
        return 

# Tukey's method using IQR to eliminate 
def detect_outliers(df,n,features):
    """Identifies outliers using IQR rule"""
    outlier_indices = []
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        # outlier step
        outlier_step = 1.5 * IQR
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        # select observations containing more than 2 outliers
        outlier_indices = Counter(outlier_indices)        
        multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
        return multiple_outliers 
        # Outliers_to_drop = detect_outliers(data,2,["col1","col2"])
        # df.loc[Outliers_to_drop] # Show the outliers rows
        # Drop outliers
        # data= data.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)



def plot_cat_box_sns(df,target='price',figsize=(12,6),
                     list_cat_vars=['zipcode', 'bins_yrbuilt','bins_sqftbasement',
                                    'bins_sqftabove','condition','grade']):
    """Plots seaborn stripplot + boxplot"""
    import seaborn as sns
#     if list_cat_vars is None:
#         list_cat_vars = ['zipcode', 'bins_yrbuilt', 'bins_sqftbasement',
#                          'bins_sqftabove','condition',
#                          'grade']


    for column in list_cat_vars:
        ## Create figure and axes
        fig, (ax1,ax2) = plt.subplots(ncols=2,figsize=figsize)

        # ax1.set_title('Price vs ',column,' scatter plot')
        title1=column+' scatter'
        ax1.set_title(title1.title())
        ax1.set_xlabel(column)
        ax1.set_ylabel(target)
        
        ## Create the seaborn plots
        sns.stripplot(x=df[column],y=df[target],marker='.',ax=ax1) 
        sns.boxplot(x=df[column],y=df[target],ax=ax2) 

        ## Create keywords for .set_xticklabels()
        tick_kwds = dict(horizontalalignment='right', 
                         fontweight='light', 
                         fontsize='x-large',   
                         rotation=45)
        ax1.set_xticklabels(ax1.get_xticklabels(),**tick_kwds)
        ax2.set_xticklabels(ax1.get_xticklabels(),**tick_kwds)

        title2=column+' boxplot'
        ax2.set_title(title2.title())
        ax2.set_xlabel(column)
        ax2.set_ylabel(target)
        fig.tight_layout()
