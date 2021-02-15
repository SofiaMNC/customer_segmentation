'''
    This module defines a set of graphical
    functions for project #5.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

import helper_functions as hf

#------------------------------------------

def plot_percentage_missing_values_for(data, long, larg):
    '''
        Plots the proportions of filled / missing values for each unique value
        in column as a horizontal bar chart.

        Parameters
        ----------------
        data : pandas dataframe with:
                - a column column
                - a column "Percent Filled"
                - a column "Percent Missing"
                - a column "Total"

       long : int
            The length of the figure for the plot

        larg : int
               The width of the figure for the plot

        Returns
        ---------------
        -
    '''

    data_to_plot = hf.get_missing_values_percent_per(data)\
                     .sort_values("Percent Filled").reset_index()

    # Constants for the plot
    TITLE_SIZE = 60
    TITLE_PAD = 100
    TICK_SIZE = 50
    TICK_PAD = 20
    LABEL_SIZE = 50
    LABEL_PAD = 50
    LEGEND_SIZE = 50

    sns.set(style="whitegrid")

    #sns.set_palette(sns.dark_palette("purple", reverse=True))

    # Initialize the matplotlib figure
    _, axis = plt.subplots(figsize=(long, larg))

    plt.title("PROPORTIONS DE VALEURS RENSEIGNÉES / NON-RENSEIGNÉES PAR COLONNE",
              fontweight="bold",
              fontsize=TITLE_SIZE, pad=TITLE_PAD)

    # Plot the Total values
    handle_plot_1 = sns.barplot(x="Total", y="index", data=data_to_plot,
                                label="non renseignées", color="thistle", alpha=0.3)

    handle_plot_1.set_xticklabels(handle_plot_1.get_xticks(), size=TICK_SIZE)
    _, ylabels = plt.yticks()
    handle_plot_1.set_yticklabels(ylabels, size=TICK_SIZE)


    # Plot the Percent Filled values
    handle_plot_2 = sns.barplot(x="Percent Filled", y="index", data=data_to_plot,
                                label="renseignées", color="darkviolet")

    handle_plot_2.set_xticklabels(handle_plot_2.get_xticks(), size=TICK_SIZE)
    handle_plot_2.set_yticklabels(ylabels, size=TICK_SIZE)


    # Add a legend and informative axis label
    axis.legend(bbox_to_anchor=(1.04, 0), loc="lower left",
                borderaxespad=0, ncol=1, frameon=True, fontsize=LEGEND_SIZE)

    axis.set(ylabel="Colonnes", xlabel="Pourcentage de valeurs (%)")

    x_label = axis.get_xlabel()
    axis.set_xlabel(x_label, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    y_label = axis.get_ylabel()
    axis.set_ylabel(y_label, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    axis.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,
                                                               pos: '{:2d}'.format(int(x)) + '%'))
    axis.tick_params(axis='both', which='major', pad=TICK_PAD)

    sns.despine(left=True, bottom=True)

    # Display the figure
    plt.show()

#------------------------------------------

def plot_2_pie_charts(data1, data2, long, larg, title):
    '''
        Plots 2 pie charts of the proportions of each modality for data1
        and data2.
        The figure has dimension (long, larg).

        Parameters
        ----------------
        data1 : pandas dataframe
                Working data for plot 1. Contains proportions in %
                for each modality as index

        data2 : pandas dataframe
                Working data for plot 2. Contains proportions in %
                for each modality as index

        long  : int
                The length of the figure for the plot

        larg  : int
                The width of the figure for the plot

        Returns
        ---------------
        -
    '''

    # Constants for the plot
    TITLE_SIZE = 25

    plt.figure(figsize=(long, larg), dpi=800)

    sns.set_palette(sns.color_palette("husl", 8))

    ax1 = plt.subplot(121, aspect='equal')

    data1.plot(kind='pie', autopct=lambda x: '{:2d}'.format(int(x)) + '%',
               fontsize=20,
               subplots=True,
               ax=ax1)

    plt.title(title + " 2017", fontsize=TITLE_SIZE)

    ax2 = plt.subplot(122, aspect='equal')

    data2.plot(kind='pie', autopct=lambda x: '{:2d}'.format(int(x)) + '%',
               fontsize=20,
               subplots=True,
               ax=ax2)

    plt.title(title + " 2018", fontsize=TITLE_SIZE)

#------------------------------------------

def plot_qualitative_dist(data, long, larg):
    '''
        Displays a bar chart showing the frequency of the modalities
        for each column of data.

        Parameters
        ----------------
        data : dataframe
               Working data containing exclusively qualitative data

        long : int
               The length of the figure for the plot

        larg : int
               The width of the figure for the plot

        Returns
        ---------------
        -
    '''

    # Contants for the plot
    TITLE_SIZE = 130
    TITLE_PAD = 1.05
    TICK_SIZE = 50
    LABEL_SIZE = 80
    LABEL_PAD = 30

    nb_rows = 2
    nb_cols = 2

    fig, axes = plt.subplots(nb_rows, nb_cols, figsize=(long, larg))

    fig.suptitle("DISTRIBUTION DES VALEURS QUALITATIVES",
                 fontweight="bold", fontsize=TITLE_SIZE, y=TITLE_PAD)

    row = column = 0

    for ind_qual in data.columns.tolist():

        data_to_plot = data.sort_values(by=ind_qual).copy()

        axis = axes[row, column]

        plot_handle = sns.countplot(y=ind_qual,
                                    data=data_to_plot,
                                    color="darkviolet",
                                    ax=axis,
                                    order=data_to_plot[ind_qual].value_counts().index)

        plt.tight_layout()

        plt.subplots_adjust(left=None,
                            bottom=None,
                            right=None,
                            top=None,
                            wspace=1.4, hspace=0.2)

        plot_handle.set_xticklabels(plot_handle.get_xticks(), size=TICK_SIZE)

        if ind_qual == "nova_group":
            ylabels = [item.get_text()[0] for item in axis.get_yticklabels()]
        else:
            ylabels = [item.get_text().upper() for item in axis.get_yticklabels()]

        plot_handle.set_yticklabels(ylabels, size=TICK_SIZE, weight="bold")

        x_label = axis.get_xlabel()
        axis.set_xlabel(x_label, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

        y_label = axis.get_ylabel()
        axis.set_ylabel(y_label.upper(), fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

        axis.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:d}'.format(int(x))))

        axis.xaxis.grid(True)

        if column < nb_cols-1:
            column += 1
        else:
            row += 1
            column = 0

#------------------------------------------

def plot_boxplots(data, long, larg, nb_rows, nb_cols):
    '''
        Displays a boxplot for each column of data.

        Parameters
        ----------------
        data    : dataframe
                  Working data containing exclusively quantitative data

        long    : int
                  The length of the figure for the plot

        larg    : int
                  The width of the figure for the plot

        nb_rows : int
                  The number of rows in the subplot

        nb_cols : int
                  The number of cols in the subplot

        Returns
        ---------------
        -
    '''

    # Constants for the plot
    TITLE_SIZE = 35
    TITLE_PAD = 1.05
    TICK_SIZE = 15
    TICK_PAD = 20
    LABEL_SIZE = 25
    LABEL_PAD = 10

    fig, axes = plt.subplots(nb_rows, nb_cols, figsize=(long, larg))

    fig.suptitle("VALEURS QUANTITATIVES - DISTRIBUTION",
                 fontweight="bold",
                 fontsize=TITLE_SIZE,
                 y=TITLE_PAD)

    row = column = 0

    for ind_quant in data.columns.tolist():
        axis = axes[row, column]

        sns.despine(left=True)

        plot_handle = sns.boxplot(x=data[ind_quant], ax=axis, color="darkviolet")

        plt.setp(axes, yticks=[])

        plt.tight_layout()

        plot_handle.set_xticklabels(plot_handle.get_xticks(), size=TICK_SIZE)

        x_label = axis.get_xlabel()
        axis.set_xlabel(x_label, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

        if ind_quant == "salt_100g":
            axis.xaxis.set_major_formatter(ticker\
                                           .FuncFormatter(lambda x,
                                                                 pos: '{:.2f}'\
                                                                 .format(float(x))))
        else:
            axis.xaxis.set_major_formatter(ticker\
                                           .FuncFormatter(lambda x,
                                                                 pos: '{:d}'\
                                                                 .format(int(x))))

        y_label = axis.get_ylabel()
        axis.set_ylabel(y_label, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

        axis.tick_params(axis='both', which='major', pad=TICK_PAD)

        axis.xaxis.grid(True)
        axis.set(ylabel="")

        if column < nb_cols-1:
            column += 1
        else:
            row += 1
            column = 0

#------------------------------------------

def plot_lorenz(lorenz_df, long, larg):
    '''
        Plots a Lorenz curve with the given title

        Parameters
        ----------------
        - lorenz_df : dataframe
                      Working data containing the Lorenz values
                      one column = lorenz value for a variable

        - long      : int
                      The length of the figure for the plot

        - larg      : int
                      The width of the figure for the plot

        Returns
        ---------------
        _
    '''

    # Constants for the plot
    TITLE_SIZE = 60
    TITLE_PAD = 100
    TICK_SIZE = 50
    TICK_PAD = 20
    LEGEND_SIZE = 50

    sns.set(style="whitegrid")

    _, axis = plt.subplots(figsize=(long, larg))

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    plt.title("VARIABLES QUANTITATIVES - COURBES DE LORENZ",
              fontweight="bold",
              fontsize=TITLE_SIZE, pad=TITLE_PAD)

    # Plot the Total values
    sns.set_color_codes("pastel")

    plot_handle = sns.lineplot(data=lorenz_df,
                               palette=sns.color_palette("hls", len(lorenz_df.columns)),
                               linewidth=5, dashes=False)

    plot_handle.set_xticklabels(plot_handle.get_xticks(), size=TICK_SIZE)
    plot_handle.set_yticklabels(plot_handle.get_yticks(), size=TICK_SIZE)

    axis.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(float(x))))
    axis.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(float(x))))

    axis.tick_params(axis='both', which='major', pad=TICK_PAD)

    axis.set_xlabel("")

    # Add a legend and informative axis label
    leg = axis.legend(bbox_to_anchor=(1.04, 0), loc="lower left",
                      borderaxespad=0, ncol=1, frameon=True, fontsize=LEGEND_SIZE)

    for legobj in leg.legendHandles:
        legobj.set_linewidth(5.0)

    # Display the figure
    plt.show()

#------------------------------------------

def plot_correlation_heatmap(data, long, larg, title):
    '''
        Plots a heatmap of the correlation coefficients
        between the quantitative columns in data

        Parameters
        ----------------
        - data : dataframe
                 Working data

        - corr : string
                 the correlation method ("pearson" or "spearman")

        - long : int
                 The length of the figure for the plot

        - larg : int
                 The width of the figure for the plot

        Returns
        ---------------
        _
    '''

    # Constants for the plot
    TITLE_SIZE = 40
    TITLE_PAD = 1
    TICK_SIZE = 20
    LABEL_SIZE = 45
    LABEL_PAD = 30

    fig, axis = plt.subplots(figsize=(long, larg))

    fig.suptitle(title, fontweight="bold",
                 fontsize=TITLE_SIZE, y=TITLE_PAD)

    plot_handle = sns.heatmap(data, mask=np.zeros_like(data, dtype=np.bool),
                              cmap=sns.diverging_palette(220, 10, as_cmap=True),
                              square=True, ax=axis,
                              annot=data, annot_kws={"fontsize":20}, fmt=".2f")

    xlabels = [item.get_text() for item in axis.get_xticklabels()]
    plot_handle.set_xticklabels(xlabels, size=TICK_SIZE, weight="bold")
    plot_handle.set_xlabel(data.columns.name, fontsize=LABEL_SIZE,
                           labelpad=LABEL_PAD, fontweight="bold")

    ylabels = [item.get_text() for item in axis.get_yticklabels()]
    plot_handle.set_yticklabels(ylabels, size=TICK_SIZE, weight="bold")
    plot_handle.set_ylabel(data.index.name, fontsize=LABEL_SIZE,
                           labelpad=LABEL_PAD, fontweight="bold")

    plt.show()

#------------------------------------------

def plot_purchase_category_proportions_for_state(data, long, larg):
    '''
        Plots the repartition of the purchase categories by state

        Parameters
        ----------------
        data : pandas dataframe
               The data to plot containing the cumsum of
               the number of purchases per category per
               state

        long : int
               The length of the figure for the plot

        larg : int
               The width of the figure for the plot

        Returns
        ---------------
        -
    '''

    # Constants for the plot
    TITLE_SIZE = 60
    TITLE_PAD = 100
    TICK_SIZE = 50
    TICK_PAD = 30
    LABEL_SIZE = 50
    LABEL_PAD = 50
    LEGEND_SIZE = 30

    # Reset index to access the Seuil as a column
    data_to_plot = data.reset_index()

    sns.set(style="whitegrid")
    palette = sns.husl_palette(len(data.columns))

    # Initialize the matplotlib figure
    _, axis = plt.subplots(figsize=(long, larg))

    plt.title("RÉPARTITION DES ACHATS PAR CATÉGORIE SUIVANT LA RÉGION",
              fontweight="bold", fontsize=TITLE_SIZE, pad=TITLE_PAD)

    # Get the list of topics from the columns of data
    column_list = list(data.columns)

    # Create a barplot with a distinct color for each topic
    for idx, column in enumerate(reversed(column_list)):
        color = palette[idx]
        plot_handle = sns.barplot(x=column, y="index", data=data_to_plot,
                                  label=str(column), orient="h", color=color)

        plot_handle.set_xticklabels(plot_handle.get_xticks(), size=TICK_SIZE)
        _, ylabels = plt.yticks()
        plot_handle.set_yticklabels(ylabels, size=TICK_SIZE)

    # Add a legend and informative axis label

    axis.legend(bbox_to_anchor=(0, -0.4, 1, 0.2), loc="lower left", mode="expand",
                borderaxespad=0, ncol=4, frameon=True, fontsize=LEGEND_SIZE)

    axis.set(ylabel="customer_state", xlabel="% d'achats par catégorie")

    x_label = axis.get_xlabel()
    axis.set_xlabel(x_label, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    y_label = axis.get_ylabel()
    axis.set_ylabel(y_label, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    axis.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:2d}'.format(int(x))))

    axis.tick_params(axis='both', which='major', pad=TICK_PAD)

    sns.despine(left=True, bottom=True)

    # Display the figure
    plt.show()

#------------------------------------------

def plot_lineplot_rfm(data, col_x, col_y, hue_col, title, long, larg):
    '''
        Plots a lineplot of col_y as a function of col_x with the
        data in data

        ----------------
        - data  : pandas dataframe
                  Working data containing the col_x and col_y columns
        - col_x : string
                  The name of a column present in data
        - col_y : string
                  The name of a column present in data
        - title : string
                  The title of the figure
        - long  : int
                  The length of the figure
        - larg  : int
                  The widht of the figure

        Returns
        ---------------
        _
    '''

    # Constants for the plot
    TITLE_SIZE = 30
    TITLE_PAD = 1
    TICK_SIZE = 20
    LABEL_SIZE = 20
    LABEL_PAD = 30
    LEGEND_SIZE = 15

    sns.set(style="whitegrid")

    plt.rcParams["font.weight"] = "bold"

    plt.rc('font', size=LABEL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=LABEL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=LABEL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=TITLE_SIZE)

    fig, axis = plt.subplots(figsize=(long, larg))

    fig.suptitle(title,
                 fontweight="bold",
                 fontsize=TITLE_SIZE, y=TITLE_PAD)

    sns.lineplot(x=col_x, y=col_y,
                 hue=hue_col, data=data)

    x_label = axis.get_xlabel()
    axis.set_xlabel(x_label.upper(), fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    y_label = axis.get_ylabel()
    axis.set_ylabel(y_label.upper(), fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    axis.legend(bbox_to_anchor=(0.2, -0.4, 0.6, 0.2), loc="lower left", mode="expand",
                borderaxespad=0, ncol=2, frameon=True, fontsize=LEGEND_SIZE)

    plt.show()

#------------------------------------------

def plot_scatterplot_with_hue(data, col_x, col_y, col_hue, title, long, larg):
    '''
        Plots a scatterplot of col_y as a function of col_x with the
        data in data

        ----------------
        - data    : pandas dataframe
                    Working data containing the col_x and col_y columns
        - col_x   : string
                    The name of a column present in data
        - col_y   : string
                    The name of a column present in data
        - col_hue : string
                    The name of the column in data to use for hue
        - title   : string
                    The title of the figure
        - long    : int
                    The length of the figure
        - larg    : int
                    The widht of the figure

        Returns
        ---------------
        _
    '''

    # Constants for the plot
    TITLE_SIZE = 30
    TITLE_PAD = 1
    TICK_SIZE = 20
    LABEL_SIZE = 20
    LABEL_PAD = 30
    LEGEND_SIZE = 15

    sns.set(style="whitegrid")

    plt.rcParams["font.weight"] = "bold"

    plt.rc('font', size=LABEL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=LABEL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=LABEL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=TITLE_SIZE)

    fig, axis = plt.subplots(figsize=(long, larg))

    fig.suptitle(title,
                 fontweight="bold",
                 fontsize=TITLE_SIZE, y=TITLE_PAD)

    sns.scatterplot(x=col_x, y=col_y,
                    hue=col_hue, sizes=(1, 15),
                    linewidth=0, palette=sns.color_palette("husl", data[col_hue].nunique()),
                    data=data)

    x_label = axis.get_xlabel()
    axis.set_xlabel(x_label.upper(), fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    y_label = axis.get_ylabel()
    axis.set_ylabel(y_label.upper(), fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    axis.legend(bbox_to_anchor=(0.2, -0.4, 0.6, 0.2), loc="lower left", mode="expand",
                borderaxespad=0, ncol=2, frameon=True, fontsize=LEGEND_SIZE)

    plt.show()

#------------------------------------------

def display_scree_plot(pca):
    '''
        Plots the scree plot for the given pca
        components.

        ----------------
        - pca : A PCA object
                The result of a PCA decomposition

        Returns
        ---------------
        _
    '''

    # Constants for the plot
    TITLE_SIZE = 30
    LABEL_SIZE = 20
    LABEL_PAD = 30

    plt.subplots(figsize=(10, 10))

    scree = pca.explained_variance_ratio_ * 100

    plt.bar(np.arange(len(scree))+1, scree)

    plt.plot(np.arange(len(scree))+1,
             scree.cumsum(), c="red", marker='o')

    plt.xlabel("Rang de l'axe d'inertie",
               fontsize=LABEL_SIZE,
               labelpad=LABEL_PAD)

    plt.ylabel("% d'inertie",
               fontsize=LABEL_SIZE,
               labelpad=LABEL_PAD)

    plt.title("Eboulis des valeurs propres",
              fontsize=TITLE_SIZE)

    plt.show(block=False)

#------------------------------------------

def plot_barplot(data, x_feature, y_feature, title, long, larg):
    '''
        Plots a barplot of y_feature = f(x_feature) in data

        Parameters
        ----------------
        data      : pandas dataframe with:
                    - a qualitative column named x_feature
                    - a quantitative column named y_feature

        x_feature : string
                    The name of the qualitative column
                    contained in data

        y_feature : string
                    The name of a quantitative column
                    contained in data

        title     : string
                    The title to give the plot

        long      : int
                    The length of the figure for the plot

         larg     : int
                    The width of the figure for the plot

        Returns
        ---------------
        -
    '''

    # Constants for the plot
    TITLE_SIZE = 50
    TITLE_PAD = 80
    TICK_SIZE = 30
    TICK_PAD = 20
    LABEL_SIZE = 50
    LABEL_PAD = 50

    sns.set(style="whitegrid")

    #sns.set_palette(sns.dark_palette("purple", reverse=True))

    # Initialize the matplotlib figure
    _, axis = plt.subplots(figsize=(long, larg))

    plt.title(title,
              fontweight="bold",
              fontsize=TITLE_SIZE, pad=TITLE_PAD)

    # Plot the Total values
    handle_plot_1 = sns.barplot(x=x_feature, y=y_feature, data=data, color="purple")

    _, xlabels = plt.xticks()
    handle_plot_1.set_xticklabels(xlabels, size=TICK_SIZE)

    handle_plot_1.set_yticklabels(handle_plot_1.get_yticks(),
                                  size=TICK_SIZE)

    x_label = axis.get_xlabel()
    axis.set_xlabel(x_label, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    y_label = axis.get_ylabel()
    axis.set_ylabel(y_label, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    axis.tick_params(axis='both', which='major', pad=TICK_PAD)
    axis.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,
                                                               pos: '{:2d}'.format(int(x))))

    # Display the figure
    plt.show()

#------------------------------------------

def plot_2_distplots(col_x1, label_col_x1, col_x2, label_col_x2, long, larg, title):
    '''
        Plots 2 distplots horizontally in a single figure

        Parameters
        ----------------
        col_x1          : pandas series
                          The data to use for x in the 1st distplot

        label_col_x1    : string
                          The label for x in the 1st distplot

        col_x2          : pandas series
                          The data to use for x in the 2nd distplot

        label_col_x2    : string
                          The label for x in the 2nd distplot

        long            : int
                          The length of the figure for the plot

        larg            : int
                          The width of the figure for the plot

        title           : string
                          The title for the plot

        Returns
        ---------------
        -
    '''

    TITLE_SIZE = 30
    TITLE_PAD = 1.05
    TICK_SIZE = 15
    LABEL_SIZE = 20
    LABEL_PAD = 30

    sns.set_palette(sns.dark_palette("purple", reverse="True"))

    figure, (ax1, ax2) = plt.subplots(ncols=2,
                                      sharey=False,
                                      figsize=(long, larg))

    figure.suptitle(title, fontweight="bold",
                    fontsize=TITLE_SIZE, y=TITLE_PAD)

    sns.despine(left=True)

    handle_plot_1 = sns.distplot(col_x1, ax=ax1)

    handle_plot_1.set_xticklabels(handle_plot_1.get_xticks(), size=TICK_SIZE)
    handle_plot_1.set_xlabel(label_col_x1, fontsize=LABEL_SIZE,
                             labelpad=LABEL_PAD, fontweight="bold")

    handle_plot_2 = sns.distplot(col_x2, ax=ax2)

    handle_plot_2.set_xticklabels(handle_plot_2.get_xticks(), size=TICK_SIZE)
    handle_plot_2.set_xlabel(label_col_x2, fontsize=LABEL_SIZE,
                             labelpad=LABEL_PAD, fontweight="bold")

    plt.setp((ax1, ax2), yticks=[])
    plt.setp((ax1, ax2), xticks=[])


    plt.tight_layout()

#------------------------------------------

def plot_corr_p_value(data, interest_cols, FEATURE, THRESHOLD, TITLE_1, TITLE_2):
    '''
        Plots a heatmap of the correlation values > THRESHOLD
        and the associated p-values for a given FEATURE in
        data.

        Parameters
        ----------------
        data          : pandas dataframe
                        The data to use to calculate the correlations.

        interest_cols : list
                        The names of the features to calculate the
                        correlations with.

        FEATURE        : string
                         The name of the feature to calculate all corre-
                         -lations against.

        THRESHOLD      : float
                         The threshold to include or not a given corre-
                         -lation in the plot.

        TITLE_1        : string
                         The title for the correlation heatmap

        TITLE_2         : string
                          The title for the p-value heatmap

        Returns
        ---------------
        -
    '''

    interest_cols.remove(FEATURE)

    corrs_heatmap_df, \
    ps_heatmap_df = hf.get_rho_p_value_heatmaps(data,
                                                interest_cols,
                                                FEATURE,
                                                THRESHOLD)

    plot_correlation_heatmap(corrs_heatmap_df, 15, 15, TITLE_1)
    plot_correlation_heatmap(ps_heatmap_df, 15, 15, TITLE_2)

#------------------------------------------

def plot_2_discrete_distplots(data1, data2, feature_name):
    '''
        Plots 2 distplots horizontally in a single figure
        for discrete features

        Parameters
        ----------------
        data1       : pandas series
                      The data to use for the 1st distplot

        data2       : pandas series
                      The data to use for the 2nd distplot

       feature_name : string
                      The name of the feature of interest

        Returns
        ---------------
        -
    '''

    TITLE_SIZE = 50
    TITLE_PAD = 1.05
    TICK_SIZE = 30
    LABEL_SIZE = 40
    LABEL_PAD = 30

    sns.set_palette(sns.dark_palette("purple", reverse=True))

    # R scores

    figure, (ax1, ax2) = plt.subplots(ncols=2, sharey=False, figsize=(30, 10))

    figure.suptitle(feature_name, fontweight="bold",
                    fontsize=TITLE_SIZE, y=TITLE_PAD)

    sns.despine(left=True)

    handle_plot_1 = sns.distplot(data1[feature_name], ax=ax1)

    handle_plot_1.set_xticklabels(handle_plot_1.get_xticks(), size=TICK_SIZE)
    ax1.xaxis.set_major_formatter(ticker\
                                  .FuncFormatter(lambda x,
                                                        pos: '{:2d}'.format(int(x))))
    handle_plot_1.set_xlabel(feature_name+" values - 2017",
                             fontsize=LABEL_SIZE,
                             labelpad=LABEL_PAD,
                             fontweight="bold")

    handle_plot_2 = sns.distplot(data2[feature_name], ax=ax2)

    handle_plot_2.set_xticklabels(handle_plot_2.get_xticks(), size=TICK_SIZE)
    ax2.xaxis.set_major_formatter(ticker\
                                 .FuncFormatter(lambda x,
                                                       pos: '{:2d}'.format(int(x))))
    handle_plot_2.set_xlabel(feature_name+" values - 2018",
                             fontsize=LABEL_SIZE,
                             labelpad=LABEL_PAD, fontweight="bold")

    plt.setp((ax1, ax2), yticks=[])

    plt.tight_layout()

#------------------------------------------

def plot_correlation_circle(pcs, data, long, larg):
    '''
        Plots 2 distplots horizontally in a single figure

        Parameters
        ----------------
        pcs     : PCA components
                  Components from a PCA

        data    : pandas dataframe
                  The original data used for the PCA
                  before any treatment

        long            : int
                          The length of the figure for the plot

        larg            : int
                          The width of the figure for the plot

        Returns
        ---------------
        -
    '''

    # Constants for the plot
    TITLE_SIZE = 30
    TITLE_PAD = 70

    plt.subplots(figsize=(long, larg))

    for i, (x_value, y_value) in enumerate(zip(pcs[0, :], pcs[1, :])):
        if(x_value > 0.2 or y_value > 0.2):
            plt.plot([0, x_value], [0, y_value], color='k')
            plt.text(x_value, y_value, data.columns[i], fontsize='14')

    plt.plot([-0.5, 0.5], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-0.5, 0.5], color='grey', ls='--')

    plt.xlim([-0.5, 0.5])
    plt.ylim([-0.5, 0.5])

    plt.title("Corrélations - Variables latentes",
              fontsize=TITLE_SIZE,
              pad=TITLE_PAD)

    plt.subplots(figsize=(long, larg))

    for i, (x_value, y_value) in enumerate(zip(pcs[2, :], pcs[3, :])):
        if(x_value > 0.2 or y_value > 0.2):
            plt.plot([0, x_value], [0, y_value], color='k')
            plt.text(x_value, y_value, data.columns[i], fontsize='14')

    plt.plot([-0.5, 0.5], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-0.5, 0.5], color='grey', ls='--')

    plt.xlim([-0.5, 0.5])
    plt.ylim([-0.5, 0.5])

    plt.title("Corrélations - Variables latentes",
              fontsize=TITLE_SIZE,
              pad=TITLE_PAD)

    plt.subplots(figsize=(long, larg))

    for i, (x_value, y_value) in enumerate(zip(pcs[4, :], pcs[5, :])):
        if(x_value > 0.2 or y_value > 0.2):
            plt.plot([0, x_value], [0, y_value], color='k')
            plt.text(x_value, y_value, data.columns[i], fontsize='14')

    plt.plot([-0.5, 0.5], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-0.5, 0.5], color='grey', ls='--')

    plt.xlim([-0.5, 0.5])
    plt.ylim([-0.5, 0.5])
    plt.title("Corrélations - Variables latentes",
              fontsize=TITLE_SIZE,
              pad=TITLE_PAD)
