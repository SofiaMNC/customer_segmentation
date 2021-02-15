'''
    This module defines a set of calculations
    functions for project 5.
'''

import time
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score


#------------------------------------------

def get_missing_values_percent_per(data):
    '''
        Calculates the mean percentage of missing values
        in a given pandas dataframe per unique value
        of a given column

        Parameters
        ----------------
        data                : pandas dataframe
                              The dataframe to be analyzed

        Returns
        ---------------
        missing_percent_df  : A pandas dataframe containing:
                                - a column "column"
                                - a column "Percent Missing" containing the percentage of
                                  missing value for each value of column
    '''

    missing_percent_df = pd.DataFrame({'Percent Missing':data.isnull().sum()/len(data)*100})
    missing_percent_df['Percent Filled'] = 100 - missing_percent_df['Percent Missing']
    missing_percent_df['Total'] = 100

    return missing_percent_df


#------------------------------------------

def describe_dataset(source_files):
    '''
        Outputs a presentation pandas dataframe for the dataset.

        Parameters
        ----------------
        sourceFiles     : dict with :
                            - keys : the names of the files
                            - values : a list containing two values :
                                - the dataframe for the data
                                - a brief description of the file

        Returns
        ---------------
        presentation_df : pandas dataframe :
                            - a column "Nom du fichier" : the name of the file
                            - a column "Nb de lignes"   : the number of rows per file
                            - a column "Nb de colonnes" : the number of columns per file
                            - a column "Description"    : a brief description of the file
    '''

    print("Les données se décomposent en {} fichier(s): \n".format(len(source_files)))

    filenames = []
    files_nb_lines = []
    files_nb_columns = []
    files_descriptions = []

    for filename, file_data in source_files.items():
        filenames.append(filename)
        files_nb_lines.append(len(file_data[0]))
        files_nb_columns.append(len(file_data[0].columns))
        files_descriptions.append(file_data[1])

    # Create a dataframe for presentation purposes
    presentation_df = pd.DataFrame({'Nom du fichier':filenames,
                                    'Nb de lignes':files_nb_lines,
                                    'Nb de colonnes':files_nb_columns,
                                    'Description': files_descriptions})

    presentation_df.index += 1

    return presentation_df

#------------------------------------------

def calculate_eta_squared(data, x_qualit, y_quantit):
    '''
        Calculate the proportion of variance in the given quantitative variable for
        the given qualitative variable

        ----------------
        - data      : dataframe
                      Working data
        - x_quantit : string
                      The name of the qualitative variable
        - y_quantit : string
                      The name of the quantitative variable

        Returns
        ---------------
        Eta_squared : float
                      The variation coefficient
    '''

    sous_echantillon = data.copy().dropna(how="any")

    data_qualit = sous_echantillon[x_qualit]
    data_quantit = sous_echantillon[y_quantit]

    moyenne_quantit = data_quantit.mean()
    classes = []

    for classe in data_qualit.unique():
        data_quantit_classe = data_quantit[data_qualit == classe]
        classes.append({'ni': len(data_quantit_classe),
                        'moyenne_classe': data_quantit_classe.mean()})

    sum_squares_terms = sum([(yj-moyenne_quantit)**2 for yj in data_quantit])
    sum_squares_errors = sum([c['ni']*(c['moyenne_classe']-moyenne_quantit)**2 for c in classes])

    return sum_squares_errors/sum_squares_terms

#------------------------------------------

def get_eta_squared(data, qualitative_cols, corr_col):
    '''
        Return all the etas squared for corr_col in relation
        to the qualitative cols given

        Parameters
        ----------------
        data            : pandas dataframe
                          Contains all the features named in qualitative_cols
                          and corr_col

       qualitative_cols : [string]
                          The name of the qualitative cols of interest
                          in data

        corr_col        : string
                          The name of the column of interest

        Returns
        ---------------
        -               : pandas dataframe
                          All the calculated etas squared
    '''

    eta_df = pd.DataFrame()

    for qual_col in qualitative_cols:
        eta_df[qual_col] = [calculate_eta_squared(data, qual_col, corr_col)]

    return eta_df.T.rename(columns={0:"Coeff de corrélation"}).sort_values("Coeff de corrélation",
                                                                           ascending=False)

#------------------------------------------

def calculate_lorenz_gini(data):
    '''
        Calculate the lorenz curve and Gini coeff
        for a given variable

        ----------------
        - data       : data series
                       Working  data

        Returns
        ---------------
        A tuple containing :
        - lorenz_df  : list
                       The values for the Lorenz curve
        - gini_coeff : float
                       The associated Gini coeff

        Source : www.openclassrooms.com
    '''

    dep = data.dropna().values
    number_of_values = len(dep)
    lorenz = np.cumsum(np.sort(dep)) / dep.sum()
    lorenz = np.append([0], lorenz) # La courbe de Lorenz commence à 0

    #---------------------------------------------------
    # Gini :
    # Surface sous la courbe de Lorenz. Le 1er segment
    # (lorenz[0]) est à moitié en dessous de 0, on le
    # coupe donc en 2, on fait de même pour le dernier
    # segment lorenz[-1] qui est à 1/2 au dessus de 1.
    #---------------------------------------------------

    area_under_curve = (lorenz.sum() -lorenz[-1]/2 -lorenz[0]/2)/number_of_values
    # surface entre la première bissectrice et le courbe de Lorenz
    area_of_interest = 0.5 - area_under_curve
    gini_coeff = [2 * area_of_interest]

    return (lorenz, gini_coeff)

#------------------------------------------

def get_lorenzs_ginis(data):
    '''
        Calculate the lorenz curve and Gini coeffs
        for all columns in the given dataframe

        ----------------
        - data       : dataframe
                       Working data

        Returns
        ---------------
        A tuple containing :
        - lorenz_df  : dataframne
                       The values for the Lorenz curve for each
                       column of the given dataframe
        - gini_coeff : dataframe
                       The associated Gini coeff for each column of
                       the given dataframe
    '''

    ginis_df = pd.DataFrame()
    lorenzs_df = pd.DataFrame()

    for ind_quant in data.columns.unique().tolist():
        lorenz, gini = calculate_lorenz_gini(data[ind_quant])
        ginis_df[ind_quant] = gini
        lorenzs_df[ind_quant] = lorenz

    len_lorenzs = len(lorenzs_df)
    xaxis = np.linspace(0-1/len_lorenzs, 1+1/len_lorenzs, len_lorenzs+1)
    lorenzs_df["index"] = xaxis[:-1]
    lorenzs_df.set_index("index", inplace=True)

    ginis_df = ginis_df.T.rename(columns={0:'Indice Gini'})

    return (lorenzs_df, ginis_df)

#------------------------------------------

def assign_random_clusters(data, n_clusters):
    '''
        Assigns a random cluster out of n_clusters to each observation
        in data.

        Parameters
        ----------------
        data             : pandas dataframe
                           Contains the observations

        - n_clusters    : int
                          Number of clusters to choose from

        Returns
        ---------------
        _   : numpy array
              Contains the assigned clusters for the observations in data
    '''

    return np.random.randint(n_clusters, size=len(data))

#------------------------------------------

def fit_plot(algorithms, data, long, larg, title):
    '''
        For each given algorithm :
        - fit them to the data on 3 iterations
        - Calculate the mean silhouette and adjusted rand scores
        - Gets the calculation time

        The function then plots the identified clusters for each algorithm.

        Parameters
        ----------------
        algorithms : dictionary with
                        - names and type of input as keys
                        - instantiated algorithms as values

        - data     : pandas dataframe
                     Contains the data to fit the algos on

        - long     : int
                     length of the plot figure

        - larg     : int
                     width of the plot figure

        - title    : string
                     title of the plot figure

        Returns
        ---------------
        scores_time : pandas dataframe
                      Contains the mean silhouette coefficient,
                      the adjusted Rnad score, the number of clusters,
                      the calculation time for each algorithm in algorithms
    '''

    scores_time = pd.DataFrame(columns=["Algorithme", "iter",
                                        "silhouette", "Rand",
                                        "Nb Clusters", "Time"])

    # Constants for the plot
    TITLE_SIZE = 45
    TITLE_PAD = 1.05
    SUBTITLE_SIZE = 25
    TICK_SIZE = 25
    LABEL_SIZE = 30
    LABEL_PAD = 30
    LEGEND_SIZE = 30

    nb_rows = int(len(algorithms)/2) if int(len(algorithms)/2) > 2 else 2
    nb_cols = 2

    fig, axes = plt.subplots(nb_rows, nb_cols, figsize=(larg, long))
    fig.suptitle(title, fontweight="bold", fontsize=TITLE_SIZE, y=TITLE_PAD)

    row = column = 0
    ITER = 3 # constant

    for algoname, algo in algorithms.items():

        cluster_labels = {}

        for i in range(ITER):
            if algoname == "Dummy":
                start_time = time.time()
                cluster_labels[i] = assign_random_clusters(data, algo)
                elapsed_time = time.time() - start_time
            else:
                start_time = time.time()
                algo.fit(data)
                elapsed_time = time.time() - start_time
                cluster_labels[i] = algo.labels_

        for i in range(ITER):
            j = i+1

            if i == 2:
                j = 0

            scores_time.loc[len(scores_time)] = [algoname, i,
                                                 silhouette_score(data,
                                                                  cluster_labels[i],
                                                                  metric="euclidean"),
                                                 adjusted_rand_score(cluster_labels[i],
                                                                     cluster_labels[j]),
                                                 len(set(cluster_labels[i])),
                                                 elapsed_time]

        # plot
        #if nb_rows > 1:
        axis = axes[row, column]
        #else:
        #    axis = axes

        data_to_plot = data.copy()
        data_to_plot["cluster_labels"] = cluster_labels[ITER-1]
        plot_handle = sns.scatterplot(x="tsne-pca-one", y="tsne-pca-two",
                                      data=data_to_plot, hue="cluster_labels",
                                      palette=sns\
                                      .color_palette("hls",
                                                     data_to_plot["cluster_labels"].nunique()),
                                      legend="full", alpha=0.3, ax=axis)

        plt.tight_layout()

        plt.subplots_adjust(left=None,
                            bottom=None,
                            right=None,
                            top=None,
                            wspace=0.3, hspace=0.4)

        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.get_xaxis().tick_bottom()
        axis.get_yaxis().tick_left()
        axis.spines['left'].set_position(('outward', 10))
        axis.spines['bottom'].set_position(('outward', 10))

        axis.set_xlabel('tsne-pca-one', fontsize=LABEL_SIZE, labelpad=LABEL_PAD)
        axis.set_ylabel('tsne-pca-two', fontsize=LABEL_SIZE, labelpad=LABEL_PAD)

        plot_handle.set_xticklabels(plot_handle.get_xticks(), size=TICK_SIZE)

        plot_handle.set_yticklabels(plot_handle.get_yticks(), size=TICK_SIZE)
        axis.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(x)))

        extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                              edgecolor='none', linewidth=0)

        scores = (r'$Silh={:.2f}$' + '\n' + r'$Rand={:.2f}$')\
                 .format(scores_time[scores_time["Algorithme"] == algoname]["silhouette"].mean(),
                         scores_time[scores_time["Algorithme"] == algoname]["Rand"].mean())

        axis.legend([extra], [scores], loc='upper left', fontsize=LEGEND_SIZE)
        title = algoname + '\n Évaluation en {:.2f} secondes'.format(elapsed_time)
        axis.set_title(title, fontsize=SUBTITLE_SIZE, fontweight="bold")

        if column < nb_cols-1:
            column += 1
        else:
            row += 1
            column = 0

    return scores_time

#------------------------------------------
def get_rho_p_value_heatmaps(data, interest_cols, FEATURE, THRESHOLD):
    '''
        Calculates the rho of Spearman for a given dataset
        and a given set of features compared to FEATURE.
        It returns the dataframe corresponding to a heatmap of the rho
        containing only the features whose rho is superior to THRESHOLD.

        Parameters
        ----------------
        data            : pandas dataframe
                          Data containing interest_cols and FEATURE as features.
                          All features must be numeric.

        interest_cols   : list
                          The names of the features to calculate the rho
                          of Spearman compared to FEATURE

        FEATURE         : string
                          The name of the feature to calculate the rho of Spearman
                          against.

        THRESHOLD       : float
                          The function will return only those features whose rho is
                          superior to THRESHOLD

        Returns
        ---------------
        sorted_corrs_df : pandas dataframe
                          The rhos of Spearnma, sorted by feature of highest rho
        sorted_ps_df    : pandas dataframe
                          The p-values associated to the calculated rhos
    '''

    # Calcul du rho de Spearman avec p-value
    corrs, p_values = stats.spearmanr(data[FEATURE],
                                      data[interest_cols])

    # Transformation des arrays en DataFrame
    corrs_df = pd.DataFrame(corrs)
    ps_df = pd.DataFrame(p_values)

    # Renommage des colonnes
    interest_cols.insert(0, FEATURE)
    corrs_df.columns = interest_cols
    ps_df.columns = interest_cols

    # Suppression des colonnes dont le rho est < THRESHOLD
    corrs_interest_index = corrs_df[corrs_df[FEATURE] > THRESHOLD].index

    corrs_df = corrs_df.iloc[corrs_interest_index,
                             corrs_interest_index].reset_index(drop=True)\
                             .sort_values([FEATURE], ascending=False)
    ps_df = ps_df.iloc[corrs_interest_index,
                       corrs_interest_index].reset_index(drop=True)

    # Classement des colonnes par ordre de rho décroissant
    sort_rows_ps_df = pd.DataFrame()

    for index_corr in corrs_df.index.tolist():
        sort_rows_ps_df = pd.concat([sort_rows_ps_df, pd.DataFrame(ps_df.iloc[index_corr, :]).T])

    # Tri par ordre de plus grand rho
    sorted_corrs_df = pd.DataFrame()
    sorted_ps_df = pd.DataFrame()

    for corr_values_index in corrs_df.sort_values([FEATURE], ascending=False).index.tolist():
        sorted_corrs_df = pd.concat([sorted_corrs_df, corrs_df.iloc[:, corr_values_index]], axis=1)
        sorted_ps_df = pd.concat([sorted_ps_df, sort_rows_ps_df.iloc[:, corr_values_index]], axis=1)

    sorted_corrs_df["Index"] = sorted_corrs_df.columns
    sorted_corrs_df.set_index("Index", inplace=True)
    sorted_ps_df["Index"] = sorted_ps_df.columns
    sorted_ps_df.set_index("Index", inplace=True)

    sorted_corrs_df.index.name = None
    sorted_ps_df.index.name = None

    return (sorted_corrs_df, sorted_ps_df)

#------------------------------------------
def get_recency_score(purchase_date, actual_year):
    '''
        Calculates a recency score as the maximum between "10 and
        the number of months that have passed since the customer last purchased"
        and 1.

        Parameters
        ----------------
        purchase_date : timestamp
                        Date of the purchase

        actual_year   : int
                        The reference year to calculate the recency score on.

        Returns
        ---------------
        _ : int
            The calculated recency score
    '''

    return max(1,
               10-round((pd.Timestamp(year=actual_year,
                                      month=12,
                                      day=31) - purchase_date)/np.timedelta64(1, 'M')))

#------------------------------------------
def get_monetary_score(average_customer_basket, AVERAGE_BASKET):
    '''
        Calculates a discrete monetary score as a function of the average value
        of all order values by the customer expressed as a multiple of the average
        brazilian basket value for e-commerce for that year.

        Parameters
        ----------------
        average_customer_basket : float
                                  average basket value

        AVERAGE_BASKET          : float
                                  The average brazilian basket value for e-commerce
                                  for that year

        Returns
        ---------------
        _ : int
            The calculated monetary score
    '''

    raw_m_score = round((average_customer_basket/AVERAGE_BASKET)*100)

    if raw_m_score <= 100:
        return raw_m_score

    if raw_m_score < 200:
        return 100

    if raw_m_score < 300:
        return 200
    if raw_m_score < 400:
        return 300

    if raw_m_score < 500:
        return 400

    if raw_m_score < 600:
        return 500

    if raw_m_score < 700:
        return 600

    if raw_m_score < 800:
        return 700

    if raw_m_score < 900:
        return 800

    if raw_m_score < 1000:
        return 900

    return 1000

#------------------------------------------

def get_exploitable_customer_data(full_dataset, year, AVERAGE_BASKET):
    '''
        Transforms the data to a more exploitable format.

        Parameters
        ----------------
        data           : pandas dataframe
                         Data of interest

        year           : int
                         The year or interest

        AVERAGE_BASKET : float
                         The average brazilian basket value for e-commerce
                         for that year

        Returns
        ---------------
        _ : pandas dataframe
            The data with the new features, one line per customer.
    '''

    data_year = full_dataset[full_dataset["order_purchase_timestamp"].dt.year == year]
    duplicates = data_year[data_year["customer_id"].duplicated()]["customer_id"]
    clients_multiple_lines_year = data_year[data_year["customer_id"].isin(duplicates)]
    clients_unique_line_year = data_year[~data_year["customer_id"].isin(duplicates)]

    new_cols = ["customer_id", "customer_city", "customer_state",
                "nb_orders", "nb_items_order", "canceled_orders",
                "first_order_date", "last_order_date",
                "payment_installments", "payment_sequential", "payment_type",
                "average_basket", "freight_value",
                "product_category_name",
                "product_height_cm", "product_length_cm",
                "product_width_cm", "product_weight_g",
                "product_volume_cm3", "review_score",
                "R_score", "F_score", "M_score"]


    # Customers with several lines : grouping by customer_id

    customer_data_year = pd.DataFrame(columns=new_cols)

    for customer_id, data_customer in clients_multiple_lines_year.groupby("customer_id"):

        means = data_customer.drop(columns=["order_id", "customer_id", "order_purchase_timestamp",
                                            "product_id", "customer_unique_id"]).mean()
        modes = data_customer.drop(columns=["order_id", "customer_id", "order_purchase_timestamp",
                                            "product_id", "customer_unique_id"]).mode().T

        #Preparing row

        customer_city = modes.loc["customer_city", 0]
        customer_state = modes.loc["customer_state", 0]
        nb_orders = data_customer["order_id"].nunique()
        nb_items_order = data_customer[["order_id", "product_id"]].groupby("order_id")\
                                                                  .count()["product_id"].mean()
        canceled_orders = data_customer[data_customer["order_status"] == "canceled"]\
                          ["order_id"].nunique()
        first_order_date = min(data_customer["order_purchase_timestamp"])
        last_order_date = max(data_customer["order_purchase_timestamp"])
        payment_installments = means.loc["payment_installments"]
        payment_sequential = means.loc["payment_sequential"]
        payment_type = modes.loc["payment_type", 0]
        average_basket = data_customer[["order_id", "payment_value"]]\
                         .groupby("order_id").agg("sum")["payment_value"].mean()
        freight_value = means.loc["freight_value"]
        product_category_name = modes.loc["product_category_name", 0]
        product_height_cm = means.loc["product_height_cm"]
        product_length_cm = means.loc["product_length_cm"]
        product_width_cm = means.loc["product_width_cm"]
        product_weight_g = means.loc["product_weight_g"]
        product_volume_cm3 = product_width_cm * product_length_cm * product_height_cm
        review_score = means.loc["review_score"]
        r_score = get_recency_score(max(data_customer["order_purchase_timestamp"]), year)
        f_score = nb_orders if nb_orders <= 10 else 10
        m_score = get_monetary_score(average_basket, AVERAGE_BASKET)

        row = [customer_id, customer_city, customer_state,
               nb_orders, nb_items_order, canceled_orders,
               first_order_date, last_order_date,
               payment_installments, payment_sequential,
               payment_type, average_basket, freight_value, product_category_name,
               product_height_cm, product_length_cm, product_width_cm,
               product_weight_g, product_volume_cm3,
               review_score, r_score, f_score, m_score]

        customer_data_year.loc[len(customer_data_year)] = row

    # Customers with one line

    already_single_customer = pd.DataFrame(columns=new_cols)

    already_single_customer["customer_id"] = clients_unique_line_year["customer_id"]
    already_single_customer["customer_city"] = clients_unique_line_year["customer_city"]
    already_single_customer["customer_state"] = clients_unique_line_year["customer_state"]
    already_single_customer["nb_orders"] = 1
    already_single_customer["nb_items_order"] = 1
    already_single_customer["canceled_orders"] = clients_unique_line_year["order_status"]\
                                         .apply(lambda x: 1 if x != "canceled" else 0)
    already_single_customer["first_order_date"] = clients_unique_line_year\
                                                  ["order_purchase_timestamp"]
    already_single_customer["last_order_date"] = clients_unique_line_year\
                                                 ["order_purchase_timestamp"]
    already_single_customer["payment_installments"] = 1
    already_single_customer["payment_sequential"] = 1
    already_single_customer["payment_type"] = clients_unique_line_year["payment_type"]
    already_single_customer["average_basket"] = clients_unique_line_year["payment_value"]
    already_single_customer["freight_value"] = clients_unique_line_year["freight_value"]
    already_single_customer["product_category_name"] = clients_unique_line_year\
                                                       ["product_category_name"]
    already_single_customer["product_height_cm"] = clients_unique_line_year["product_height_cm"]
    already_single_customer["product_length_cm"] = clients_unique_line_year["product_length_cm"]
    already_single_customer["product_width_cm"] = clients_unique_line_year["product_width_cm"]
    already_single_customer["product_weight_g"] = clients_unique_line_year["product_weight_g"]
    already_single_customer["product_volume_cm3"] = clients_unique_line_year["product_height_cm"]\
                                                    * clients_unique_line_year["product_length_cm"]\
                                                    * clients_unique_line_year["product_width_cm"]
    already_single_customer["review_score"] = clients_unique_line_year["review_score"]
    already_single_customer["R_score"] = clients_unique_line_year["order_purchase_timestamp"]\
                                         .apply(lambda x: get_recency_score(x, year))
    already_single_customer["F_score"] = clients_unique_line_year["order_status"]\
                                         .apply(lambda x: 1 if x != "canceled" else 0)
    already_single_customer["M_score"] = clients_unique_line_year["payment_value"]\
                                         .apply(lambda x: get_monetary_score(x, AVERAGE_BASKET))

    # Concaténation des deux dataframes
    final_df = pd.concat([customer_data_year, already_single_customer])
    final_df["nb_orders"] = final_df["nb_orders"].astype("int")
    final_df["canceled_orders"] = final_df["canceled_orders"].astype("int")
    final_df["R_score"] = final_df["R_score"].astype("int")
    final_df["F_score"] = final_df["F_score"].astype("int")
    final_df["M_score"] = final_df["M_score"].astype("int")

    final_df = final_df.drop(columns=["product_height_cm",
                                      "product_length_cm",
                                      "product_width_cm"])

    final_df["first_order_date"] = pd.to_datetime(final_df["first_order_date"],
                                                  format='%Y-%m-%d %H:%M:%S')
    final_df["first_order_year"] = final_df["first_order_date"].apply(lambda x: x.year)
    final_df["first_order_month"] = final_df["first_order_date"].apply(lambda x: x.month)
    final_df["first_order_day"] = final_df["first_order_date"].apply(lambda x: x.day)
    final_df["first_order_hour"] = final_df["first_order_date"].apply(lambda x: x.hour)
    final_df["first_order_min"] = final_df["first_order_date"].apply(lambda x: x.minute)

    final_df["last_order_date"] = pd.to_datetime(final_df["last_order_date"],
                                                 format='%Y-%m-%d %H:%M:%S')
    final_df["last_order_year"] = final_df["last_order_date"].apply(lambda x: x.year)
    final_df["last_order_month"] = final_df["last_order_date"].apply(lambda x: x.month)
    final_df["last_order_day"] = final_df["last_order_date"].apply(lambda x: x.day)
    final_df["last_order_hour"] = final_df["last_order_date"].apply(lambda x: x.hour)
    final_df["last_order_min"] = final_df["last_order_date"].apply(lambda x: x.minute)

    return final_df
