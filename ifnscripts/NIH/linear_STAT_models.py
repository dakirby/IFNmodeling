import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures, scale
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE
import os
import re
#from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

#from yellowbrick.features import RFECV


def read_ImmGen_csv(fname):
    df = pd.read_csv(fname, header=None, low_memory=False)
    df_feature_names = np.concatenate((['Cell_type'], df.iloc[:, 1].values[1:]))
    df = df.T.drop([0, 1], axis=0)
    df.columns = df_feature_names
    df = df.set_index('Cell_type')
    return df


def mean_by_index_group(df):
    """
    Takes groupby on index and then averages each column in group
    :param df: index is cell type, each row is an entry, columns are genes
    :return: pandas DataFrame with mean gene expression for each cell type
    """
    mean_gene_response = df.groupby(level=0)
    mean_gene_response_df = {g[0]: g[1] for g in list(mean_gene_response)}
    gene_labels = df.columns.values
    cell_type_labels = list(mean_gene_response_df.keys())
    mean_response_data = [np.mean(mean_gene_response_df[cell_type_labels[idx]].values.astype(float), axis=0) for idx in range(len(cell_type_labels))]
    mean_response_df = pd.DataFrame(data=mean_response_data, index=cell_type_labels, columns=gene_labels)
    return mean_response_df


def custom_linear(feature_names, features, response_variables):
    # Fit using best features specified by feature_names
    model = LinearRegression()
    if len(feature_names)==1:
        predictor = features[feature_names].values.reshape((-1,1))
    else:
        predictor = features[feature_names].values
    model.fit(predictor, response_variables.values)

    r2 = model.score(predictor, response_variables.values)
    r2_adj = 1 - (1 - r2) * (len(response_variables.values) - 1) / (len(response_variables.values) - np.shape(predictor)[1] - 1)
    print(r2_adj)

def multivariateGrid(col_x, col_y, col_k, df, k_is_color=False, scatter_alpha=.5):
    """
    Source: https://stackoverflow.com/questions/35920885/how-to-overlay-a-seaborn-jointplot-with-a-marginal-distribution-histogram-fr
    :param col_x: DataFrame column name to use for x data
    :param col_y: DataFrame column name to use for y data
    :param col_k: DataFrame column name to use for hue
    :param df: The DataFrame to use
    :param k_is_color: True if the column for col_k is the value to pass to the color kwarg. Otherwise use hematopoetic_colormap
    :param scatter_alpha: transparency for the scatter plot
    :return: sns.JointGrid object
    """
    stem_palette = sns.color_palette("Greys", 5)
    myeloid_palette = sns.color_palette("Greens", 5)
    lymphoid_palette = sns.color_palette("cubehelix", 8)[::-1]

    hematopoetic_colormap = {"SC": stem_palette[3],
                             "Mo": myeloid_palette[0], "GN": myeloid_palette[1], "MF": myeloid_palette[2], "DC": myeloid_palette[3],
                             "preB": lymphoid_palette[1], "preT": lymphoid_palette[2], "B": lymphoid_palette[6], "T": lymphoid_palette[3],
                             "NK": lymphoid_palette[4], "NKT": lymphoid_palette[5],
                             "BEC": "tomato", "FRC": "crimson"}
    def colored_scatter(x, y, c=None):
        def scatter(*args, **kwargs):
            args = (x, y)
            if c is not None:
                kwargs['c'] = c
            kwargs['alpha'] = scatter_alpha
            kwargs['s'] = 3
            plt.scatter(*args, **kwargs)
            ax = plt.gca()
            ax.set_xlim((-40, 80))
            ax.set_ylim((-15, 40))

        return scatter

    g = sns.JointGrid(x=col_x, y=col_y, data=df)
    color = None
    legends=[]
    for name, df_group in df.groupby(col_k):
        legends.append(name)
        if k_is_color:
            color=name
        elif name in hematopoetic_colormap.keys():
            color=hematopoetic_colormap[name]
        g.plot_joint(colored_scatter(df_group[col_x], df_group[col_y], color))
        if len(df_group[col_x].values)!=1 and len(df_group[col_y].values)!=1:
            sns.distplot(df_group[col_x].values, ax=g.ax_marg_x, color=color)
            sns.distplot(df_group[col_y].values, ax=g.ax_marg_y, color=color, vertical=True)
    # Do also global Hist:
    #sns.distplot(df[col_x].values, ax=g.ax_marg_x, color='grey')
    #sns.distplot(df[col_y].values.ravel(), ax=g.ax_marg_y, color='grey', vertical=True)
    plt.legend(legends, markerscale=6.)
    return g


# Import data set and split into predictor variables (receptors, JAKs, SOCS, etc.) and response variables (STATs)
ImmGen_df = pd.read_excel('ImmGen_signaling_with_protein_response.xlsx', sheet_name='Sheet1', axis=1)
cell_type_to_index = pd.Series(ImmGen_df.index, index=ImmGen_df.Cell_type.values).to_dict()

#create df of results and features from full dataframe
response_variable_names = ['pSTAT1', 'pSTAT3']
response_variables = ImmGen_df[response_variable_names]

predictor_variable_names = [c for c in ImmGen_df.columns.values if c not in response_variable_names + ['Cell_type']]
predictor_variables = ImmGen_df[predictor_variable_names]

# center and normalize the variables
x_centered = scale(predictor_variables, with_mean='True', with_std='False')
centered_predictor_variables = pd.DataFrame(x_centered, columns=predictor_variables.columns)

# --------------------
# Quadratic regression
# --------------------
def quadratic():
    poly = PolynomialFeatures(degree=2, interaction_only=True)
    poly_variables = poly.fit_transform(centered_predictor_variables)

    poly_var_train, poly_var_test, poly_res_train, poly_res_test = train_test_split(poly_variables, response_variables.values,
                                                                          test_size=0.25, random_state=4)
    # Create linear regression object
    regr = LinearRegression()
    # Train the model using the training sets
    regr.fit(poly_var_train, poly_res_train)
    score = regr.score(poly_var_test, poly_res_test)
    # Make predictions using the testing set
    STAT_pred = regr.predict(poly_var_test)
    # The coefficients
    #print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Quadratic regression mean squared error: %.2f"
          % mean_squared_error(poly_res_test, STAT_pred))
    # Explained variance score: 1 is perfect prediction
    print('Quadratic regression variance score: %.2f' % r2_score(poly_res_test, STAT_pred))

# --------------------------
# Lasso quadratic regression
# --------------------------
def lasso_quadratic():
    poly = PolynomialFeatures(degree=2, interaction_only=True)
    poly_variables = poly.fit_transform(centered_predictor_variables)

    poly_var_train, poly_var_test, poly_res_train, poly_res_test = train_test_split(poly_variables, response_variables.values,
                                                                          test_size=0.25, random_state=4)
    alpha_tests = np.logspace(-3, 2)
    r2_array = []
    for a in alpha_tests:
        clf = Lasso(alpha=a)
        clf.fit(poly_var_train, poly_res_train)
        STAT_pred = clf.predict(poly_var_test)
        s = r2_score(poly_res_test, STAT_pred)
        r2_array.append(s)
    best_alpha = alpha_tests[np.where(r2_array==max(r2_array))]
    print("Lasso quadratic regression best alpha value: {}".format(best_alpha))
    print("Lasso quadratic regression best r2 score: {}".format(max(r2_array)))
    plt.figure()
    ax=plt.gca()
    ax.set_xscale('log')
    plt.plot(alpha_tests, r2_array)
    plt.show()

# --------------------------
# Lasso linear regression
# --------------------------
def lasso_linear():
    linear_var_train, linear_var_test, linear_res_train, linear_res_test = train_test_split(centered_predictor_variables.values, response_variables.values,
                                                                          test_size=0.25, random_state=4)
    alpha_tests = np.logspace(-3, 2)
    r2_array = []
    for a in alpha_tests:
        clf = Lasso(alpha=a)
        clf.fit(linear_var_train, linear_res_train)
        STAT_pred = clf.predict(linear_var_test)
        s = r2_score(linear_res_test, STAT_pred)
        r2_array.append(s)
    best_alpha = alpha_tests[np.where(r2_array==max(r2_array))]
    clf = Lasso(alpha=best_alpha)
    clf.fit(linear_var_train, linear_res_train)
    STAT_pred = clf.predict(linear_var_test)

    print("Lasso linear regression best alpha value: {}".format(best_alpha))
    print("Lasso linear regression best r2 score: {}".format(max(r2_array)))
    plt.figure()
    ax=plt.gca()
    ax.set_xscale('log')
    plt.plot(alpha_tests, r2_array)
    plt.show()

    fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(8,4))
    ax[0].scatter(linear_res_test[:,0], STAT_pred[:,0])
    ax[0].plot(np.linspace(min(linear_res_test[:,0]), max(linear_res_test[:,0])),
               np.linspace(min(linear_res_test[:,0]), max(linear_res_test[:,0])), 'k--')
    ax[0].set_title('pSTAT1')
    ax[0].set_xlabel('measured')
    ax[0].set_ylabel('predicted')

    ax[1].scatter(linear_res_test[:,1], STAT_pred[:,1])
    ax[1].set_title('pSTAT3')
    ax[1].set_xlabel('measured')
    ax[1].set_ylabel('predicted')
    ax[1].plot(np.linspace(min(linear_res_test[:,1]), max(linear_res_test[:,1])),
               np.linspace(min(linear_res_test[:,1]), max(linear_res_test[:,1])), 'k--')
    plt.show()

    print(linear_res_test)
    print(STAT_pred)


# -----------------------
# ImmGen genetic analysis
# -----------------------
def gene_response(plot_heatmap=False):
    gene_response_df = read_ImmGen_csv("ImmGen_IFNg_response_GSE112876_Normalized_Data.csv")
    gene_naive_df = read_ImmGen_csv('ImmGen_raw.csv')
    gene_response_df.index = [i.split(".")[0] for i in gene_response_df.index.values]
    gene_naive_df.index = [i.split(".")[0] for i in gene_naive_df.index.values]
    core_cell_types = []
    for x in gene_response_df.index.values:
        if x not in core_cell_types:
            core_cell_types.append(x)
    common_core_cell_types = [i for i in core_cell_types if i in gene_naive_df.index.values]

    # Taking the group mean is hard in pandas. This was the hacky solution I came up with:
    mean_response_df = mean_by_index_group(gene_response_df.loc[common_core_cell_types])

    # Treat each cell type copy in gene_naive_df.loc[common_core_cell_types] as a data point to normalize with for
    # matching cell type in mean_response_df
    points = []
    for cell_type in common_core_cell_types:
        fold_change = np.array(mean_response_df.loc[cell_type].values).astype(float) / np.array(gene_naive_df.loc[cell_type].values).astype(float)
        temp_df = pd.DataFrame(data=fold_change, index=[cell_type for _ in range(len(fold_change))], columns=mean_response_df.columns)
        points.append(temp_df)
    fold_change_df = pd.concat(points)

    if plot_heatmap:
        log2_fold_change = fold_change_df.apply(np.log2)
        log2_fold_change = log2_fold_change.loc[:, ~log2_fold_change.columns.duplicated()]
        keep_list = []
        for c in range(len(log2_fold_change.columns.values)):
            if abs(log2_fold_change.iloc[:, c].mean()) < np.log2(10):
            #if log2_fold_change.columns.values[c] not in [s.title() for s in predictor_variable_names]:
                keep_list.append(False)
            else:
                keep_list.append(True)
        print(sum(keep_list))
        downsampled_df = log2_fold_change[log2_fold_change.columns[keep_list]]
        sns.clustermap(downsampled_df)
        plt.show()
    return fold_change_df


# -----------------
# Feature selection
# -----------------
def feature_selection(poly=False, best_n_features=1, visualize=False, plot_measured_vs_pred=True,
                      barplot=True, extrapolate=False, pSTAT_score=False):
    # Feature selection set up
    model = LinearRegression()
    rfe = RFE(estimator=model, n_features_to_select=1, step=1)
    # Data set up
    if poly==True:
        poly = PolynomialFeatures(degree=2, interaction_only=True)
        poly_features = poly.fit_transform(centered_predictor_variables)
        features = pd.DataFrame(poly_features, columns=poly.get_feature_names(ImmGen_df.columns))
        labels = features.columns
    else:
        features = centered_predictor_variables
        labels = centered_predictor_variables.columns
    # Perform feature selection. RFE method can only handle 1D y data so do each and combine as a heuristic
    rfe.fit(features, ImmGen_df['pSTAT1'])
    ranking_pSTAT1 = rfe.ranking_
    #print("pSTAT1", [x for _,x in sorted(zip(ranking_pSTAT1, labels))])
    rfe.fit(features, ImmGen_df['pSTAT3'])
    ranking_pSTAT3 = rfe.ranking_
    #print("pSTAT3", [x for _, x in sorted(zip(ranking_pSTAT3, labels))])
    combined_rank = [x for _, x in sorted(zip(np.add(ranking_pSTAT3,ranking_pSTAT1), labels))]
    print("\nCombined ranking:", combined_rank[0:10]," ... and worse predictors\n")

    # Visualize feature selection
    if visualize == True:
        r2_record = []
        for i in range(1, len(combined_rank)):
            model = LinearRegression()
            if i == 1:
                predictor = features[combined_rank[0:i]].values.reshape((-1, 1))
            else:
                predictor = features[combined_rank[0:i]].values
            model.fit(predictor, response_variables.values)
            r2 = model.score(predictor, response_variables.values)
            r2_adj = 1 - (1 - r2) * (len(response_variables.values) - 1) / (
                        len(response_variables.values) - np.shape(predictor)[1] - 1)
            r2_record.append(r2_adj)
        plt.plot(r2_record, 'k--')
        plt.xlabel("number of features")
        plt.ylabel(r'$R^{2}_{adj}$')
        plt.show()

    # Fit using best predictive feature
    model = LinearRegression()
    if best_n_features==1:
        predictor = features[combined_rank[0:best_n_features]].values.reshape((-1,1))
    else:
        predictor = features[combined_rank[0:best_n_features]].values
    model.fit(predictor, response_variables.values)
    STAT_pred = model.predict(predictor)
    r2 = model.score(predictor, response_variables.values)
    r2_adj = 1 - (1 - r2) * (len(response_variables.values) - 1) / (len(response_variables.values) - np.shape(predictor)[1] - 1)

    if plot_measured_vs_pred==True:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
        ax[0].scatter(response_variables.values[:, 0], STAT_pred[:, 0])
        ax[0].plot(np.linspace(min(response_variables.values[:, 0]), max(response_variables.values[:, 0])),
                   np.linspace(min(response_variables.values[:, 0]), max(response_variables.values[:, 0])), 'k--')
        ax[0].set_title('pSTAT1')
        ax[0].set_xlabel('measured')
        ax[0].set_ylabel('predicted')

        ax[1].scatter(response_variables.values[:, 1], STAT_pred[:, 1])
        ax[1].set_title('pSTAT3')
        ax[1].set_xlabel('measured')
        ax[1].set_ylabel('predicted')
        ax[1].plot(np.linspace(min(response_variables.values[:, 1]), max(response_variables.values[:, 1])),
                   np.linspace(min(response_variables.values[:, 1]), max(response_variables.values[:, 1])), 'k--')
        if best_n_features==1:
            plt.suptitle("Regression on feature " + str(combined_rank[0:best_n_features])+"\n" + r"$R^{2}_{adj}$" + " = {:.2f}".format(r2_adj))
        else:
            plt.suptitle("Regression on features " + str(combined_rank[0:best_n_features])+"\n" + r"$R^{2}_{adj}$" + " = {:.2f}".format(r2_adj))
        plt.show()

    # barplot
    if barplot==True:
        arr1 = [[ImmGen_df.Cell_type.values[i], 'CyTOF', float(response_variables.values[i, 0]), float(response_variables.values[i, 1])] for i in range(len(response_variables.values))]
        arr2 = [[ImmGen_df.Cell_type.values[i], 'Linear Regression', float(STAT_pred[i, 0]), float(STAT_pred[i, 1])] for i in range(len(STAT_pred))]
        df = pd.DataFrame(data=arr2+arr1, columns=['Cell_type', 'Class', 'pSTAT1','pSTAT3'])

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12.8, 5))
        fig.suptitle(r'$R^{2}_{adj}$' +'= {:.2f}'.format(r2_adj))
        subplot_titles = ['pSTAT1', 'pSTAT3']
        sns.barplot(x="Cell_type", y="pSTAT1", data=df, hue='Class', ax=ax[0])
        sns.barplot(x="Cell_type", y="pSTAT3", data=df, hue='Class', ax=ax[1])
        for i in [0, 1]:
            ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=90)
            ax[i].set_title(subplot_titles[i])
        plt.tight_layout()
        plt.show()

    # Extrapolate to all ImmGen cell types
    if extrapolate == True:
        ImmGen_raw = read_ImmGen_csv('ImmGen_raw.csv')
        top_features = [i.title() for i in combined_rank[0:best_n_features]]

        centered_ImmGen_raw = scale(ImmGen_raw, with_mean='True', with_std='False')
        centered_ImmGen_raw_df = pd.DataFrame(centered_ImmGen_raw, columns=ImmGen_raw.columns, index=ImmGen_raw.index)
        hematopoetic_class = [immgenName.split('.')[0] for immgenName in ImmGen_raw.index]

        # ImmGen has duplicate column names because different ProbeID's correspond to the same gene name.
        # I have no prior on which ProbeID to select, so here I simply use the first occurence in the DataFrame.
        ImmGen_raw_no_duplicates = centered_ImmGen_raw_df.loc[:, ~centered_ImmGen_raw_df.columns.duplicated()]
        predictor = ImmGen_raw_no_duplicates[top_features].values
        STAT_extrapolate = model.predict(predictor)
        STAT_extrapolate_df = pd.DataFrame(data=STAT_extrapolate, index=ImmGen_raw_no_duplicates.index, columns=['pSTAT1', 'pSTAT3'])
        STAT_extrapolate_df.insert(0, 'Hematopoetic_class', hematopoetic_class)
        # Control and Test cell types should be dropped for visualization
        STAT_extrapolate_df = STAT_extrapolate_df[~STAT_extrapolate_df.Hematopoetic_class.str.contains('Control|TEST', flags=re.IGNORECASE, regex=True)]
        # Save in CSV for later
        STAT_extrapolate_df.sort_values(by=['pSTAT1']).to_csv(os.path.join(os.getcwd(), 'STAT_extrapolate.csv'))

        # Cluster into 3 clusters for visualization
        #y_pred = KMeans(n_clusters=2, random_state=4).fit_predict(STAT_extrapolate)
        # Plot pSTAT space
        #plt.scatter(STAT_extrapolate[:, 0], STAT_extrapolate[:, 1], c=y_pred, alpha=0.2)
        #plt.xlabel('pSTAT1')
        #plt.ylabel('pSTAT3')

        query_cell_types = ["NK", "NKT", "SC", "preB", "preT", "T", "B", "GN", "Mo", "DC", "MF", "FRC", "BEC"]
        #query_cell_types = ["NK", "NKT", "SC", "preB", "preT", "T", "B"]
        #query_cell_types = ["SC", "GN", "Mo", "DC", "MF", "FRC", "BEC"]
        jointGrid = multivariateGrid("pSTAT1", "pSTAT3", "Hematopoetic_class",
                                     STAT_extrapolate_df[STAT_extrapolate_df['Hematopoetic_class'].isin(query_cell_types)],
                                     k_is_color=False, scatter_alpha=0.8)
        #plt.suptitle("Regression on features " + str(
        #    combined_rank[0:best_n_features]) + "\n" + r"$R^{2}_{adj}$" + " = {:.2f}".format(r2_adj))
        plt.tight_layout()
        jointGrid.savefig("ImmGen_pSTAT_extrapolation_with_cell_type.pdf")

    if pSTAT_score == True:
        ImmGen_raw = read_ImmGen_csv('ImmGen_raw.csv')
        centered_ImmGen_raw = scale(ImmGen_raw, with_mean='True', with_std='False')
        centered_ImmGen_raw_df = pd.DataFrame(centered_ImmGen_raw, columns=ImmGen_raw.columns, index=ImmGen_raw.index)
        # ImmGen has duplicate column names because different ProbeID's correspond to the same gene name.
        # I have no prior on which ProbeID to select, so here I simply use the first occurence in the DataFrame.
        ImmGen_raw_no_duplicates = centered_ImmGen_raw_df.loc[:, ~centered_ImmGen_raw_df.columns.duplicated()]
        top_features = [i.title() for i in combined_rank[0:best_n_features]]
        predictor = ImmGen_raw_no_duplicates[top_features].values
        STAT_extrapolate = model.predict(predictor)
        STAT_extrapolate_df = pd.DataFrame(data=STAT_extrapolate, index=ImmGen_raw_no_duplicates.index, columns=['pSTAT1', 'pSTAT3'])

        # Build scoring function
        pca = PCA(n_components=2)
        pca.fit(STAT_extrapolate)
        def pSTAT_score(pSTAT_vec):
            return pSTAT_vec[0]*pca.components_[0][0] + pSTAT_vec[1]*pca.components_[0][1]

        # Score all cells for their pSTAT1/3 IFNg response
        pSTAT_score_response = [pSTAT_score(STAT_extrapolate[i]) for i in range(len(STAT_extrapolate))]
        STAT_extrapolate_df.insert(0, 'pSTAT_score', pSTAT_score_response)
        STAT_extrapolate_df = STAT_extrapolate_df.loc[~STAT_extrapolate_df.index.str.contains('Control|TEST', flags=re.IGNORECASE, regex=True)]
        STAT_extrapolate_df.index = [i.split(".")[0] for i in STAT_extrapolate_df.index]

        # Get fold change in genes due to IFNg stimulation, according to ImmGen database
        fold_change_df = gene_response()
        log2_fold_change = fold_change_df.apply(np.log2)
        common_cell_types = [i for i in log2_fold_change.index if i in STAT_extrapolate_df.index]

        # Take group averages
        temp1 = {}
        temp2 = {}
        for c in common_cell_types:
            temp1[c] = np.mean(log2_fold_change.loc[c].values, axis=0)
            temp2[c] = np.mean(STAT_extrapolate_df.loc[c].values, axis=0)
        mean_log2_fold_change = pd.DataFrame.from_dict(temp1, orient='index', columns=log2_fold_change.columns)
        mean_pSTAT = pd.DataFrame.from_dict(temp2, orient='index', columns=STAT_extrapolate_df.columns)

        # Score the mean transcriptional response
        mean_log2_fold_change.insert(0, 'transcript_score', mean_log2_fold_change.dot(mean_log2_fold_change.loc['B']))

        plt.figure()
        plt.scatter(mean_pSTAT.pSTAT_score.values, mean_log2_fold_change.transcript_score.values)
        plt.xlabel('pSTAT score')
        plt.ylabel('transcript score')
        p1 = plt.gca()
        for line in range(0, len(mean_pSTAT.pSTAT_score.values)):
            p1.text(mean_pSTAT.pSTAT_score.values[line] + 0.2, mean_log2_fold_change.transcript_score.values[line],
                    mean_pSTAT.index.values[line],
                    horizontalalignment='left', size='medium', color='black', weight='semibold')
        plt.show()


if __name__ == "__main__":
    # custom_linear(['STAT1'], centered_predictor_variables, response_variables)
    gene_response(plot_heatmap=True)

    #feature_selection(best_n_features=4, visualize=False, plot_measured_vs_pred=False,
    #                  barplot=False, extrapolate=False, pSTAT_score=True)

