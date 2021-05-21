import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import interpolate
from scipy.spatial import ConvexHull
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from matplotlib.pylab import rcParams

##########################################################
# visualizza le prime n e le ultime n righe del data_frame
def display_n(data_frame,n): 
    with pd.option_context('display.max_rows',n*2):
        display(data_frame)

##########################################################
# visualizza una Heatmap della correlazione
def plot_heatmap_corr(data_frame):
    plt.figure(figsize=(16, 6))
    corr = data_frame.corr()
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    cut_off = 0.2  # only show cells with abs(correlation) at least this value
    extreme_1 = 0.75  # show with a star
    extreme_2 = 0.85  # show with a second star
    extreme_3 = 0.90  # show with a third star
    mask |= np.abs(corr) < cut_off
    corr = corr[~mask]  # fill in NaN in the non-desired cells

    remove_empty_rows_and_cols = True
    if remove_empty_rows_and_cols:
        wanted_cols = np.flatnonzero(np.count_nonzero(~mask, axis=1))
        wanted_rows = np.flatnonzero(np.count_nonzero(~mask, axis=0))
        corr = corr.iloc[wanted_cols, wanted_rows]

    annot = [[f"{val:.4f}"
              + ('' if abs(val) < extreme_1 else '\n★')  # add one star if abs(val) >= extreme_1
              + ('' if abs(val) < extreme_2 else '★')  # add an extra star if abs(val) >= extreme_2
              + ('' if abs(val) < extreme_3 else '★')  # add yet an extra star if abs(val) >= extreme_3
              for val in row] for row in corr.to_numpy()]
    heatmap = sns.heatmap(corr, vmin=-1, vmax=1, annot=annot, fmt='', cmap='BrBG')
    heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize': 18}, pad=16)
    plt.show()    

##########################################################
# calcola la correlazione
def compute_correlation(x,y):
    w_hat = np.sum((x-np.mean(x))*(y-np.mean(y)))/np.sum((x-np.mean(x))**2)
    b_hat = np.mean(y) - w_hat * np.mean(x)
    return b_hat, w_hat, np.corrcoef(x,y)[0,1]

##########################################################
# visualizza la retta di correlazione
def plot_correlation(data_frame, xcol, ycol, xlabel, ylabel):
    x = data_frame[xcol].values
    y = data_frame[ycol].values
    plt.figure(figsize=(16, 8))
    plt.scatter(x, y, c='black')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # regression line
    b_hat, w_hat, corr = compute_correlation(x, y)
    x_line = np.array([np.min(x), np.max(x)])
    y_line = b_hat + w_hat * x_line
    plt.title(f'correlazione {corr:.3f}')
    plt.plot(x_line, y_line)
    plt.show()

##########################################################
# Split random to train, test set
def split_train_test(df, test_size, x_features, y_feature):
    indexes = [df.columns.get_loc(col) for col in x_features]
    data_x = df.values[:,indexes]    
    data_x = data_x.astype(np.float32)
    data_y = df[y_feature].values
    data_y = data_y.astype(np.float32)
    
    return train_test_split(data_x, data_y, test_size=test_size, random_state=1)
    
##########################################################
# visualizza la distribuzione degli errori
def plot_errors(val_y, y_pred):
    # Calcola l'errore come scostamento delle predizioni dal valore reale
    errors = np.abs(val_y - y_pred) 

    plt.figure(figsize=(14, 4))
    plt.title("Distribuzione degli errori")
    plt.hist(x = errors)
    plt.show()

##########################################################
# Visualizza l'andamento reale e quello predetto
def plot_line(x,y):
    plt.figure(figsize=(14, 4))
    plt.plot(x, y)
    plt.scatter(x, y)
    plt.show()

##########################################################
# Visualizza l'andamento reale e quello predetto
def plot_real_prediction(y_real, y_pred, x_label, y_label):
    plt.figure(figsize=(14, 4))
    plt.title("Confronto tra valori reali e valori predetti")
    plt.plot(y_real, label='Real')
    plt.plot(y_pred, label='Prediction')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    
##########################################################
# display confusion matrix
def plot_confusion_matrix(y_real, y_pred):
    cmat = confusion_matrix(y_real, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(pd.DataFrame(cmat), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    print(f'TN - True Negative: {cmat[0,0]}')
    print(f'FP - False Positive: {cmat[0,1]}')
    print(f'FN - False Negative: {cmat[1,0]}')
    print(f'TP - True Positive: {cmat[1,1]}')
    print(f'Accuracy Rate: {np.divide(np.sum([cmat[0,0],cmat[1,1]]),np.sum(cmat)):.3f}')
    print(f'Misclassification Rate: {np.divide(np.sum([cmat[0,1],cmat[1,0]]),np.sum(cmat)):.3f}')

##########################################################
# visualizza al ROC curve
def plot_roc_curve(y_real, y_pred):
    fpr, tpr, _ = metrics.roc_curve(y_real,  y_pred)
    auc = metrics.roc_auc_score(y_real, y_pred)

    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={auc:.4f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    fig.show()

##########################################################
# Trova il miglior Estimator tramite Grid Search
def search_best_estimator(estimator, param_grid, train_x, train_y, folds=3):
    # Creazione di un oggetto di tipo GridSearchCV
    grid_search_cv = GridSearchCV(estimator, param_grid, scoring='neg_mean_squared_error', cv=folds)
    # Esecuzione della ricerca degli iperparametri 
    grid_search_cv.fit(train_x, train_y)
    print(f'Best Params: {grid_search_cv.best_params_}')
    print(f'Best Score: {grid_search_cv.best_score_:.3f}')

    best_estimator = grid_search_cv.best_estimator_
    best_estimator.fit(train_x, train_y)
    return best_estimator

##########################################################
# applica l'estimator e calcola RMSE root-mean-square error 
# su Train e Test. Ritorna R2 Score
def evaluate_estimator(estimator, train_x, train_y, test_x, test_y):
    # Ottenimento delle predizioni (train) e calcolo RMSE
    train_y_pred = estimator.predict(train_x)
    rmse = np.sqrt(mean_squared_error(train_y, train_y_pred))
    print(f'Train RMSE: {rmse:.4f}') 

    # Ottenimento delle predizioni (test) e calcolo RMSE
    test_y_pred = estimator.predict(test_x)
    rmse = np.sqrt(mean_squared_error(test_y, test_y_pred))
    print(f'Test (RMSE): {rmse:.4f}') 
    
    # R2 score
    print(f'R2 score: {estimator.score(test_x, test_y):.4f}')
    return test_y_pred

##########################################################
# x, x_2, x_3, ... x_power
def x_poly(data, power):
    predictors=['x']
    if power>=2:
        predictors.extend([f'x_{i}' for i in range(2,power+1)])
        
    x = data[predictors]
    return x
    
##########################################################
# Ritorna un array con [r2, rmse, rss, intercept, coefficienti]
def regression_metrics(reg, x, y, y_pred):    
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = reg.score(x, y)
    rss = sum((y_pred - y)**2)
    ret = []
    ret.extend([r2])
    ret.extend([rmse])
    ret.extend([rss])
    ret.extend([reg.intercept_])
    ret.extend(reg.coef_)
    return ret

##########################################################
def plot_regression(data, y_pred, idx, tot_plots, num_cols, title):
    rcParams['figure.figsize'] = 12, 16
    num_rows = math.ceil(tot_plots/num_cols)
    #plt.figure(figsize=(12, 16))
    plt.subplot(tot_plots, num_cols, idx)
    plt.subplots_adjust(hspace=0.5)
    plt.plot(data['x'], y_pred)
    plt.plot(data['x'], data['y'], '.')
    plt.title(title)
    
##########################################################
def plot_kmeans_cluster(x, y, clusters):
    kmeans = KMeans(n_clusters = clusters, random_state=0)
    df = pd.DataFrame(np.column_stack([x,y]), columns=['x','y'])
    df['cluster'] = kmeans.fit_predict(df[['x', 'y']])
    all_colors = ["red", "coral", "gold", "yellowgreen", "green", "mediumaquamarine",
                  "mediumturquoise", "cornflowerblue", "blue", "purple"]
    df['c'] = df['cluster'].apply(lambda x: all_colors[int(x)])

    # get centroids
    centroids = kmeans.cluster_centers_
    cen_x = [i[0] for i in centroids] 
    cen_y = [i[1] for i in centroids]

    fig, ax = plt.subplots(1, figsize=(8,8))
    plt.scatter(df.x, df.y, c=df.c, alpha = 0.6, s=10)
    plt.scatter(cen_x, cen_y, marker='^', s=70)
        
    for i in df.cluster.unique():
        # get the convex hull
        points = df[df.cluster == i][['x', 'y']].values
        hull = ConvexHull(points)
        x_hull = np.append(points[hull.vertices,0],
                        points[hull.vertices,0][0])
        y_hull = np.append(points[hull.vertices,1],
                        points[hull.vertices,1][0])
        # plot shape
        plt.fill(x_hull, y_hull, alpha=0.3, c=all_colors[i])
    return df

def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
def plot_log_reg(x, y, data, clf, xmin=None, xmax=None, alpha=1, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.scatter(data[x], data[y], color='black', zorder=20, alpha=alpha)
    if xmin is None:
        xmin = x.min()
    if xmax is None:
        xmax = x.max()
    X_test = np.linspace(xmin, xmax, 300)

    loss = scipy.special.expit(X_test * clf.coef_ + clf.intercept_).ravel()
    ax.plot(X_test, loss, linewidth=3)

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.tight_layout()
    sns.despine()
    return fig, ax    

#############################################   
# plot error rate
def plot_error_rate(errors):
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 40), errors, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')        