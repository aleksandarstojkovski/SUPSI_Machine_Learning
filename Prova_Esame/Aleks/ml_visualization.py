import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from skimage import color, transform, feature
from sklearn.metrics import confusion_matrix
import itertools
import warnings

import ml_utilities

# Configurazione di Matplotlib
plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def legend_drawing(numFigure, colors, labels,loc):
    plt.figure(numFigure)
    items=[mpatches.Rectangle((0, 0), 1, 1, fc=c) for c in colors]
    plt.legend(items, labels, loc=loc)
    plt.draw()

def plot_performance_curves(epochs_training_loss,epochs_training_accuracy,epochs_validation_accuracy):
    _, ax1 = plt.subplots(dpi=96, figsize=(8, 6))
    ax2 = ax1.twinx()
    ax1.set_ylim(0,epochs_training_loss[0] * 1.1)
    ax2.set_ylim(40,100)
    ax1.plot(range(0,len(epochs_training_loss)), epochs_training_loss, 'r')
    ax2.plot(range(0,len(epochs_training_accuracy)), epochs_training_accuracy, 'b')
    ax2.plot(range(0,len(epochs_validation_accuracy)), epochs_validation_accuracy, 'g')
    legend_drawing(1,('r','b','g','c'),("Loss","Training Accuracy","Validation Accuracy"),0)
    ax1.set_xlabel('# Epoche')
    ax1.set_ylabel('Loss')
    ax2.set_ylabel('Accuratezza %')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes,title='Confusion matrix',cmap=plt.cm.Blues,figsize=(8,6)):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(dpi=96,figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def get_colors_from_colormap(colorMap, step):
    cmap = matplotlib.cm.get_cmap('Spectral')

    colors = []
    #for i in frange(0, 1 + step, step):
    for i in np.arange(0.0, 1.0 + step, step):
        colors.append(matplotlib.colors.to_hex(cmap(i)))

    return colors
	
def calculate_2D_min_max(dataset,border):
    xData = [d[0] for d in dataset]
    yData = [d[1] for d in dataset]

    xMin = np.amin(xData) - border
    xMax = np.amax(xData) + border
    yMin = np.amin(yData) - border
    yMax = np.amax(yData) + border

    return xMin, xMax, yMin, yMax

def calculate_cluster_areas(centroids, xMin, xMax, yMin, yMax, step, borders):
    xRange = np.arange(xMin-borders[0], xMax+borders[0], step)
    yRange = np.arange(yMin-borders[1], yMax+borders[1], step)

    Z = np.empty((yRange.shape[0], xRange.shape[0]))

    centroidCount = centroids.shape[0]
    distances = np.empty(centroidCount)
    for i, j in itertools.product(range(yRange.shape[0]), range(xRange.shape[0])):
        p = np.array([xRange[j], yRange[i]])
        for c in range(centroidCount):
            distances[c] = ml_utilities.compute_square_euclidean_distance(p, centroids[c])
        Z[i, j] = np.argmin(distances)

    return Z, xRange, yRange
	
def color_fading(colors,fadingFactor):
    fadedColors=[]
    for c in colors:
        rgbColor=matplotlib.colors.to_rgb(c)
        fadedRgbColor = [fadingFactor * c1 + (1 - fadingFactor) * c2 for (c1, c2) in zip(rgbColor, (1,1,1))]
        fadedColors.append(matplotlib.colors.to_hex(fadedRgbColor))
    
    return fadedColors
	
def decisionboundaries_drawing(numFigure, xRange, yRange, decisionBoundaryMap, colorMap, figsize=None):
    plt.figure(numFigure, figsize=figsize)
    xx, yy = np.meshgrid(xRange, yRange)
    plt.pcolormesh(xx, yy, decisionBoundaryMap, cmap=colorMap, alpha=1)
    plt.draw()
	
def plotting_patterns(numFigure, patterns, patternColor,patternSize=16,marker='o'):
    plt.figure(numFigure)
    plt.scatter(patterns[:, 0], patterns[:, 1],s=patternSize, c = patternColor,marker=marker, edgecolors = '#000000')        
    plt.draw()

def set_view_limits(xMin, xMax, yMin, yMax, borders):
    matplotlib.pyplot.xlim((xMin-borders[0], xMax+borders[0]))
    matplotlib.pyplot.ylim((yMin-borders[1], yMax+borders[1]))
	
def draw_clustering_decision_boundaries(data, labels, centroids, step=0.45, figsize=None, borders=None):
    xMin, xMax, yMin, yMax = calculate_2D_min_max(data, step)
    
    if borders is None:
        borders = ((xMax-xMin)*0.05, (yMax-yMin)*0.05)
    Z, xRange, yRange = calculate_cluster_areas(centroids, xMin, xMax, yMin, yMax, step, borders)

    clusterCount = centroids.shape[0]
    allcolors = ["blue", "burlywood", "cadetblue", "chartreuse", "coral",
                 "cornflowerblue", "crimson", "cyan", "darkblue", "darkcyan", "darkgoldenrod", "darkgreen", "darkkhaki",
                 "darkmagenta", "darkolivegreen",
                 "darkorange", "darkorchid", "darkred", "darksalmon", "darkseagreen", "darkslateblue", "darkslategray",
                 "darkslategrey", "darkturquoise", "deeppink",
                 "deepskyblue", "dodgerblue", "firebrick", "gold", "goldenrod", "green",
                 "greenyellow",
                 "grey",
                 "honeydew",
                 "hotpink",
                 "indianred",
                 "indigo",
                 "ivory",
                 "khaki",
                 "lavender",
                 "lavenderblush",
                 "lawngreen",
                 "lemonchiffon",
                 "lightblue",
                 "lightcoral",
                 "lightcyan",
                 "lightgoldenrodyellow",
                 "lightgray",
                 "lightgreen",
                 "lightgrey",
                 "lightpink",
                 "lightsalmon",
                 "lightseagreen",
                 "lightskyblue",
                 "lightslategray",
                 "lightslategrey",
                 "lightsteelblue",
                 "lightyellow",
                 "lime",
                 "limegreen",
                 "linen",
                 "magenta",
                 "maroon",
                 "mediumaquamarine",
                 "mediumblue",
                 "mediumorchid",
                 "mediumpurple",
                 "mediumseagreen",
                 "mediumslateblue",
                 "mediumspringgreen",
                 "mediumturquoise",
                 "mediumvioletred",
                 "midnightblue",
                 "mintcream",
                 "mistyrose",
                 "moccasin",
                 "navajowhite",
                 "navy",
                 "oldlace",
                 "olive",
                 "olivedrab",
                 "orange",
                 "orangered",
                 "orchid",
                 "palegoldenrod",
                 "palegreen",
                 "paleturquoise",
                 "palevioletred",
                 "papayawhip",
                 "peachpuff",
                 "peru",
                 "pink",
                 "plum",
                 "powderblue",
                 "purple",
                 "rebeccapurple",
                 "red",
                 "rosybrown",
                 "royalblue",
                 "saddlebrown",
                 "salmon",
                 "sandybrown",
                 "seagreen",
                 "seashell",
                 "sienna",
                 "silver",
                 "skyblue",
                 "slateblue",
                 "slategray",
                 "slategrey",
                 "snow",
                 "springgreen",
                 "steelblue",
                 "tan",
                 "teal",
                 "thistle",
                 "tomato",
                 "turquoise",
                 "violet",
                 "wheat",
                 "yellow",
                 "yellowgreen"]
    patternColors = allcolors[:clusterCount]
    fadedColors = color_fading(patternColors, 0.7)
    colorMap = matplotlib.colors.ListedColormap(fadedColors)

    decisionboundaries_drawing(0, xRange, yRange, Z, colorMap, figsize=figsize)

    plotting_patterns(0, centroids, 'black', 96, 'x');

    separatedClassData = ml_utilities.separate_pattern_classes(data, labels, clusterCount)
    for i in range(clusterCount):
        if (separatedClassData[i].shape[0] > 0):
            plotting_patterns(0, separatedClassData[i], patternColors[i])
    set_view_limits(xMin, xMax, yMin, yMax, borders)

def calculate_2D_decision_boundary_map(classifier, xMin, xMax, yMin, yMax, step):
    xRange = np.arange(xMin, xMax, step)
    yRange = np.arange(yMin, yMax, step)

    decisionBoundaryMap = np.empty((yRange.shape[0], xRange.shape[0]), dtype=int)

    for i, j in itertools.product(range(yRange.shape[0]), range(xRange.shape[0])):
        p = np.array([[xRange[j], yRange[i]]])
        decisionBoundaryMap[i, j] = int(classifier.predict(p)[0])

    return decisionBoundaryMap, xRange, yRange

def draw_classification_decision_boundaries(data, labels, num_class, num_figure, num_rows, num_columns, num_sub_figure,
											  x_range, y_range, z, decision_boundary_colors, pattern_colors, title,
											  legend_colors, legend_labels):
    subplotting_decisionboundaries_drawing(num_figure, num_rows, num_columns, num_sub_figure,
                                                  x_range, y_range, z, decision_boundary_colors)
    subplot_legend_drawing(num_figure, num_rows, num_columns, num_sub_figure, legend_colors, legend_labels, 3)
    separate_class_data = ml_utilities.separate_pattern_classes(data, labels, num_class)
    for i in range(num_class):
        subplotting_patterns(num_figure, num_rows, num_columns, num_sub_figure,
                                    separate_class_data[i], pattern_colors[i])

    subplotting_title(num_figure, num_rows, num_columns, num_sub_figure, title)

def subplotting_title(numFigure, numRows, numColums, numSubFigure, title):
    fg = plt.figure(numFigure)
    subplot_fig = fg._axstack.as_list()[numSubFigure-1]

    subplot_fig.set_title(title)
    plt.draw()
	
def subplotting_patterns(numFigure, numRows, numColums, numSubFigure, patterns, patternColor):
    fg = plt.figure(numFigure)
    subplot_fig = fg._axstack.as_list()[numSubFigure-1]

    plt.sca(subplot_fig)
    plotting_patterns(numFigure, patterns, patternColor)
	
def subplotting_decisionboundaries_drawing(numFigure, numRows, numColums, numSubFigure, xRange, yRange,
                                           decisionBoundaryMap, colorMap):
    fg = plt.figure(numFigure)
    subplot_fig = fg._axstack.as_list()[numSubFigure-1]

    plt.sca(subplot_fig)
    decisionboundaries_drawing(numFigure, xRange, yRange, decisionBoundaryMap, colorMap)

def subplot_legend_drawing(numFigure, numRows, numColums, numSubFigure, colors, labels, loc):
    fg = plt.figure(numFigure)
    subplot_fig = fg._axstack.as_list()[numSubFigure - 1]

    plt.sca(subplot_fig)
    legend_drawing(numFigure, colors, labels, loc)
	
def show_2D_results(classifier, *args, figsize=(18, 8)):
    feature_count = args[0][0].shape[1]
    # Solo per il training set con 2 features
    if feature_count == 2:
        # Disegno dei pattern e delle aree di probabilità
        all_patterns_coordinates = np.concatenate([args[i][0] for i in range(len(args))])

        x_min, x_max, y_min, y_max = calculate_2D_min_max(all_patterns_coordinates, 10)
        decision_boundary_map, x_range, y_range = calculate_2D_decision_boundary_map(classifier,
																					x_min, x_max,
																					y_min, y_max, 2)

        #plt.figure(num=1, figsize=(18, 8), dpi=96)
        _, axes = plt.subplots(nrows=1, ncols=len(args), figsize=figsize, dpi=96, squeeze=False) # TODO: righe/colonne


        colors = ["red", "coral", "gold", "yellowgreen", "green", "mediumaquamarine",
                  "mediumturquoise", "cornflowerblue", "blue", "purple"]
        faded_colors = color_fading(colors, 0.7)
        color_map = matplotlib.colors.ListedColormap(faded_colors)

        legend_labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

        for i in range(len(args)):
            draw_classification_decision_boundaries(args[i][0], args[i][1], 10, 1, 1, len(args), i+1, x_range, y_range,
													  decision_boundary_map, color_map, colors, 
													  args[i][2], faded_colors, legend_labels)
        plt.show()
    else:
        print('show_results può visualizzare solamente pattern di bidimensionali')