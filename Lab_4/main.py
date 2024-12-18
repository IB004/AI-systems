import random
from math import sqrt

from matplotlib.widgets import Button
from scipy import stats

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def ft(a, format=3):
    format_str = "{: 4." + str(format) + "f}"
    return format_str.format(a)


def ft_rjust(a, just=8, format=3):
    return ft(a, format).rjust(just)


def basic_stats(X):
    n = len(X)
    min_x = min(X)
    max_x = max(X)
    mean = sum(X) / n
    std_deviation = sqrt( sum(map(lambda x: (x - mean)**2, X)) / (n - 1) )
    return n, min_x, max_x, mean, std_deviation


def conf_interval(n, mean, std_dev, conf_level):
    (a, b) = stats.t.interval(conf_level, n - 1, mean, std_dev)
    return (b - a) / 2


def print_stats_table(df):
    print(
        'Name'.ljust(32),
        ' N ',
        'Min'.rjust(8), ' -', ' Max'.ljust(8),
        'Mean'.rjust(8),
        'St_dev'.rjust(8),
        '90%'.rjust(8),
        '95%'.rjust(8),
        '99%'.rjust(8),
    )
    for column in df:
        n, min_x, max_x, mean, std_dev = basic_stats(df[column])
        delta_90 = conf_interval(n, mean, std_dev, 0.90)
        delta_95 = conf_interval(n, mean, std_dev, 0.95)
        delta_99 = conf_interval(n, mean, std_dev, 0.99)
        print(
            column.ljust(32),
            n,
            ft_rjust(min_x), ' -', ft(max_x).ljust(8),
            ft_rjust(mean),
            ft_rjust(std_dev),
            ft_rjust(delta_90),
            ft_rjust(delta_95),
            ft_rjust(delta_99)
        )
    print()


def euclidean_distances(X_learn, x_test, S):
    return np.sqrt(np.sum(S * (X_learn - x_test) ** 2, axis=1))


def knn_classify(X_learn, Y_learn, x_test, k, S):
    distances = euclidean_distances(X_learn, x_test, S)
    k_nearest_indexes = np.argsort(distances)[:k]
    nearest = Y_learn[k_nearest_indexes]
    return np.mean(nearest).round()


def error_matrix(Res, Y_test):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for r, y in zip(Res, Y_test):
        if r == y:
            if y == 1:
                tp += 1
            else:
                tn += 1
        else:
            if y == 1:
                fp += 1
            else:
                fn += 1
    return tp, fp, fn, tn


def acc_params(tp, fp, fn, tn):
    acc = (tp + tn) / (tp + fp + fn + tn)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)

    return acc, ppv, npv


def s_to_columns(S):
    lst = []
    for i in range(len(S)):
        if S[i] > 0:
            lst.append(i)
    return lst


def print_model_row(model):
    columns = s_to_columns(model.S)
    columns_names = [df.columns[i] for i in columns]
    print(
        model.name.ljust(32),
        str(model.k).rjust(4),
        str(model.tp).rjust(4),
        str(model.fp).rjust(4),
        str(model.fn).rjust(4),
        str(model.tn).rjust(4),
        ft_rjust(model.acc, just=6),
        ft_rjust(model.ppv, just=6),
        ft_rjust(model.npv, just=6),
        ' '.rjust(6),
        '  '.join(columns_names)
    )


def print_models_table(models):
    print(
        'Name'.ljust(32),
        'k'.rjust(4),
        'TP'.rjust(4),
        'FP'.rjust(4),
        'FN'.rjust(4),
        'TN'.rjust(4),
        'ACC'.rjust(6),
        'PPV'.rjust(6),
        'NPV'.rjust(6),
        ' '.rjust(6),
        'Columns'.ljust(8),
    )

    for key, model in models.items():
        print_model_row(model)


df = pd.read_csv("./diabetes.csv")
df = df.dropna()

print_stats_table(df)

dfm = df.copy()
# apply normalization techniques
for column in df.columns[:-1]:
    n = len(dfm[column])
    mean = sum(dfm[column]) / n
    std_deviation = sqrt(sum(map(lambda x: (x - mean) ** 2, dfm[column])) / (n - 1))
    dfm[column] = (dfm[column] - mean) / std_deviation

print_stats_table(dfm)


n = len(dfm.index)
learn_rows = int(0.8 * n)
Y_learn = dfm.iloc[:learn_rows, -1]
Y_test = dfm.iloc[learn_rows:, -1]
X_learn = dfm.iloc[:learn_rows, :-1]
X_test = dfm.iloc[learn_rows:, :-1]


class Model:
    def __init__(self, name, k, S, x_learn, x_test, y_learn, y_test):
        self.name = name
        self.k = k
        self.S = S
        self.x_learn = x_learn.to_numpy()
        self.x_test = x_test.to_numpy()
        self.y_learn = y_learn.to_numpy()
        self.y_test = y_test.to_numpy()
        self.res = np.array([knn_classify(self.x_learn, self.y_learn, tst, self.k, S) for tst in self.x_test])
        self.tp, self.fp, self.fn, self.tn = error_matrix(self.res, self.y_test)
        self.acc, self.ppv, self.npv = acc_params(self.tp, self.fp, self.fn, self.tn)


models = {
    'GlucoseBMIAge3': Model('Glucose BMI Age', 3, [0, 1, 0, 0, 0, 1, 0, 1], X_learn, X_test, Y_learn, Y_test),
    'GlucoseBMIAge5': Model('Glucose BMI Age', 5, [0, 1, 0, 0, 0, 1, 0, 1], X_learn, X_test, Y_learn, Y_test),
    'GlucoseBMIAge7': Model('Glucose BMI Age', 7, [0, 1, 0, 0, 0, 1, 0, 1], X_learn, X_test, Y_learn, Y_test),
    'GlucoseBMIAge9': Model('Glucose BMI Age', 9, [0, 1, 0, 0, 0, 1, 0, 1], X_learn, X_test, Y_learn, Y_test),
    'GlucoseBMIAge13': Model('Glucose BMI Age', 13, [0, 1, 0, 0, 0, 1, 0, 1], X_learn, X_test, Y_learn, Y_test),
    'All3': Model('All', 3, [1, 1, 1, 1, 1, 1, 1, 1], X_learn, X_test, Y_learn, Y_test),
    'All5': Model('All', 5, [1, 1, 1, 1, 1, 1, 1, 1], X_learn, X_test, Y_learn, Y_test),
    'All7': Model('All', 7, [1, 1, 1, 1, 1, 1, 1, 1], X_learn, X_test, Y_learn, Y_test),
    'All9': Model('All', 9, [1, 1, 1, 1, 1, 1, 1, 1], X_learn, X_test, Y_learn, Y_test),
}

print_models_table(models)


def show_graphic(model):
    ax.clear()
    ax.grid()

    columns = s_to_columns(model.S)[:3]

    xs = [model.x_learn[:, columns[0:1]][i] for i in range(len(model.y_learn)) if model.y_learn[i] == 0]
    ys = [model.x_learn[:, columns[1:2]][i] for i in range(len(model.y_learn)) if model.y_learn[i] == 0]
    zs = [model.x_learn[:, columns[2:3]][i] for i in range(len(model.y_learn)) if model.y_learn[i] == 0]
    ax.scatter(xs, ys, zs, c=zs, cmap='cool')

    xs = [model.x_learn[:, columns[0:1]][i] for i in range(len(model.y_learn)) if model.y_learn[i] == 1]
    ys = [model.x_learn[:, columns[1:2]][i] for i in range(len(model.y_learn)) if model.y_learn[i] == 1]
    zs = [model.x_learn[:, columns[2:3]][i] for i in range(len(model.y_learn)) if model.y_learn[i] == 1]
    ax.scatter(xs, ys, zs, c=zs, cmap='spring')

    xs = [model.x_test[:, columns[0:1]][i] for i in range(len(model.res)) if model.res[i] == 0 and model.y_test[i] == 0]
    ys = [model.x_test[:, columns[1:2]][i] for i in range(len(model.res)) if model.res[i] == 0 and model.y_test[i] == 0]
    zs = [model.x_test[:, columns[2:3]][i] for i in range(len(model.res)) if model.res[i] == 0 and model.y_test[i] == 0]
    ax.scatter(xs, ys, zs, color='blue', marker='d')

    xs = [model.x_test[:, columns[0:1]][i] for i in range(len(model.res)) if model.res[i] == 0 and model.y_test[i] == 1]
    ys = [model.x_test[:, columns[1:2]][i] for i in range(len(model.res)) if model.res[i] == 0 and model.y_test[i] == 1]
    zs = [model.x_test[:, columns[2:3]][i] for i in range(len(model.res)) if model.res[i] == 0 and model.y_test[i] == 1]
    ax.scatter(xs, ys, zs, color='blue', marker='x')

    xs = [model.x_test[:, columns[0:1]][i] for i in range(len(model.res)) if model.res[i] == 1 and model.y_test[i] == 1]
    ys = [model.x_test[:, columns[1:2]][i] for i in range(len(model.res)) if model.res[i] == 1 and model.y_test[i] == 1]
    zs = [model.x_test[:, columns[2:3]][i] for i in range(len(model.res)) if model.res[i] == 1 and model.y_test[i] == 1]
    ax.scatter(xs, ys, zs, color='red', marker='d')

    xs = [model.x_test[:, columns[0:1]][i] for i in range(len(model.res)) if model.res[i] == 1 and model.y_test[i] == 0]
    ys = [model.x_test[:, columns[1:2]][i] for i in range(len(model.res)) if model.res[i] == 1 and model.y_test[i] == 0]
    zs = [model.x_test[:, columns[2:3]][i] for i in range(len(model.res)) if model.res[i] == 1 and model.y_test[i] == 0]
    ax.scatter(xs, ys, zs, color='red', marker='x')

    ax.set_xlabel(dfm.columns[columns[0]])
    ax.set_ylabel(dfm.columns[columns[1]])
    ax.set_zlabel(dfm.columns[columns[2]])

    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_zlim(-2.2, 2.2)


    ax.set_title(model.name)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    text = f'''
    K: {model.k}
    
    TP: {model.tp}   FP: {model.fp}
    FN: {model.fn}   TN: {model.tn}
    
    ACC: {ft(model.acc)} PPV: {ft(model.ppv)} NPV: {ft(model.npv)}   
    '''
    ax.text2D(1.05, 0.95, text, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)

    plt.draw()


def show_random():
    columns = [1, 1, 1, 0, 0, 0, 0, 0]
    random.shuffle(columns)
    ks = [1, 3, 5, 7, 9, 13, 15]
    random.shuffle(ks)
    model = Model('Random', ks[0], columns, X_learn, X_test, Y_learn, Y_test)

    print_model_row(model)

    show_graphic(model)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

show_graphic(models['GlucoseBMIAge3'])


axes_button = plt.axes([0.02, 0.85, 0.1, 0.075])
button_3 = Button(axes_button, 'Glucose BMI Age (3)')
button_3.on_clicked(lambda e: show_graphic(models['GlucoseBMIAge3']))

axes_button = plt.axes([0.02, 0.70, 0.1, 0.075])
button_5 = Button(axes_button, 'Glucose BMI Age (5)')
button_5.on_clicked(lambda e: show_graphic(models['GlucoseBMIAge5']))

axes_button = plt.axes([0.02, 0.55, 0.1, 0.075])
button_7 = Button(axes_button, 'Glucose BMI Age (7)')
button_7.on_clicked(lambda e: show_graphic(models['GlucoseBMIAge7']))

axes_button = plt.axes([0.02, 0.40, 0.1, 0.075])
button_9 = Button(axes_button, 'Glucose BMI Age (9)')
button_9.on_clicked(lambda e: show_graphic(models['GlucoseBMIAge9']))

axes_button = plt.axes([0.02, 0.25, 0.1, 0.075])
button_13 = Button(axes_button, 'Glucose BMI Age (13)')
button_13.on_clicked(lambda e: show_graphic(models['GlucoseBMIAge13']))

axes_button = plt.axes([0.02, 0.10, 0.1, 0.075])
button_rd = Button(axes_button, 'Random')
button_rd.on_clicked(lambda e: show_random())

plt.show()

