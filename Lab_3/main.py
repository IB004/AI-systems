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
        '  N  ',
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


def prepare_matrixes(X, Y):
    A = X.to_numpy()
    A = np.insert(A, 0, 1, axis=1)  # add ones vector
    Y = Y.to_numpy()
    return A, Y


def count_r(X, Y, W):
    A, Y = prepare_matrixes(X, Y)
    mean_y = sum(Y) / len(Y)
    SSE = sum(map(lambda x: x ** 2, (np.dot(A, W) - Y)))
    SST = sum(map(lambda y: (y - mean_y) ** 2, Y))
    R = 1 - SSE / SST
    return R


def linear_regression(X, Y):
    A, Y = prepare_matrixes(X, Y)
    W = np.dot(np.transpose(A), A)
    if isinstance(W, float):
        W = [[W]]
    W = np.linalg.inv(W).dot(np.transpose(A)).dot(Y)
    return W


def print_models_table(models):
    print(
        'Name'.ljust(32),
        'R_learn'.rjust(8),
        'R_test'.rjust(8),
        'w_0'.rjust(6),
        'w_1'.rjust(6),
        'w_2'.rjust(6),
        'w_3'.rjust(6),
        'w_4'.rjust(6),
        'w_5'.rjust(6),

    )
    for key, model in models.items():
        print(
            model.name.ljust(32),
            ft_rjust(model.r_learn),
            ft_rjust(model.r_test),
            (' ').join(map(lambda el: ft(el), model.w.tolist())),
        )
    print()


df = pd.read_csv("./Student_Performance.csv")
df = df.dropna()
df = df.replace({'Extracurricular Activities': {'Yes': 1, 'No': 0}})
print('\n' * 8)

print_stats_table(df)

dfm = df.copy()
# apply normalization techniques
for column in df.columns:
    dfm[column] = (dfm[column] - dfm[column].min()) / (dfm[column].max() - dfm[column].min())

print_stats_table(dfm)


n = len(dfm.index)
learn_rows = int(0.8 * n)
Y_learn = dfm.iloc[:learn_rows, -1]
Y_test = dfm.iloc[learn_rows:, -1]
X_learn = dfm.iloc[:learn_rows, :-1]
X_test = dfm.iloc[learn_rows:, :-1]


class Model:
    def __init__(self, name, x_learn, x_test, y_learn, y_test):
        self.name = name
        self.x_learn = x_learn
        self.x_test = x_test
        self.y_learn = y_learn
        self.y_test = y_test
        self.w = linear_regression(x_learn, y_learn)
        self.r_learn = count_r(x_learn, y_learn, self.w)
        self.r_test = count_r(x_test, y_test, self.w)


models = {
    'Hours Studied': Model('Hours Studied', X_learn.iloc[:, 0:1], X_test.iloc[:, 0:1], Y_learn, Y_test),
    'Previous Scores': Model('Previous Scores', X_learn.iloc[:, 1:2], X_test.iloc[:, 1:2], Y_learn, Y_test),
    'Extracurricular Activities': Model('Extracurricular Activities', X_learn.iloc[:, 2:3], X_test.iloc[:, 2:3], Y_learn, Y_test),
    'Sleep Hours': Model('Sleep Hours', X_learn.iloc[:, 3:4], X_test.iloc[:, 3:4], Y_learn, Y_test),
    'Sample Question Papers Practiced': Model('Sample Question Papers Practiced', X_learn.iloc[:, 4:5], X_test.iloc[:, 4:5], Y_learn, Y_test),
    'H.St. + Pr.Sc.': Model('H.St. + Pr.Sc.', X_learn.iloc[:, 0:2], X_test.iloc[:, 0:2], Y_learn, Y_test),
    'All': Model('All', X_learn.iloc[:, :], X_test.iloc[:, :], Y_learn, Y_test),
}

print_models_table(models)


def show_graphic(model):
    ax.clear()
    ax.grid()

    xs = model.x_learn
    ys = model.y_learn
    ax.plot(xs, ys, 'o', color='blue', alpha = 0.8)

    xs = model.x_test
    ys = model.y_test
    ax.plot(xs, ys, 'o', color='lime', alpha = 0.2)

    b = model.w[0]
    k = model.w[1]

    x = np.arange(0, 1.01, 0.01)
    ax.plot(x, b + k * x, color='red')

    ax.set_title(model.name)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, f'w_0: {ft(b)}\nw_1: {ft(k)}\nR_learn: {ft(model.r_learn)}\nR_test: {ft(model.r_test)}', transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)

    plt.draw()


fig, ax = plt.subplots()
show_graphic(models['Hours Studied'])


axes_button = plt.axes([0.01, 0.85, 0.08, 0.075])
button_hs = Button(axes_button, 'Hours Studied')
button_hs.on_clicked(lambda e: show_graphic(models['Hours Studied']))

axes_button = plt.axes([0.01, 0.65, 0.08, 0.075])
button_ps = Button(axes_button, 'Previous Scores')
button_ps.on_clicked(lambda e: show_graphic(models['Previous Scores']))

axes_button = plt.axes([0.01, 0.45, 0.08, 0.075])
button_ea = Button(axes_button, 'Extra Activities')
button_ea.on_clicked(lambda e: show_graphic(models['Extracurricular Activities']))

axes_button = plt.axes([0.01, 0.25, 0.08, 0.075])
button_sh = Button(axes_button, 'Sleep Hours')
button_sh.on_clicked(lambda e: show_graphic(models['Sleep Hours']))

axes_button = plt.axes([0.01, 0.05, 0.08, 0.075])
button_pp = Button(axes_button, 'Papers Practiced')
button_pp.on_clicked(lambda e: show_graphic(models['Sample Question Papers Practiced']))

plt.show()

