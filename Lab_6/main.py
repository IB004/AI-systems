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


def print_stats_table(df, title=""):
    print(f"================== {title} ==================")
    print(
        'Name'.ljust(32),
        '  N  ',
        'Min'.rjust(6), ' -', ' Max'.ljust(6),
        'Mean'.rjust(10),
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


def replace_with_median(df, column_name, min_value):
    replace = df[column_name].median()
    if df[column_name].dtype == np.int64:
        replace = int(replace)
    df.loc[df[column_name] < min_value, column_name] = replace


df = pd.read_csv("./diabetes.csv")
print_stats_table(df, "Initial")

df = df.dropna()
replace_with_median(df, "Glucose", 1)
replace_with_median(df, "BloodPressure", 1)
replace_with_median(df, "SkinThickness", 1)
replace_with_median(df, "Insulin", 1)
replace_with_median(df, "BMI", 1)
replace_with_median(df, "Age", 1)
print_stats_table(df, "Replace zeroes with means")

dfm = df.copy()
# apply normalization techniques
for column in df.columns[:-1]:
    n = len(dfm[column])
    mean = sum(dfm[column]) / n
    std_deviation = sqrt(sum(map(lambda x: (x - mean) ** 2, dfm[column])) / (n - 1))
    dfm[column] = (dfm[column] - mean) / std_deviation
print_stats_table(dfm, "After normalization")









fig, ax = plt.subplots()


def draw_graphic(name):
    ax.clear()
    df[name].hist(
        ax=ax,
        bins=20,
        grid=False,
        edgecolor='black',
        color='orange',
        rwidth=0.8
    )
    ax.set_title(name)
    plt.draw()


axes_button = plt.axes([0.01, 0.8, 0.08, 0.075])
button_1 = Button(axes_button, 'Pregnancies')
button_1.on_clicked(lambda e: draw_graphic('Pregnancies'))

axes_button = plt.axes([0.01, 0.7, 0.08, 0.075])
button_2 = Button(axes_button, 'Glucose')
button_2.on_clicked(lambda e: draw_graphic('Glucose'))

axes_button = plt.axes([0.01, 0.6, 0.08, 0.075])
button_3 = Button(axes_button, 'BloodPressure')
button_3.on_clicked(lambda e: draw_graphic('BloodPressure'))

axes_button = plt.axes([0.01, 0.5, 0.08, 0.075])
button_4 = Button(axes_button, 'SkinThickness')
button_4.on_clicked(lambda e: draw_graphic('SkinThickness'))

axes_button = plt.axes([0.01, 0.4, 0.08, 0.075])
button_5 = Button(axes_button, 'Insulin')
button_5.on_clicked(lambda e: draw_graphic('Insulin'))

axes_button = plt.axes([0.01, 0.3, 0.08, 0.075])
button_6 = Button(axes_button, 'BMI')
button_6.on_clicked(lambda e: draw_graphic('BMI'))

axes_button = plt.axes([0.01, 0.2, 0.08, 0.075])
button_7 = Button(axes_button, 'DiabetesPedigree')
button_7.on_clicked(lambda e: draw_graphic('DiabetesPedigreeFunction'))

axes_button = plt.axes([0.01, 0.1, 0.08, 0.075])
button_8 = Button(axes_button, 'Age')
button_8.on_clicked(lambda e: draw_graphic('Age'))


draw_graphic("Glucose")
plt.show()








def gradient(x, y, y_pred, n, learning_rate):
    return learning_rate * np.dot(x.T, (y_pred - y)) / n


def newton(x, y, y_pred, n, learning_rate):
    grad = gradient(x, y, y_pred, n, 1)
    y_pred = y_pred[np.newaxis]
    d = np.dot(y_pred.T, (1 - y_pred))
    d = np.diag(np.diag(d))
    a = np.dot(x.T, d)
    hessian = - np.dot(a, x)
    if np.linalg.det(hessian) != 0:
        b = np.linalg.inv(hessian)
    else:
        b = np.zeros((hessian.shape[0], hessian.shape[0]))
    return - np.dot(b, grad)


class LogReg:

    def __init__(self):
        self.thetas = None
        self.loss_history = []

    def train(self, x, y, iter=1000, learning_rate=0.01, optimiz=gradient):
        x, y = x.copy(), y.copy()
        self.add_ones(x)

        thetas = np.zeros(x.shape[1])
        n = x.shape[0]

        for i in range(iter):
            y_pred = self.h(x, thetas)
            opt = optimiz(x, y, y_pred, n, learning_rate)
            thetas -= opt
            self.thetas = thetas

            loss = self.log_loss(y, y_pred)
            if i % (iter // 20) == 0:
                self.loss_history.append(loss)

    def test(self, x, th=0.5):
        x = x.copy()
        self.add_ones(x)
        z = np.dot(x, self.thetas)
        probs = np.array([self.sigmoid(value) for value in z])
        return np.where(probs >= th, 1, 0), probs

    # util functions
    def add_ones(self, x):
        return x.insert(0, 'x0', np.ones(x.shape[0]))

    def h(self, x, thetas):
        z = np.dot(x, thetas)
        return np.array([self.sigmoid(value) for value in z])

    def log_loss(self, y, y_pred):
        y_one_loss = y * np.log(y_pred + 1e-9)
        y_zero_loss = (1 - y) * np.log(1 - y_pred + 1e-9)
        return -np.mean(y_zero_loss + y_one_loss)

    def sigmoid(self, z):
        if z >= 0:
            return 1 / (1 + np.exp(-z))
        else:
            return np.exp(z) / (np.exp(z) + 1)


def error_matrix(res, y_test):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for r, y in zip(res, y_test):
        if r == y:
            if r == 1:
                tp += 1
            else:
                tn += 1
        else:
            if r == 1:
                fp += 1
            else:
                fn += 1
    return tp, fp, fn, tn


def count_metrics(res, y_test):
    tp, fp, fn, tn = error_matrix(res, y_test)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return accuracy, precision, recall, f1


n = len(dfm.index)
learn_rows = int(0.8 * n)
Y_learn = dfm.iloc[:learn_rows, -1]
Y_test = dfm.iloc[learn_rows:, -1]
X_learn = dfm.iloc[:learn_rows, :-1]
X_test = dfm.iloc[learn_rows:, :-1]


def print_exps_header():
    header = ""
    header += "iters".rjust(5)
    header += "l_rate".rjust(8)
    header += "optim".rjust(10)
    header += "accur".rjust(8)
    header += "prec".rjust(8)
    header += "rec".rjust(8)
    header += "f1".rjust(8)
    print(header)


def print_exp_row(iters, rate, opt, accuracy, precision, recall, f1):
    print(f"{str(iters).rjust(5)}{ft_rjust(rate)}{'  gradient' if opt == gradient else '    newton'}{ft_rjust(accuracy)}{ft_rjust(precision)}{ft_rjust(recall)}{ft_rjust(f1)}")


exps = [
    [LogReg(), 300, 0.1, gradient],
    [LogReg(), 300, 0.1, newton],
    [LogReg(), 3000, 0.1, gradient],
    [LogReg(), 3000, 0.1, newton],
    [LogReg(), 3000, 0.001, gradient],
    [LogReg(), 3000, 0.001, newton],
]

print_exps_header()

for exp in exps:
    log_reg = exp[0]
    iters = exp[1]
    rate = exp[2]
    opt = exp[3]
    log_reg.train(X_learn, Y_learn, iter=iters, learning_rate=rate, optimiz=opt)
    res, probs = log_reg.test(X_test)
    accuracy, precision, recall, f1 = count_metrics(res, Y_test)
    print_exp_row(iters, rate, opt, accuracy, precision, recall, f1)
