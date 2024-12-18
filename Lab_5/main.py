import math
import random
from math import sqrt

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


def error_matrix(res, test, threshold):
    if threshold == 1:
        res_class = [0 for r in res]
    else:
        res_class = [1 if r >= threshold else 0 for r in res]

    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for r, t in zip(res_class, test):
        if r == t:
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


def acc_params(tp, fp, fn, tn):
    acc = (tp + tn) / (tp + fp + fn + tn)

    if tp == (tp + fp):
        pres = 1
    else:
        pres = tp / (tp + fp)

    if tp == (tp + fn):
        recall = 1
    else:
        recall = tp / (tp + fn)

    return acc, pres, recall


def count_tpr(tp, fp, fn, tn):
    return tp / (tp + fn)


def count_fpr(tp, fp, fn, tn):
    return fp / (fp + tn)


def freq(c, target):
    frq = 0
    for t in target:
        if t == c:
            frq += 1
    return frq


def info(target):
    inf = 0
    n = len(target)
    for c in target.unique():
        frq = freq(c, target)
        inf += frq / n * math.log2(frq / n)
    return -1 * inf


def with_feature_value(feature, value, target):
    target_v = []
    for f, t in zip(feature, target):
        if f == value:
            target_v.append(t)
    return pd.DataFrame(target_v)


def info_feature(feature, target):
    inf_f = 0
    split = 0
    for v in feature.unique():
        target_v = with_feature_value(feature, v, target).iloc[:, 0]
        n_i = len(target_v)
        n = len(target)
        inf_f += n_i / n * info(target_v)
        split += n_i / n * math.log2(n_i / n)
    return inf_f, -1 * split


def get_gain_ratios(features, target):
    gain_ratio = []
    inf = info(target)
    for i in feature_set:
        feature = features.iloc[:, i]  # column
        inf_f, split = info_feature(feature, target)
        if split == 0:
            gain_ratio.append(0)
        else:
            gain_ratio.append((inf - inf_f) / split)
    return gain_ratio


def max_ind(lst):
    ind = 0
    mx = lst[0]
    for i, el in zip(range(len(lst)), lst):
        if el > mx:
            mx = el
            ind = i
    return ind, mx

names = [
    "1 Student Age",
    "2 Sex",
    "3 Graduated high-school type",
    "4 Scholarship type",
    "5 Additional work",
    "6 Regular artistic or sports activity",
    "7 Do you have a partner",
    "8 Total salary if available",
    "9 Transportation to the university",
    "10 Accomodation type in Cyprus",
    "11 Mother's education",
    "12 Father's education",
    "13 Number of sisters/brothers (if available)",
    "14 Parental status",
    "15 Mother's occupation",
    "16 Father's occupation",
    "17 Weekly study hours",
    "18 Reading frequency (non-scientific books/journals)",
    "19 Reading frequency (scientific books/journals)",
    "20 Attendance to the seminars/conferences related to the department",
    "21 Impact of your projects/activities on your success",
    "22 Attendance to classes",
    "23 Preparation to midterm exams 1",
    "24 Preparation to midterm exams 2",
    "25 Taking notes in classes",
    "26 Listening in classes",
    "27 Discussion improves my interest and success in the course",
    "28 Flip-classroom",
    "29 Cumulative grade point average in the last semester (/4.00)",
    "30 Expected Cumulative grade point average in the graduation (/4.00)",
    "COURSE ID"
]


class DecisionTree:
    def __init__(self, fts, tgs):
        self.fts = fts
        self.tgs = tgs
        self.nodes = []
        self.n = len(self.tgs)
        self.name = ""
        self.feature_ind = -1
        self.values = []
        self.pos_rate = len(self.tgs[self.tgs == 1]) / self.n

    def build(self, depth=0):
        if info(self.tgs) == 0:
            return
        gain_ratios = get_gain_ratios(self.fts, self.tgs)
        ind, max_gain = max_ind(gain_ratios)
        if max_gain == 0:
            return
        self.feature_ind = feature_set[ind]
        self.name = names[self.feature_ind]

        feature = self.fts.iloc[:, self.feature_ind]  # column
        self.values = feature.unique()
        self.values.sort()
        for v in self.values:
            fts = self.fts[self.fts.iloc[:, self.feature_ind] == v]
            tgs = with_feature_value(feature, v, self.tgs).iloc[:, 0]
            self.nodes.append(DecisionTree(fts, tgs))
        for node in self.nodes:
            node.build(depth=depth+1)

    def combine(self, min_count):
        if len(self.nodes) < 2:
            return

        ns = [node.n for node in self.nodes]
        ns.sort()
        n1 = ns[0]
        n1_node: DecisionTree = [node for node in self.nodes if node.n == n1][0]
        n2 = ns[1]
        n2_node: DecisionTree = [node for node in self.nodes if node.n == n2][0]
        if n1 < min_count:
            n2_node.fts = pd.concat([n1_node.fts, n2_node.fts], ignore_index=True)
            n2_node.tgs = pd.concat([n1_node.tgs, n2_node.tgs], ignore_index=True)
            n2_node.n += n1_node.n
            n2_node.name = ""
            n2_node.feature_ind = -1
            n2_node.values = []
            n2_node.pos_rate = len(n2_node.tgs[n2_node.tgs == 1]) / n2_node.n

            self.nodes.remove(n1_node)
            self.combine(min_count)

        for node in self.nodes:
            node.combine(min_count)

    def test(self, feature_list):
        if self.feature_ind == -1:
            return self.pos_rate
        for node in self.nodes:
            if node.feature_ind > -1 and node.fts.iloc[0, self.feature_ind] == feature_list[self.feature_ind]:
                return node.test(feature_list)
        return self.pos_rate

    def print(self, nest_level=0):
        print()
        print("\t" * nest_level, f"name: {self.name}")
        print("\t" * nest_level, f"n: {self.n}")
        if self.feature_ind == -1:
            print("\t" * nest_level, f"pos_rate: {ft(self.pos_rate)}")
            return
        for node in self.nodes:
            node.print(nest_level = nest_level + 1)


df = pd.read_csv("./data.csv")
df = df.dropna()
# df = df.sample(frac=1)
df['GOOD GRADE'] = [1 if grade >= 5 else 0 for grade in df['GRADE']]


feature_set = list(range(0, 31))
feature_count = math.ceil(sqrt(len(feature_set)))
random.shuffle(feature_set)
feature_set = feature_set[:feature_count]


# feature_set = [e - 1 for e in [26, 2, 29, 7, 20, 4]]
# feature_count = len(feature_set)

n = len(df.index)
train_rows = int(0.8 * n)

features_train = df.iloc[:train_rows, 1:32]
targets_train = df.iloc[:train_rows, -1]

features_test = df.iloc[train_rows:, 1:32]
targets_test = df.iloc[train_rows:, -1]

dtree = DecisionTree(features_train, targets_train)
dtree.build()
# dtree.combine(4)
dtree.print()

results = []
tests = []
for i in range(len(features_test.index)):
    feature_list = list(features_test.iloc[i])  # row
    res = dtree.test(feature_list)
    test = list(targets_test)[i]

    results.append(res)
    tests.append(test)

    print(f"{ft_rjust(res)}  |  {test}")


tpr = []
fpr = []
prs = []
rec = []
thresholds = np.linspace(0, 1, 100)
for th in thresholds:
    tp, fp, fn, tn = error_matrix(results, tests, th)
    tpr.append(count_tpr(tp, fp, fn, tn))
    fpr.append(count_fpr(tp, fp, fn, tn))
    acc, pres, recall = acc_params(tp, fp, fn, tn)
    prs.append(pres)
    rec.append(recall)


fig = plt.figure()
ax = fig.add_subplot()


def add_graphic(xs, ys, title="Title", x_label="x", y_label="y", area=0):
    plt.grid()

    plt.plot(xs, ys, "-o")

    plt.title(title)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    text = \
f'''
Area: {ft(area)}
'''
    plt.text(0.8, 0.6, text, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)

    plt.xlabel(x_label)
    plt.ylabel(y_label)


def count_area(xs, ys):
    xs, ys = (list(t) for t in zip(*sorted(zip(xs, ys))))
    area = 0
    prev_x = xs[0]
    prev_y = ys[0]
    for x, y in zip(xs[1:], ys[1:]):
        h = x - prev_x
        area += (prev_y + y) * 0.5 * h
        prev_x = x
        prev_y = y
    return area


print("Выбранные признаки: \n\t" + '\n\t'.join([names[i] for i in feature_set]))

tp, fp, fn, tn = error_matrix(results, tests, 0.5)
acc, pres, recall = acc_params(tp, fp, fn, tn)
print(f"Accuracy: {ft(acc)} Precision: {ft(pres)} Recall: {ft(recall)}")

add_graphic(fpr, tpr, title="ROC", x_label="FPR", y_label="TPR", area=count_area(fpr, tpr))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random', alpha=.8)
plt.show()


add_graphic(rec, prs, title="PR", x_label="Recall", y_label="Precision", area=count_area(rec, prs))
plt.plot([0, 1], [0, 0], linestyle='--', lw=2, color='r', label='---', alpha=.8)
plt.show()







