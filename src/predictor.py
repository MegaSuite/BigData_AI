# "* ==========================================================",
# "* Description: predictor.py",
# "* All rights reserved.",
# "* Date: 2023/12/22 16:46",
# "* ==========================================================",
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import visual as vs


def linear_reg_single(x_train, x_test, y_train, y_test):
    """
    单次线性回归
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return:
    """
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    acc_ln = metrics.accuracy_score(y_test, y_pred.round())

    return acc_ln


def linear_reg(x_train, x_test, y_train, y_test):
    """
    进行四次线性回归，并对准确率进行平均
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return:
    """
    accuracy_list = []
    for i in range(1, 5):
        accuracy_single = linear_reg_single(x_train, x_test, y_train, y_test)
        accuracy_list.append(accuracy_single)
    print("Linear Regression:")
    print(f"Accuracy:{sum(accuracy_list) / len(accuracy_list):.5f}")


def knn_reg_single(x_train, x_test, y_train, y_test):
    """
    单次KNN回归
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return:
    """
    knn = KNeighborsRegressor(n_neighbors=32)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    acc_knn = metrics.accuracy_score(y_test, y_pred.round())

    return acc_knn


def knn_reg(x_train, x_test, y_train, y_test):
    """
    进行四次KNN回归，并对准确率进行平均
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return:
    """
    accuracy_list = []
    for i in range(1, 5):
        accuracy_single = knn_reg_single(x_train, x_test, y_train, y_test)
        accuracy_list.append(accuracy_single)
    print("KNN Regressor:")
    print(f"Accuracy:{sum(accuracy_list) / len(accuracy_list):.5f}")


def random_forest_reg_single(x_train, x_test, y_train, y_test):
    """
    单次随机森林回归
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return:
    """
    rf = RandomForestRegressor(n_estimators=560, random_state=42)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    acc_rf = metrics.accuracy_score(y_test, y_pred.round())

    return acc_rf


def random_forest_reg(x_train, x_test, y_train, y_test):
    """
    进行四次随机森林回归，并对准确率进行平均
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return:
    """
    accuracy_list = []
    for i in range(1, 5):
        accuracy_single = random_forest_reg_single(x_train, x_test, y_train, y_test)
        accuracy_list.append(accuracy_single)
    print("Random Forest Regressor:")
    print(f"Accuracy:{sum(accuracy_list) / len(accuracy_list):.5f}")


def logistic_reg_single(x_train, x_test, y_train, y_test):
    """
    单次逻辑回归
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return:
    """
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(x_train, y_train)
    y_pred = logreg.predict(x_test)
    acc_lr = metrics.accuracy_score(y_test, y_pred.round())

    return acc_lr


def logistic_reg(x_train, x_test, y_train, y_test):
    """
    进行四次逻辑回归，并对准确率进行平均
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return:
    """
    accuracy_list = []
    for i in range(1, 5):
        accuracy_single = logistic_reg_single(x_train, x_test, y_train, y_test)
        accuracy_list.append(accuracy_single)
    print("Logistic Regression:")
    print(f"Accuracy:{sum(accuracy_list) / len(accuracy_list):.5f}")
