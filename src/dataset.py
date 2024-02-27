# "* ==========================================================",
# "* Description: dataset.py",
# "* All rights reserved.",
# "* Date: 2023/12/22 16:38",
# "* ==========================================================",
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split


def data_init():
    """
    从'/data'读取数据，合并数据，数据预处理
    :return: df_merged
    """""
    # 读取数据，设置列名，删除‘Timestamp’列
    df_mv = pd.read_csv('../data/movies.csv', engine='python', sep=',',
                        names=["MovieID", "Title", "Genres"], encoding='ISO-8859-1')
    df_mv.drop(df_mv.index[0], inplace=True)
    df_rt = pd.read_csv('../data/ratings.csv', engine='python', sep=',',
                        names=["UserID", "MovieID", "Rating", "Timestamp"], encoding='ISO-8859-1')
    df_rt.drop(df_rt.index[0], inplace=True)
    df_rt.drop("Timestamp", axis=1, inplace=True)
    df_tg = pd.read_csv('../data/tags.csv', engine='python', sep=',',
                        names=["UserID", "MovieID", "Tag", "Timestamp"], encoding='ISO-8859-1')
    df_tg.drop(df_tg.index[0], inplace=True)
    df_tg.drop("Timestamp", axis=1, inplace=True)
    # 合并数据
    df_merge1 = df_mv.merge(df_rt, on='MovieID', how='outer')  # Movie and their ratings
    df_merge2 = df_mv.merge(df_tg, on='MovieID', how='outer')  # Movie and their tags
    df_merged = df_merge1.merge(df_merge2, how='outer')
    # 规范化数据
    df_merged.fillna(0, inplace=True)
    df_merged.MovieID = df_merged.MovieID.astype(int)
    df_merged.UserID = df_merged.UserID.astype(int)
    df_merged.Rating = df_merged.Rating.astype(float)
    # 添加平均评分列
    avg_ratings = df_merged.groupby('MovieID')['Rating'].mean()
    df_merged = pd.concat([df_mv, avg_ratings], axis=1)
    df_merged['Year'] = df_merged.Title.str.extract("\((\d{4})\)", expand=True)
    df_merged.fillna(0, inplace=True)
    df_merged.MovieID = df_merged.MovieID.astype(int)
    # 整数化
    df_merged.Rating = df_merged.Rating.astype(int)
    df_merged.Year = df_merged.Year.astype(int)

    return df_merged


def onehot_encoding(df_merged):
    """
    对df_merged中的Genres列进行onehot编码
    :param df_merged:
    :return:
    """
    list2series = pd.Series(df_merged.Genres.str.split('|').tolist())
    list2series.fillna('0', inplace=True)
    mlb = MultiLabelBinarizer()
    df_genres = pd.DataFrame(mlb.fit_transform(list2series), columns=mlb.classes_, index=df_merged.index)
    # 将onehot编码后的数据与df_merged合并，删除原Genres列和Title列
    df_merged = df_merged.join(df_genres)
    df_merged.drop("Title", axis=1, inplace=True)
    df_merged.drop("Genres", axis=1, inplace=True)
    df_merged.drop("0", axis=1, inplace=True)

    return df_merged


def data_split(df_merged):
    """
    将数据集划分为训练集和测试集,按照9:1的比例
    :param df_merged:
    :return:
    """
    x_feature = df_merged.drop("Rating", axis=1)
    y_target = df_merged['Rating']
    x_train, x_test, y_train, y_test = train_test_split(x_feature, y_target, test_size=0.1, random_state=None)

    return x_train, x_test, y_train, y_test
