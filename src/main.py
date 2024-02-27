# "* ==========================================================",
# "* Description: main.py",
# "* All rights reserved.",
# "* Date: 2023/12/10 15:45",
# "* ==========================================================",
import pandas as pd
import time as t
import dataset
import predictor

start_pg = t.perf_counter()  # 项目起始计时戳

# 设置pandas显示参数，使其能够显示所有列和行
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# 主函数
if "__main__" == __name__:
    print("\nRunning....\n")
    # 数据集初始化，将三个数据集合并
    df_merged = dataset.data_init()
    # 数据集预处理，将数据集中的缺失值进行填充，使用独热编码对数据集进行编码
    df_merged = dataset.onehot_encoding(df_merged)
    # 数据集划分，按照9:1的比例划分训练集和测试集
    x_train, x_test, y_train, y_test = dataset.data_split(df_merged)
    # 线性回归
    start = t.perf_counter()
    print("===============================")
    predictor.linear_reg(x_train, x_test, y_train, y_test)
    end = t.perf_counter()
    print(f"Time Consumed: {(end - start):.3f} s")
    # K近邻回归
    start = t.perf_counter()
    print("===============================")
    predictor.knn_reg(x_train, x_test, y_train, y_test)
    end = t.perf_counter()
    print(f"Time Consumed: {(end - start):.3f} s")
    # 随机森林回归
    start = t.perf_counter()
    print("===============================")
    predictor.random_forest_reg(x_train, x_test, y_train, y_test)
    end = t.perf_counter()
    print(f"Time Consumed: {(end - start):.3f} s")
    # 逻辑回归
    start = t.perf_counter()
    print("===============================")
    predictor.logistic_reg(x_train, x_test, y_train, y_test)
    end = t.perf_counter()
    print(f"Time Consumed: {(end - start):.3f} s")

end_pg = t.perf_counter()  # 项目结束计时戳
print("\n\n")
print("===============================")
print(f"|Program Running Time: {(end_pg - start_pg):.3f} s|")
print("===============================")
# 输出项目运行时间
