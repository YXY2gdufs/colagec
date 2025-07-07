from pathlib import Path
import pandas as pd
from datasets import Dataset, DatasetDict

CURRENT_DIR = Path(__file__).parent
DATA_DIR = CURRENT_DIR.parent / "data"

TRAIN_FILE_ALL = pd.read_csv('/Users/yxy/PycharmProjects/CoLA-baselines-master/25COLA-data/En/CoLA/train.csv')
# df_train = pd.read_csv('/Users/yxy/PycharmProjects/CoLA-baselines-master/multi-language/datasets/combined_cola_ALL/乱序/combined_cola_train_811.csv')
# TRAIN_FILE = df_train.sample(frac=0.3, random_state=42)

DEV_FILE_ALL = pd.read_csv('/Users/yxy/PycharmProjects/CoLA-baselines-master/25COLA-data/En/CoLA/train.csv')
# df_dev = pd.read_csv('/Users/yxy/PycharmProjects/CoLA-baselines-master/multi-language/datasets/combined_cola_ALL/乱序/combined_cola_dev_811.csv')
# IN_DOMAIN_DEV_FILE = df_dev.sample(frac=0.3, random_state=42)
# OUT_OF_DOMAIN_DEV_FILE = '../data/out_of_domain_dev.csv'
TEST_FILE = pd.read_csv('/Users/yxy/PycharmProjects/CoLA-baselines-master/25COLA-data/En/CoLA/train.csv')

# 使用俄语
# TRAIN_FILE = '../data/in_domain_train.csv'
# IN_DOMAIN_DEV_FILE = '../data/in_domain_dev.csv'
# OUT_OF_DOMAIN_DEV_FILE = '../data/out_of_domain_dev.csv'
# TEST_FILE = '../data/test.csv'
# TRAIN_FILE = DATA_DIR / "in_domain_train.csv"
# IN_DOMAIN_DEV_FILE = DATA_DIR / "in_domain_dev.csv"
# OUT_OF_DOMAIN_DEV_FILE = DATA_DIR / "out_of_domain_dev.csv"
# TEST_FILE = DATA_DIR / "test.csv"

# 使用全部数据的时候用以下代码
# def read_splits(*, as_datasets):
#     # train_df, in_domain_dev_df, out_of_domain_dev_df, test_df = map(
#     #     pd.read_csv, (TRAIN_FILE, IN_DOMAIN_DEV_FILE, OUT_OF_DOMAIN_DEV_FILE, TEST_FILE)
#     # )
#     train_df, in_domain_dev_df = map(
#         pd.read_csv, (TRAIN_FILE, IN_DOMAIN_DEV_FILE)
#     )
#
#     # concatenate datasets to get aggregate metrics
#     # dev_df = pd.concat((in_domain_dev_df, out_of_domain_dev_df))
#     dev_df = in_domain_dev_df
#
#     # if as_datasets:
#     #     train, dev, test = map(Dataset.from_pandas, (train_df, dev_df, test_df))
#     #     return DatasetDict(train=train, dev=dev, test=test)
#     # else:
#     #     return train_df, dev_df, test_df
#     if as_datasets:
#         train, dev = map(Dataset.from_pandas, (train_df, dev_df))
#         return DatasetDict(train=train, dev=dev)
#     else:
#         return train_df, dev_df

# 使用sample方法挑出一部分数据的时候用
def read_splits(*, as_datasets=True, sample_frac=None, random_state=42):
    # 直接使用已经加载并采样的DataFrame对象
    train_df = TRAIN_FILE_ALL
    dev_df = DEV_FILE_ALL
    test_df = TEST_FILE
    print(type(train_df))

    # concatenate datasets to get aggregate metrics
    # dev_df = pd.concat((in_domain_dev_df, out_of_domain_dev_df))

#    if as_datasets:
#        train, dev = map(Dataset.from_pandas, (train_df, dev_df))
#        return DatasetDict(train=train, dev=dev)
#    else:
#        return train_df, dev_df
    if as_datasets:
        train_dataset = Dataset.from_pandas(train_df)
        dev_dataset = Dataset.from_pandas(dev_df)
        test_dataset = Dataset.from_pandas(test_df)
        return DatasetDict(train=train_dataset, dev=dev_dataset, test=test_dataset)
    else:
        return train_df, dev_df, test_df
