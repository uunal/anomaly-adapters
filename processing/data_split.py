import pandas as pd

from sklearn.model_selection import train_test_split

def read():
    #HDFS_DATASET = "data/hdfs/anomaly/prepared.csv"
    FIREWALL_DATASET = "data/firewall/anomaly/prepared.csv"
    dataset = pd.read_csv(FIREWALL_DATASET)
    logs=dataset['log'].tolist()
    print(len(logs))

def main():
    HDFS_DATASET = "data/hdfs/anomaly/prepared.csv"
    FIREWALL_DATASET = "data/firewall/anomaly/prepared.csv"
    fdataset = pd.read_csv(FIREWALL_DATASET)
    fdataset.drop(['type'], axis=1)
    hdataset = pd.read_csv(HDFS_DATASET)
    dataset = pd.concat([fdataset,hdataset])
    logs=dataset['log'].tolist()
    labels=dataset['label'].tolist()
    #types=dataset['type'].tolist()
    # DO NOT FORGET to change stratify column if ..
    X_train, X_test, y_train, y_test = train_test_split(logs, labels, 
                                                        train_size=0.80, 
                                                        random_state=666,
                                                        stratify=labels)

    train_dict={}
    train_dict['log'] = X_train
    train_dict['label'] = y_train 

    train_pd = pd.DataFrame(train_dict)
    train_pd.to_csv("data/all/anomaly/train.csv", index=False, header=True)

    test_dict={}
    test_dict['log'] = X_test
    test_dict['label'] = y_test 

    test_pd = pd.DataFrame(test_dict)
    test_pd.to_csv("data/all/anomaly/eval.csv", index=False, header=True)



if __name__ == "__main__":
    main()