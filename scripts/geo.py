from sklearn.cluster import Birch
from sklearn.cluster import KMeans
from sklearn.metrics import log_loss
import pandas as pd 
import numpy as np

def cluster_latlon(n_clusters, data, method = "birch"):
    def fit_cluster(coords, method):
        if method == "birch":
            brc = Birch(branching_factor=100, n_clusters=n_clusters, threshold=0.01,compute_labels=True)
            brc.fit(coords)
            clusters=brc.predict(coords)
        elif method == "kmeans":
            km = KMeans(n_clusters=n_clusters)
            km.fit(coords)
            clusters=km.predict(coords)
        return clusters
    # data must be df_full
    #split the data between "around NYC" and "other locations" basically our first two clusters 
    lat_lon_cols = ['latitude', "longitude"]
    mask = (data.longitude>-74.05)&(data.longitude<-73.75)&(data.latitude>40.4)&(data.latitude<40.9)
    data_c=data.loc[mask, lat_lon_cols].copy(deep=True)
    data_e=data.loc[~mask, lat_lon_cols].copy(deep=True)
    #put it in matrix form
    coords=data_c.as_matrix(columns=lat_lon_cols)

    clusters = fit_cluster(coords, method)
    data_c["cluster_"+str(n_clusters)]=clusters
    data_e["cluster_"+str(n_clusters)]=-1 #assign cluster label -1 for the non NYC listings 
    assert data_c.shape[0] + data_e.shape[0] == data.shape[0]
    data = pd.merge(data, data_c[["cluster_"+str(n_clusters)]], left_index = True, right_index = True, how = 'left')
    data = pd.merge(data, data_e[["cluster_"+str(n_clusters)]], left_index = True, right_index = True, how = 'left')
    data["cluster_"+str(n_clusters)] = data["cluster_"+str(n_clusters)+"_x"].fillna(0)+data["cluster_"+str(n_clusters)+"_y"].fillna(0)
    assert data[data["cluster_"+str(n_clusters)].isnull()].shape[0] == 0
    #plt.scatter(data_c["longitude"], data_c["latitude"], c=data_c["cluster_"+str(n_clusters)], s=10, linewidth=0.1)
    #plt.title(str(n_clusters)+" Neighbourhoods from clustering")
    #plt.show()
    return data

def compute_logloss(n_cluster, data, train_index, val_index):
    def fit_classifier(X_train, y_train):
        from sklearn.ensemble import RandomForestClassifier as RFC
        clf=RFC(n_estimators=1000, random_state=42)
        clf.fit(X_train, y_train)
        return clf   
    # data is df_full
    data_cluster=cluster_latlon(n_cluster,data, method="kmeans")
    df_train=data_cluster[data_cluster["set"]=="train"].copy(deep=True)
    data_cluster = None

    target_num_map={"high":0, "medium":1, "low":2}
    df_train["interest_level_num"]=df_train["interest_level"].apply(lambda x: target_num_map[x]).values
    features = ["bathrooms", "bedrooms", "price",        
                    "num_photos", "num_features", "num_description_words",                    
                    "created_month", "created_day", "created_hour", "cluster_"+str(n_cluster)
                ]

    X_train, y_train =df_train.loc[train_index, features].copy(deep=True), df_train.loc[train_index, "interest_level_num"].copy(deep=True) 
    X_val, y_val = df_train.loc[val_index, features].copy(deep=True), df_train.loc[val_index, "interest_level_num"].copy(deep=True)
    for col in features:
        assert len(X_train[X_train[col].isnull()]) == 0   
    print("Fitting for %s clusters" % n_cluster)
    clf = fit_classifier(X_train, y_train)
    print("Done")
    y_val_pred = clf.predict_proba(X_val)
    return log_loss(y_val, y_val_pred)

