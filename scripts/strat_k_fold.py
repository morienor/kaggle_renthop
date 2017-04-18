def compute_logloss(n_cluster,data):
    data_cluster=cluster_latlon(n_cluster,data)
    train=data_cluster[data_cluster["Source"]=="train"]

    target_num_map={"high":0, "medium":1, "low":2}
    y=np.array(train["interest_level"].apply(lambda x: target_num_map[x]))
    
    features = ["bathrooms", "bedrooms", "price", 
                                                        
                    "num_photos", "num_features", "num_description_words",                    
                    "created_month", "created_day", "created_hour", "cluster_"+str(n_cluster)
                   ]
    
    X_train, X_val,y_train, y_val =train_test_split( train[features], y, test_size=0.33, random_state=42)
    clf.fit(X_train, y_train)

    y_val_pred = clf.predict_proba(X_val)
    return log_loss(y_val, y_val_pred)