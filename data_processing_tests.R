# Read data set
data_df = read.csv("D:/PhD Program/Final Research/Dissertation/Writeup/best_parm_data.csv")


# Perform t-test/kruskal test to determine significance outlier treatement

kruskal_model <- kruskal.test(Best.Accuracy~Anomaly.Treated., data=data_df)
kruskal_model

