import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub

dagshub.init(repo_owner='akhilamurarka',repo_name='MLOPS-MLFlow-Experiments',mlflow=True)

TRACKING_URI=""

mlflow.set_tracking_uri(TRACKING_URI)

wine=load_wine()
X=wine.data
y=wine.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=42)

max_depth=10
n_estimators=5

# Mention your experiment below
# if we define experiment name which is not there, then it creates the new experiment
""" mlflow.set_experiment('YT-MLOPS_exp1') """

with mlflow.start_run(experiment_id=496870060506666675):
  rf=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators,random_state=42)
  rf.fit(X_train,y_train)
  y_pred=rf.predict(X_test)
  accuracy=accuracy_score(y_pred,y_test)
  
  mlflow.log_metric('accuracy',accuracy)
  mlflow.log_param('max_depth',max_depth)
  mlflow.log_param('n_estimators',n_estimators)
  
  # creating 
  cm=confusion_matrix(y_pred,y_test)
  plt.figure(figsize=(6,6))
  sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=wine.target_names,yticklabels=wine.target_names)
  plt.ylabel('Actual')
  plt.xlabel('Predicted')
  plt.title('Confusion Matrix')
  
  plt.savefig("Confusion_matrix.png")
  
  # log artifacts
  mlflow.log_artifact("Confusion_matrix.png")
  mlflow.log_artifact(__file__)
  
  # tags
  mlflow.set_tags({
    'Author':"akhil",
    "Project":"Wine Classification"
  })
  
  # log the model
  mlflow.sklearn.log_model(rf,"Random-Forest-Model")
  
  print(accuracy)
  