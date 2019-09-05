# Authors: Alan Lo, Arjun Goyal, Richard Hardis, James Will Trawick

library(kernlab)
library(ggplot2)
library(kknn)

setwd("/Users/alan/Dropbox (GaTech)/Fall19/6501/6501-hw/hw2")  # Change this for your local machine
data_df = read.table("credit_card_data-headers.txt", header = TRUE)
#data = as.matrix(data_df)

# KSVM
ksvm_accuracy = function(train_data, test_data, kern, C){
  model = ksvm(train_data[,1:10],train_data[,11], type="C-svc", kernel=kern, C=C, scaled=TRUE)
  pred = predict(model,test_data[,1:10])
  acc = sum(pred == test_data[,11]) / nrow(test_data)
  return(acc)
}

# KNN
kknn_accuracy = function(train_data, k_val, kern){
  pred_responses <- rep(0, nrow(train_data)) #vector of 0s to insert the predicted response value of the model.
  for (i in 1:nrow(train_data)){
    #using [-i] to exclude the ith data point from the nearest neighbor calculation
    kknn.model = kknn(R1 ~., train = train_data[-i,], test = train_data[i,], k = k_val, kernel = kern, scale = TRUE)
    #fitted(kknn.model) gives the value of the fraction of nearest neighbors to the ith data point that are 1.
    fit = fitted(kknn.model)
    binary_response <- as.integer(fit + 0.5) #maps the fitted value to 0 or 1
    pred_responses[i] <- binary_response #adds the model's fitted value to the pred_responses vector
  }
  acc = sum(pred_responses == train_data[,11]) / nrow(train_data)
  return(acc)
}

# Function for K-folds.  Takes in dataset, model type, model parameters (parameters needs to be flexible depending on SVM, KNN or others)
get_kfolds_expected_accuracy = function(data_df, k_folds, MODEL_TYPE, MODEL_PARAMETER){
  "
  Arguments:
    data_array: data dataframe of the dataset to run k-folds cross validation with
    k_folds: the number of folds to use in k-folds cross validation
    model_function: the name of the modeling function.  Either 'ksvm' or 'kknn' for this HW.

  Return:
    expected_accuracy: the average accuracy of the models in the k-folds number of training-testing splits
  "

  # Get the k folds data groups
  number_test_points = floor(nrow(data_df)/k_folds)
  data_copy = data_df
  kf_groups = list()

  for (i in 1:k_folds){
    if (i==k_folds){kf_groups[[i]] = data_copy}
    else{
      set_list = train_test_split(data_copy, 0, number_test_points)
      new_test_group = set_list[[2]]
      data_copy = set_list[[1]]
      kf_groups[[i]] = new_test_group
    }
  }

  # Blank accuracy list
  accuracies = list()

  # Get the accuracy performance of each of the k models
  for (i in 1:k_folds){
    # Get the test group of data and the training group of data
    test_df = kf_groups[[i]]
    train_groups = kf_groups[-i]
    training_df = train_groups[[1]]

    for (group in train_groups[-1]){
      training_df = rbind(training_df, group)
    }

    # Transform test and train dataframes into matrices
    test = test_df
    train = training_df

    # Train the model on the training data
    if (MODEL_TYPE=='ksvm'){    # KSVM
      train_mat = as.matrix(train)
      test_mat = as.matrix(test)
      current_accuracy = ksvm_accuracy(train_mat, test_mat, "rbfdot", MODEL_PARAMETER)
    }
    else{    # KKNN
      current_accuracy = kknn_accuracy(train, MODEL_PARAMETER, "rectangular")
    }

    # Put the test accuracy into the accuracy list
    accuracies[[i]] = current_accuracy
  }

  # Find the average of the accuracy list
  expected_accuracy = mean(unlist(accuracies))

  # Return the expected accuracy
  return(expected_accuracy)
}

# Function for Splitting data randomly into training and testing data.  Returns two dataframes or arrays?
train_test_split = function(data, test_percent=0, num_test_points=-1){
  "
  Arguments:
    data: the full dataset as a dataframe
    test_percent: The percentage of the data to partition into the test set
    num_test_points: Optional parameter to set the number of points to reserve for testing instead of percentage based
  Return:
    list: c(training dataframe, test dataframe)
  "
  num_observations = nrow(data) # How many data points are there?
  print(num_observations)

  if(num_test_points==-1){    # The user has requested to use a specific number of points for testing
    num_test_points = floor(num_observations*(test_percent/100))
  }

  test_indices = sample(1:num_observations, num_test_points, replace=FALSE)
  test_set = data[test_indices,]
  training_set = data[-test_indices,]    # -test_indices removes all the rows with the test indices

  return(list(training_set,test_set))
}

# Get the cross-validated accuracy of a certain model with a given dataset
expected_accuracy_over_kfolds = get_kfolds_expected_accuracy(data_df, 10, 'ksvm', 6)

# knn_accs = c()
# svm_accs = c()

# for (i in 1:10){
#   knn_accs = c(knn_accs, get_kfolds_expected_accuracy(data_df, 10, 'kknn', i))
#   svm_accs = c(svm_accs, get_kfolds_expected_accuracy(data_df, 10, 'ksvm', i))
# }

# print(knn_accs)
# print(svm_accs)

split_data = train_test_split(data_df, 20)
test_data = split_data[[2]]
train_val_data = split_data[[1]]
split_data_tv = train_test_split(train_val_data, 20)
train_data = split_data_tv[[1]]
validate_data = split_data_tv[[2]]

train_data = as.matrix(train_data)
validate_data = as.matrix(validate_data)
test_data = as.matrix(test_data)

# 60% Training Data, %20 Validation Data, %20 Test Data

svm_validation_accs = c()
knn_validation_accs = c()

for (i in 1:5){
  C = 10^i
  svm_validation_accs = c(svm_validation_accs, ksvm_accuracy(train_data, validate_data, "vanilladot", C))
  knn_validation_accs = c(knn_validation_accs, kknn_accuracy(train_data = as.data.frame(train_data), i, "rectangular"))
}

svm_index_best_acc = min(which(svm_validation_accs==max(svm_validation_accs)))
knn_index_best_acc = min(which(knn_validation_accs==max(knn_validation_accs)))
best_C_val = 10^(index_best_acc-1)
best_k_val = knn_index_best_acc
print(best_C_val)
print(best_k_val)

test_svm_acc = ksvm_accuracy(train_data, test_data, "vanilladot", best_C_val)
test_knn_acc = kknn_accuracy(as.data.frame(test_data), best_k_val, "rectangular")
print(test_svm_acc)
print(test_knn_acc)

###########################
##  K - Means Iris
##########################
library(datasets)
library(ggplot2)
ggplot(iris, aes(Sepal.Length, Sepal.Width, color=Species))+geom_point()
summary(iris)
# Remove the species label from Iris dataset
iris_no_class = iris[,-5]
iris_class = iris[,c("Species")]

# Normalize data?
normalize <- function(x){
  return ((x-min(x))/(max(x)-min(x)))
}

iris_no_class$Sepal.Length<- normalize(iris_no_class$Sepal.Length)
iris_no_class$Sepal.Width<- normalize(iris_no_class$Sepal.Width)
iris_no_class$Petal.Length<- normalize(iris_no_class$Petal.Length)
iris_no_class$Petal.Width<- normalize(iris_no_class$Petal.Width)

kmeans_acc = function(train_data, k){
  model = kmeans(train_data,k)
  #return(model$size)

  par(mfrow=c(2,2))
  plot(iris_no_class[c(1,2)],col=model$cluster)
  plot(iris_no_class[c(1,2)],col=iris_class)
  plot(iris_no_class[c(3,4)],col=model$cluster)
  plot(iris_no_class[c(3,4)],col=iris_class)
  return(table(model$cluster,iris_class))
  #return(model$cluster)

}

# Test different values of k

for(i in 1:10){
  kmeans_acc(iris_no_class,i)
}


kmeans_acc(iris_no_class, 4)









set.seed(200)
k.max <- 15
#create a vector of the total within cluster sum of squares for each value of k from 1 to 10
tot_wss <- sapply(1:k.max, function(k){kmeans(iris[,3:4],k,nstart=20,iter.max=20)$tot.withinss})
tot_wss

plot(1:k.max, tot_wss, type="b", main = "Plot of Total Distances by Number of Clusters", xlab = "Number of Clusters (k)", ylab = "Total Distances (Within Cluster Sum of Squares)")

"
kmeans_acc(iris_no_class, 3)
"

kmeans_acc(iris_no_class, 3)

