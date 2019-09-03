# Authors: Alan Lo, Arjun Goyal, Richard Hardis, James Will Trawick

library(kernlab)
library(ggplot2)
library(kknn)
setwd("C:\Users\richa\Documents\GitHub\6501-hw\HW2")  # Change this for your local machine
data_df = read.table("credit_card_data-headers.txt", header = TRUE)
data = as.matrix(data_df)

# Plot best factor against all other factors
model <- ksvm(data[,1:10],data[,11],type="C-svc", kernel="vanilladot",C=100,scaled=TRUE)
a = colSums(model@xmatrix[[1]] * model@coef[[1]])
a0 = -model@b

col_list = colnames(data_df[,-11])
low_factors = col_list[col_list != "A9"]
low_factors

accA9 = sum(data_df$R1 * data_df$A9) / nrow(data_df) * 100

data_df$R1 = as.factor(data_df$R1)
ggplot(data_df, aes(x=A9, y=A2, shape=R1, color=R1))+
  geom_point()



# KNN
kknn_accuracy = function(k_val){
  pred_responses <- rep(0, nrow(cred_card_data_headers)) #vector of 0s to insert the predicted response value of the model.
  for (i in 1:nrow(cred_card_data_headers)){
    #using [-i] to exclude the ith data point from the nearest neighbor calculation
    kknn.model = kknn(R1 ~., train = cred_card_data_headers[-i,], test = cred_card_data_headers[i,], k = k_val, kernel = "rectangular",scale = TRUE)
    #fitted(kknn.model) gives the value of the fraction of nearest neighbors to the ith data point that are 1.
    fit <- fitted(kknn.model)
    binary_response <- as.integer(fit + 0.5) #maps the fitted value to 0 or 1
    pred_responses[i] <- binary_response #adds the model's fitted value to the pred_responses vector
  }
  acc = sum(pred_responses == cred_card_data_headers[,11]) / nrow(cred_card_data_headers)
  return(acc)
}

k_acc <- rep(0, 20)
for (i in 1:20){ #values of k
  k_acc[i] = kknn_accuracy(i)
}


max(k_acc) #best k accuracy

max_k <- which(k_acc %in% max(k_acc))
max_k

ggplot(as.data.frame(k_acc), aes(x= 1:20, y=k_acc)) + geom_point() +
  ggtitle("Accuracies of K-Nearest Neighbors Model by Values of K") +
  xlab("Integer Values of K") + ylab("Accuracy")

"'
Write the functions below:
'"

# Function for K-folds.  Takes in dataset, model type, model parameters (parameters needs to be flexible depending on SVM, KNN or others)

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
  
  if(num_test_points==-1){    # The user has requested to use a specific number of points for testing
    num_test_points = floor(num_observations*(test_percent/10))
  }
  
  test_indices = sample(1:num_observations, num_test_points, replace=FALSE)
  test_set = data[test_indices,]
  training_set = data[-test_indices,]    # -test_indices removes all the rows with the test indices
  
  return(c(training_set,test_set))
}

