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




