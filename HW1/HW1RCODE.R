# Authors: Alan Lo, Arjun Goyal, Richard Hardis, James Will Trawick

library(kernlab)
library(ggplot2)
library(kknn)
setwd("C:/.../HW1")
data_df = read.table("credit_card_data-headers.txt", header = TRUE)
data = as.matrix(data_df)

model_func = function(C, kern, scaling){
  model <- ksvm(data[,1:10],data[,11],type="C-svc", kernel=kern,C=C,scaled=scaling)
  
  a = colSums(model@xmatrix[[1]] * model@coef[[1]])
  #a
  a0 = -model@b
  
  pred = predict(model,data[,1:10])
  acc = sum(pred == data[,11]) / nrow(data) * 100
  return(acc)
}

#Generate the list of C values be order of magnitude
clist = c()
for (i in -2:6){
  clist = c(clist, 10^i)
}

acc_v_T = c()
acc_v_F = c()
acc_p_T = c()
acc_p_F = c()

model1 = 1
if (model1 == 1){
for (i in clist){
  val_v_T = model_func(i, "vanilladot", TRUE)
  val_v_F = model_func(i, "vanilladot", FALSE)
  val_p_T = model_func(i, "polydot", TRUE)
  val_p_F = model_func(i, "polydot", FALSE)
  acc_v_T = c(acc_v_T, val_v_T)
  acc_v_F = c(acc_v_F, val_v_F)
  acc_p_T = c(acc_p_T, val_p_T)
  acc_p_F = c(acc_p_F, val_p_F)
}
}

# Create the accuracy df
acc_v_T_df = data.frame(c_vals=clist, accuracy=acc_v_T)
acc_v_F_df = data.frame(c_vals=clist, accuracy=acc_v_F)
acc_p_T_df = data.frame(c_vals=clist, accuracy=acc_p_T)
acc_p_F_df = data.frame(c_vals=clist, accuracy=acc_p_F)

plotit1 = 1
if (plotit1 == 1){
# Plot accuracy vs C values
  ggplot(data=acc_v_T_df, aes(x=c_vals, y=accuracy, group=1))+
    scale_x_log10()+
    geom_line()+
    geom_point()+
    labs(title="Accuracy of Linear SVM Over Range of C With Scaling",x="C Values", y = "Accuracy (%)")+
    theme_classic()
  
  ggplot(data=acc_v_F_df, aes(x=c_vals, y=accuracy, group=1))+
    scale_x_log10()+
    geom_line()+
    geom_point()+
    labs(title="Accuracy of Linear SVM Over Range of C Without Scaling",x="C Values", y = "Accuracy (%)")+
    theme_classic()
  
  ggplot(data=acc_p_T_df, aes(x=c_vals, y=accuracy, group=1))+
    scale_x_log10()+
    geom_line()+
    geom_point()+
    labs(title="Accuracy of Polynomial SVM Over Range of C With Scaling",x="C Values", y = "Accuracy (%)")+
    theme_classic()

  ggplot(data=acc_p_F_df, aes(x=c_vals, y=accuracy, group=1))+
      scale_x_log10()+
      geom_line()+
      geom_point()+
      labs(title="Accuracy of Polynomial SVM Over Range of C Without Scaling",x="C Values", y = "Accuracy (%)")+
      theme_classic()
}

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




