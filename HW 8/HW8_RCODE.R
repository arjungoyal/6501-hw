#Load the required packages and the dataset into R
library(caret)
library(glmnet)
library(leaps)
crime <- read.csv("uscrime.txt", header= T, sep = "\t")

#Split the data into training and test sets
set.seed(700)
inTraining <- createDataPartition(crime$Crime, p = .75, list = FALSE)
training <- crime[inTraining,]
testing  <- crime[-inTraining,]

#Build a model using the leapSeq method of variable selection
set.seed(825)
stepLeapFit <- train(Crime ~ ., data = training, 
                     method = "leapSeq", tuneGrid = data.frame(nvmax = 1:10),
                     trControl = trainControl(## 10-fold CV
                       method = "repeatedcv",
                       number = 10,
                       ## repeated ten times
                       repeats = 10),
                     verbose=F
)
summary(stepLeapFit$finalModel)

#The regression coefficients for the stepLeapFit model
coef(stepLeapFit$finalModel, stepLeapFit$bestTune[['nvmax']])

#The prediction on the test set for the model
stepLeapTestPrediction <- predict(stepLeapFit, testing[-ncol(testing)], interval = "prediction")
stepLeapTestPrediction

#The MSPE of this model on the test data
mean((stepLeapTestPrediction-testing$Crime)^2)

#The Full Model using all the predictor variables and its MSPE on the test data
stepLeapFull <- lm(Crime~., data=training)
stepLeapFullPrediction<- predict(stepLeapFull, testing[-ncol(testing)], interval = "prediction")[,1]

mean((stepLeapFullPrediction-testing$Crime)^2)

#Comparison using the AICs of each model
stepLeapModel <- lm(Crime~Po1+M.F+Ineq, data=training)

exp((AIC(stepLeapModel) - AIC(stepLeapFull))/2)

#LASSO APPROACH - Scaling the Data 
trainingmatrix = as.matrix(training[,-ncol(training)])
trainingresponse = as.vector(training[,ncol(training)])
trainingresponsevector = as.numeric(unlist(trainingresponse))

scaledTraining <- scale(trainingmatrix, center = TRUE, scale = TRUE)

testingmatrix = as.matrix(testing[,-ncol(testing)])
testingresponse = as.vector(testing[,ncol(testing)])
testingresponsevector = as.numeric(unlist(testingresponse))

scaledTesting <- scale(testingmatrix, center = TRUE, scale = TRUE)

#Building the model
set.seed(100)
crossvalfit <- cv.glmnet(scaledTraining, trainingresponsevector)
plot(crossvalfit)
crossvalfit$lambda.min
coef(crossvalfit, s = "lambda.min")

#Running the prediction on the test data and evaluating using MSPE
LASSOTestPrediction <- predict(crossvalfit, scaledTesting, interval = "prediction", s = "lambda.min")

mean((LASSOTestPrediction-testingresponsevector)^2)

#ELASTIC NET

#Running the approach using 10 different values of alpha from 0 to 0.9
MSPE <- rep(0, 10)
i = 1
for(alpha in seq(0, 0.9, 0.1)){
  print(alpha)
  set.seed(100)
  crossvalfit1 <- cv.glmnet(scaledTraining, trainingresponsevector, alpha = alpha)
  
  ElasticNetTestPrediction <- predict(crossvalfit1, scaledTesting, interval = "prediction", s = "lambda.min")
  print(ElasticNetTestPrediction)
  MSPE[i] = mean((ElasticNetTestPrediction-testingresponsevector)^2)
  
  i = i + 1
}

#Plot of MSPE vs. Alpha
plot(x=seq(0,0.9,0.1), MSPE, main = "Plot of MSPE vs Alpha Value", xlab = "Alpha")


#The model that uses an alpha value of 0.2
set.seed(100)
crossvalfitEN <- cv.glmnet(scaledTraining, trainingresponsevector, alpha = 0.2)
coef(crossvalfitEN, s = "lambda.min")
