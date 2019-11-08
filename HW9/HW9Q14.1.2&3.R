setwd("C:/Users/james/Desktop")
df = read.table('breast-cancer-wisconsin.data.txt',sep=',',header = F)


nodata = df[df$V7 == "?", ] #df with missing data in column v7

missingdata = nodata[,-c(1,7)]

missingdatarows = as.numeric(rownames(nodata)) #rows with no data in column v7

regressiondata = df[-missingdatarows, ] #df of good rows

gooddata =  df[-missingdatarows, ] #df of good rows

response = as.numeric(as.character(regressiondata$V7)) #Response column is column with missing data

regressiondata["Response"] = response  #adding this column to df as a numeric column

regressiondata = regressiondata[,-c(1,7)] #running regression and excluding column 1 (ID numbers) and column 7(response column)

regmodel = lm(Response~., data=regressiondata) #regression

v7vals = as.numeric(predict(regmodel, newdata = missingdata)) #predicting missing values based on other attributes

newdata = nodata[,-c(7)]

newdata1 = newdata


library(tibble)


newdata = add_column(newdata, V7 = v7vals, .after = 6) # add regression estimation to df

newdf = rbind(newdata, gooddata) #combine back data frames

pertregvals = rnorm(length(v7vals)) + v7vals

newdata1 = newdata1[,-c(7)]
newdata1 = add_column(newdata1, V7 = pertregvals, .after = 6)

newdf1 = rbind(newdata1, gooddata)
