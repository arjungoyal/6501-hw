# install.packages('caTools')
library(caTools)
getwd()
setwd('C:/Users/richa/Documents/GitHub/6501-hw/HW6')
crime_df = read.table("uscrime.txt", header = TRUE)

pred_df = crime_df[1:15]
#pdf_scaled = scale(pred_df, scale = TRUE, center=TRUE)

pca_model = prcomp(pred_df, center=TRUE, scale. = FALSE)

summary(pca_model)