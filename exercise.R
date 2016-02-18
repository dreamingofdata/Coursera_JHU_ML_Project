library(caret)
library(parallel)
library(doParallel)

set.seed(84738)

final_test <- read.csv("pml-testing.csv", stringsAsFactors = FALSE)
import_data <- read.csv("pml-training.csv", stringsAsFactors = FALSE)

################################################
#data cleansing
################################################
# Removing index and time-series columns 
import_data <- import_data[,-c(1:7)]

#removing columns high in missing values
library(YaleToolkit)

variable_summary <- whatis(import_data)
variable_summary$index <- 1:nrow(variable_summary)
columns_to_exclude <- variable_summary[variable_summary$missing != 0,"index"]

import_data <- import_data[,-columns_to_exclude]

#Remove variables where there are values = ""
first_flag = TRUE
for (x in 1:length(import_data)){
  if (sum(import_data[,x]=="") != 0){
    if (first_flag){
      first_flag = FALSE
      columns_to_exclude = x
    } else {
      columns_to_exclude <- c(columns_to_exclude, x)
    }
  }
}
import_data <- import_data[,-columns_to_exclude]


#data splitting
inTrain <- createDataPartition(y=import_data$classe, p=0.75, list=FALSE)
training <- import_data[inTrain,]
testing <- import_data[-inTrain,]

#Set parallelization and train a model
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)
fitControl <- trainControl(method = "cv", number = 10, allowParallel = TRUE)
fit <- train(classe ~ ., data=training, method="rf", trControl = fitControl)
stopCluster(cluster)

save(fit, file="fit.RData") #for use later, rather than retraining each time
#Save time with the below
#load(file = "fit.RData")

#Test model against holdback
predictions <- predict(fit, testing)
cm <- confusionMatrix(predictions, testing$classe)
print(cm)



# Quiz solution
# B A B A A E D B A A B C B A E E A B B B