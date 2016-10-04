# Dumbbells Done Wrong
Bill Kimler  
February 22, 2016  

This analysis examines exercise techniques measured on 6 individuals using a dataset provided by the fine people at   ([Groupware@LES](http://groupware.les.inf.puc-rio.br/har)). Using various wearable sensors, a multitude of measurements (like acceleration, pitch, roll, kurtosis, etc) was gathered to help describe _how_ the subject performed the exercise. Independently, an observer classified the subject's performance as A through E, with Class A being proper technique and the other four indicating improper methods. The goal of this analysis is to see whether a model can be built based on the measurements that could accurately predict the observed classification of the subject's exercise performance.

#Data Processing
R was used as the primary tool for the data analysis. The following libraries were used:

```r
library(knitr)
library(caret)       #For model building
library(parallel)    #For speed
library(doParallel)  #For even more speed
library(YaleToolkit) #For data cleaning
```



###Data retrieval
The training and test files can be found at this [GitHub repo](https://github.com/dreamingofdata/Coursera_JHU_ML_Project). Assuming these zipped files are stored locally, they are unzipped, read in, and the uncompressed files deleted.


```r
unzip("pml-training.zip")
imported_data <- read.csv("pml-training.csv", stringsAsFactors = FALSE)
file.remove("pml-training.csv")
```

###Data cleansing
The first seven columns (`X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window`) do not relate to motion measurements and were dropped.

```r
data_cleansing <- imported_data[,-c(1:7)]
```

Now, using the [YaleToolkit](https://cran.r-project.org/web/packages/YaleToolkit/index.html) library, we use the `whatis` function to help quickly determine where any missing data may be located. 


```r
variable_summary <- whatis(data_cleansing)
```

Given there are 153 variables, we'll choose to prune any that have missing measurements in any record. This may seem like an extreme action, but it turns out in this dataset there were particular measurements for which the large majority of 19622 rows were `NA`. 


```r
variable_summary[15:20, 1:4]
```

```
##           variable.name      type missing distinct.values
## 15       min_pitch_belt   numeric   19216              16
## 16         min_yaw_belt character       0              68
## 17  amplitude_roll_belt   numeric   19216             148
## 18 amplitude_pitch_belt   numeric   19216              13
## 19   amplitude_yaw_belt character       0               4
## 20 var_total_accel_belt   numeric   19216              65
```



```r
variable_summary$index <- 1:nrow(variable_summary)
columns_to_exclude <- variable_summary[variable_summary$missing != 0,"index"]

data_cleansing <- data_cleansing[,-columns_to_exclude]
```

The previous steps went after `NA` values. The routine below performs the same pruning, but for blank ("") values.


```r
first_flag = TRUE
for (x in 1:length(data_cleansing)){
  if (sum(data_cleansing[,x]=="") != 0){
    if (first_flag){
      first_flag = FALSE
      columns_to_exclude = x
    } else {
      columns_to_exclude <- c(columns_to_exclude, x)
    }
  }
}
data_cleansing <- data_cleansing[,-columns_to_exclude]
```

We are now left with 53 variables to build a model from, all of which are guaranteed to have values (an important factor for tree-building algorithms).

###Data partitioning
Now, we'll split our imported & cleansed training data into training and test partitions for model building (75-25 split). The result column `classe` is what we're aiming for the other 52 columns to predict. 


```r
set.seed(84738)  #Setting seed to my favorite number
inTrain <- createDataPartition(y=data_cleansing$classe, p=0.75, list=FALSE)
training <- data_cleansing[inTrain,]
testing <- data_cleansing[-inTrain,]
```

#Model Building - Random Forest
We will use the random forest training method in the `caret` package to create a model from the training data.

Because this can be time-intensive process, the routine below will use parallelization of a multi-core system to distribute the workload, using all available cores (aside from one which is left alone for OS processes).

Once a model is built, it will save it locally to a file named `fit.RData`. Thus, in subsequent runs, that file will be looked for and if detected, loaded into memory rather than running the model building process again.


```r
if (file.exists("fit.RData")) {
  #Look for locally saved model file and use it if found
  load(file = "fit.RData")
} else {
  #Set parallelization and train a model
  cluster <- makeCluster(detectCores() - 1)
  registerDoParallel(cluster)
  fitControl <- trainControl(method = "cv", number = 10, allowParallel = TRUE)
  fit <- train(classe ~ ., data=training, method="rf", trControl = fitControl)
  stopCluster(cluster)
  
  #Save the model for use later, rather than retraining each time
  save(fit, file="fit.RData") 
}
```

###Testing the model against the holdback test set
Now to see how this model holdsup to the `testing` data that was held back.

```r
predictions <- predict(fit, testing)
cm <- confusionMatrix(predictions, testing$classe)
cm
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    2    0    0    0
##          B    0  945    6    0    0
##          C    0    2  849   13    0
##          D    0    0    0  790    1
##          E    0    0    0    1  900
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9949          
##                  95% CI : (0.9925, 0.9967)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9936          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9958   0.9930   0.9826   0.9989
## Specificity            0.9994   0.9985   0.9963   0.9998   0.9998
## Pos Pred Value         0.9986   0.9937   0.9826   0.9987   0.9989
## Neg Pred Value         1.0000   0.9990   0.9985   0.9966   0.9998
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1927   0.1731   0.1611   0.1835
## Detection Prevalence   0.2849   0.1939   0.1762   0.1613   0.1837
## Balanced Accuracy      0.9997   0.9971   0.9946   0.9912   0.9993
```

The overall accuracy of 99.49%, with a 95% confidence interval of 99.25 - 99.67, is not too shabby! For the record, this model correctly predicted all 20 of the Quiz test records.

The top factors in this model are

```r
varImp(fit)
```

```
## rf variable importance
## 
##   only 20 most important variables shown (out of 52)
## 
##                      Overall
## roll_belt             100.00
## yaw_belt               79.18
## magnet_dumbbell_z      68.39
## magnet_dumbbell_y      63.26
## pitch_belt             62.67
## pitch_forearm          61.37
## roll_forearm           50.20
## magnet_dumbbell_x      48.82
## magnet_belt_y          44.40
## accel_belt_z           43.78
## accel_dumbbell_y       43.15
## roll_dumbbell          42.84
## magnet_belt_z          42.12
## accel_dumbbell_z       36.92
## roll_arm               33.76
## accel_forearm_x        32.88
## gyros_dumbbell_y       29.23
## yaw_dumbbell           28.89
## total_accel_dumbbell   28.89
## accel_dumbbell_x       28.73
```

