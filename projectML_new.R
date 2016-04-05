##Load necessary packages
library(caret)
library(forecast)
##Download data
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile="training.csv")
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile="testing.csv")
modelData<-read.csv("training.csv")
predictData<-read.csv("testing.csv")
##Set seed and split the data
set.seed(133)
inTrain = createDataPartition(modelData$classe, p = 0.6)[[1]]
training = modelData[ inTrain,]
testing = modelData[-inTrain,]

##Filtering just accelerator variables and the output variable
temp<-grep("(accel|classe)",colnames(training))
training<-training[,temp]
testing<-testing[,temp]
predictData<-predictData[,temp]

summary(training)
for (i in seq_along(colnames(training))) {
  print(colnames(training)[i])
  print(mean(training[,i], na.rm=TRUE))
  print(sd(training[,i], na.rm=TRUE))
  }
##Looking at the training data summary as well as mean and standard dev for each variable, I decide to do two things:
##1. Remove variables ‘var_total_accel_belt’, ‘var_accel_arm’, ‘var_accel_dumbbell’, ‘var_accel_forearm’, 
##because lot of these values are equal to NA
temp<-grep("(var)",colnames(training))
training<-training[,-temp]
testing<-testing[,-temp]
predictData<-predictData[,-temp]
##2. Standarize the data, because for variables "accel_belt_x","accel_belt_z", "accel_arm_x", "accel_arm_y","accel_arm_z", "accel_dumbbell_x","accel_dumbbell_z",  "accel_forearm_x",  "accel_forearm_z"
##standard deviation is much higher than mean
preObj<-preProcess(training[,-17], method=c("center","scale"))
training_sd<-predict(preObj, training[,-17])
testing_sd<-predict(preObj, testing[,-17])
predict_sd<-predict(preObj, predictData)
training_sd<-cbind(training_sd,classe=training$classe)
testing_sd<-cbind(testing_sd,classe=testing$classe)

##Additionally, I applied principal components - because variables in col 1,3,4 are highly correlated (cor>0.8) and I want to create two variables instead of these three
M<-abs(cor(training_sd[,-17]))
diag(M)<-0
which(M>0.8,arr.ind=TRUE)

preProc<-preProcess(training_sd[,c(1,3,4)],method="pca",pcaComp=2)
trainPC<-predict(preProc, training_sd[,c(1,3,4)])
trainPC<-cbind(training_sd[,-c(1,3,4)],trainPC)
testPC<-predict(preProc, testing_sd[,c(1,3,4)])
testPC<-cbind(testing_sd[,-c(1,3,4)],testPC)
predictPC<-predict(preProc, predict_sd[,c(1,3,4)])
predictPC<-cbind(predict_sd[,-c(1,3,4)],predictPC)

## I tried several models relatively easy to compute - i.e. linear discriminant analysis, CART and bagged CART.
##Bagged CART provided the best accuracy on testing sample.
modFit_treebag<-train(factor(classe) ~ .,method="treebag",data=trainPC)

##Checking how model performs on testing sample
pred_treebag<-predict(modFit_treebag, testPC)
confusionMatrix(testPC$classe,pred_treebag)
##Predicting outcome on additional sample (testing.csv uploaded in the beggining)
predicted<-predict(modFit_treebag, predictPC)
View(data.frame(predicted,seq_along(predictData[,1]) ))

##Calculating out-of-sample error - in other words, how much percentage of predictions were incorrect on the testing sample
print(sum(pred_treebag!=testPC$classe)/length(testPC$classe))