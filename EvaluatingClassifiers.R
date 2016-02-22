library(ggplot2)

#First Load the dataset
data <- read.csv('titanic3.csv')

#We are only interested in some variables for this exercise
data <- data[c('survived', 'pclass','sex', 'age', 'sibsp')] 

#Drop NAs
data <- na.omit(data)

#Partition the data set to Train and Test
library(caret)
intrain<-createDataPartition(y=data$survived,p=0.7,list=FALSE)
titanic.train<-data[intrain,]
titanic.test<-data[-intrain,]

# Apply a Logistic Regression model on train data
titanic.survival.train = glm(survived ~ pclass + sex + pclass:sex + age + sibsp,family = binomial(logit), data = titanic.train)

#Here is the summary
summary(titanic.survival.train)

#Predict the test data and calculate confusion matrix
prediction <- ifelse(predict(titanic.survival.train, titanic.test, type='response') > 0.5, TRUE, FALSE)
confusion  <- table(prediction, as.logical(titanic.test$survived),dnn = c('Predicted','Observed'))


#Now Let's do the same task this time with Random Forest
library(randomForest)
titanic.survival.train.rf = randomForest(as.factor(survived) ~ pclass + sex + age + sibsp, data=titanic.train,ntree=5000, importance=TRUE)
predictionRF <- predict(titanic.survival.train.rf, titanic.test, type='response')
confusionRF  <- table(predictionRF, as.logical(titanic.test$survived),dnn = c('Predicted','Observed'))

#Now we plot ROC curves
library(ROCR)
prob <- predict(titanic.survival.train, titanic.test, type='response')
pred <- prediction(prob,titanic.test$survived)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf, col=rainbow(7), main="ROC curve Titanic (Logistic Regression)", xlab="1 - Specificity", ylab="Sensitivity")    
abline(0, 1) #add a 45 degree line

#Here is the code to create Lift Chart
perf <- performance(pred,"lift","rpp")
plot(perf, main="Lift Chart(Logistic Regression)",xlim=c(0,1))

#ROC and Lift Chart for Random Forest
probRF <- predict(titanic.survival.train.rf, titanic.test, type='prob')
predRF <- prediction(probRF[,2],titanic.test$survived)
perfRF <- performance(predRF, measure = "tpr", x.measure = "fpr")
par(new=TRUE)
plot(perfRF, col=rainbow(5))   

perfRF <- performance(predRF,"lift","rpp")
plot(perfRF, main="Lift Chart(Random Forest)",xlim=c(0,1))

