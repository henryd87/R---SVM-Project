irisData <- read.csv("Iris1.csv",header=TRUE)
install.packages('caret', dependencies = TRUE)
install.packages('skimr', dependencies = TRUE)
library(datasets)
library(caret)
library(skimr)
data(iris)
iris <- datasets::iris
library(RCurl)
View(iris)
summary(iris$Sepal.Length)
sum(is.na(iris))
skim(iris)


plot(iris,col="orange")
#specifying variables
plot(iris$Sepal.Width, iris$Sepal.Length)
#Histogram - series of bars, each bar shows a range and its
#frequency
hist(iris$Sepal.Width,col="purple")

featurePlot(x=iris[,1:4],
            y=iris$Species,
            plot="box",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales=list(x=list(relation="free"),
                        y=list(relation="free")))
#Data splitting for training and testing data.
TrainingIndex = createDataPartition(iris$Species, p=.8, list=FALSE)
TrainingSet <- iris[TrainingIndex,]
TestingSet<-iris[-TrainingIndex,]
#It already randomizes the order for you.

#Create the scatter plot of both Training and Testing Data.
plot(TrainingSet, TestingSet)
Model <- train(Species ~ ., data=TrainingSet,
               method='svmPoly',
               na.action=na.omit,
               preProcess=c("scale","center"),
               trControl=trainControl(method="none"),
               tuneGrid=data.frame(degree=1,scale=1,C=1))
#creates parameters for classification model. Species is the
#class label name, so it can change! data is to TrainingSet.
#na.action means if there's a missing value, we will omit it.
#Remember, a train model uses training set to build our
#predictive model.
#~~~~~~
#The Training Model is a ten-fold cross validation model.
#Means dividing 120 flowers of the 80% subset into 10 subgroups
#with 12 flowers. 90% of those groups will be trained and then test
#on the remaining group. Do this for each of the 10 groups.
#Each group will be tested on, overall.
#~~~~~~~
#*
#*Do this when writing comments next time.*#
#**###
#*Build CV model
Model.cv <- train(Species ~ ., data = TrainingSet,
                  method = "svmPoly",
                  na.action = na.omit,
                  preProcess=c("scale","center"),
                  trControl= trainControl(method="cv", number=10),
                  tuneGrid = data.frame(degree=1,scale=1,C=1)
)
Model.training <-predict(Model, TrainingSet) # Apply model to make prediction on Training set
Model.testing <-predict(Model, TestingSet) # Apply model to make prediction on Testing set
Model.cv <-predict(Model.cv, TrainingSet) # Perform cross-validation

#*confusion matrices allow for us to see
#*false positive, true negatives, etc.
#*
Model.training.confusion <-confusionMatrix(Model.training, TrainingSet$Species)
Model.testing.confusion <-confusionMatrix(Model.testing, TestingSet$Species)
Model.cv.confusion <-confusionMatrix(Model.cv, TrainingSet$Species)

print(Model.training.confusion)
#*This show us that for Versicolor, we have 39 True positives
#*and 1 false positive.
#*Accuracy is ~98%
print(Model.testing.confusion)
print(Model.cv.confusion)

# Feature importance
Importance <- varImp(Model)
plot(Importance)
plot(Importance, col = "red")

#* Analysis
#* Sensitivity and Specificity were >=97.5 for the most part,
#* meaning: Sensitivity is the measure of how many true positives
#* were correctly classified when compared true positives 
#* and false positives. The highest for this is the class Setosa.
#* Specificity refers to how many true negatives were correctly
#* classified among false negatives and true negatives.
