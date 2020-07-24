setwd("/Users/phucvu/R")
#setwd("/home/phucvu/R")

df <- read.table("adult.data", 
                 sep = ',', 
                 fill = FALSE, 
                 strip.white = TRUE)

colnames(df) <- c("age", "workclass", "fnlwgt", "education", "education_num", "marital_status", 
                 "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss", 
                 "hours_per_week", "native_country", "salary")

sapply(df, class)

head(df)

library(ggplot2)
library(scales)
library(plyr)
#library(dplyr)
library(data.table)
library(caret)
library(reshape2)

#Decision tree package
library(rpart)
library(rpart.plot)
library(e1071)

#Encoder
library(CatEncoders)

#Classification
library(class)

cat("Nombre de colonne, Nombre de lignes: ", dim(df))

str(df)

summary(df)

colSums(is.na(df))

#colSums(df.isin("?")) ##TODO

categorical <- vector()
for (i in colnames(df)){
    if (is.factor(df[[i]])) {
        categorical <- c(categorical, i)
    }  
}

categorical

sum(is.na(df$salary))

unique(df$salary)

count(df$salary)

ggplot(df, aes(x=salary)) + geom_bar(stat="count") + ggtitle("Distribution de la salaire")

ggplot(df, aes(x=salary, fill=sex)) + 
        geom_bar(stat="count", position=position_dodge()) + 
        ggtitle("Distribution de la salaire/sex")

ggplot(df, aes(x=salary, fill=race)) + 
        geom_bar(stat="count", position=position_dodge()) + 
        ggtitle("Distribution de la salaire/race")

unique(df$workclass)

count(df$workclass)

df$workclass[ df$workclass == "?" ] <- NA
df$workclass = factor(df$workclass)

count(df$workclass)

ggplot(df, aes(x=workclass)) + 
        geom_bar(stat="count") + 
        ggtitle("Distribution de la variable workclass")

ggplot(df, aes(x=workclass, fill=salary)) + 
        geom_bar(stat="count") + 
        ggtitle("Distribution de la salaire/workclass")

unique(df$occupation)

df$occupation[ df$occupation == "?" ] <- NA
df$occupation = factor(df$occupation)

count(df$occupation)

ggplot(df, aes(x=occupation, fill=salary)) + 
        geom_bar(stat="count") + 
        ggtitle("Distribution de la variable occupation/salaire")

unique(df$native_country)

df$native_country[df$native_country == "?"] <- NA
df$native_country = factor(df$native_country)

summary(df$native_country)

ggplot(df, aes(x=native_country)) + 
        geom_bar(stat="count") + 
        ggtitle("Distribution de la variable native_country")

df$native_country <- NULL
categorical <- categorical[categorical != "native_country"]

numerical <- vector()
for (i in colnames(df)){
    if (is.numeric(df[[i]])) {
        numerical <- c(numerical, i)
    }  
}

numerical

head(df[,numerical])

colSums(is.na(df[,numerical])) ##TODO

length(unique(df$age))

ggplot(df, aes(x=age)) + 
    geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                   bins=10, fill="white", color="blue") +
    geom_density(alpha=.2) + # Overlay with transparent density plot
    ggtitle("Distribution de la variable age")

##TODO

ggplot(df, aes(x=salary, y=age)) + 
geom_boxplot() + 
ggtitle("Distribution de la variable age/salary") +
scale_y_continuous(breaks = scales::pretty_breaks(n = 10))

ggplot(df, aes(x=salary, y=age, fill=sex)) + 
  geom_boxplot( position=position_dodge()) + 
    ggtitle("Distribution de la variable salary/sex/age")

ggplot(df) + aes(x=capital_gain, group=salary, fill=salary) + 
  geom_histogram(bins=10, color='black') + ggtitle('Distribution de capital_gain/salary')

sum(df$capital_gain == 0)/length(df$capital_gain)

df$capital_gain <- NULL
numerical <- numerical[numerical != "capital_gain"]

ggplot(df) + aes(x=capital_loss, group=salary, fill=salary) + 
  geom_histogram(bins=10, color='black') + ggtitle('Distribution de capital_loss/salary')

sum(df$capital_loss == 0)/length(df$capital_loss)

df$capital_loss <- NULL
numerical <- numerical[numerical != "capital_loss"]

df$native_country[is.na(df$native_country)] <- names(which.max(table(df$native_country)))
df$workclass[is.na(df$workclass)] <- names(which.max(table(df$workclass)))
df$occupation[is.na(df$occupation)] <- names(which.max(table(df$occupation)))

#df = na.omit(df)

newDF <- df

for (i in categorical) {
    enc <- LabelEncoder.fit(df[[i]])
    newDF[[i]] <- transform(enc, df[[i]])
}

head(newDF)

#Creation de la matrice corrélation
newDF <- round(cor(newDF),2)

#Melt data to bring the correlation values in two axis
melted_data <- melt(newDF)
#head(melted_data)

ggheatmap = ggplot(melted_data, aes(Var2, Var1, fill = value)) + geom_tile(color = "white") + scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, limit = c(-1, 1), space = "Lab", name = "Correlation \n Rate") + theme_minimal() + coord_fixed()
ggheatmap + geom_text(aes(Var2, Var1, label = value), color = "black", size = 2) + theme(axis.title.x = element_blank(), axis.title.y = element_blank())

df = subset(df, select = -c(relationship, education))

categorical <- categorical[categorical != "relationship"]
categorical <- categorical[categorical != "education"]

#Splitter la cible
#categorical <- categorical[categorical != "salary"]

testDf <- df

# Random sample indexes
train_index <- sample(1:nrow(testDf), 0.7 * nrow(testDf))
test_index <- setdiff(1:nrow(testDf), train_index)

# Build X_train, y_train, X_test, y_test
X_train <- testDf[train_index, -10]
y_train <- testDf[train_index, "salary"]

X_test <- testDf[test_index, -10]
y_test <- testDf[test_index, "salary"]

dim(X_train)

dim(X_test)

predictData <- function(model, feature_test, target_test) {
    pred <- predict(model, feature_test, type = "class")
    mc <- table(pred, target_test)
#    print(confusionMatrix(pred, y_test))
    print(mc)
    return(mc)
}

buildMC <- function(table, colNames, rowNames) {
    colLeft <- c(table[1,1], table[2,1])
    colRight <- c(table[1,2], table[2,2])
    mc <- data.frame(colLeft, colRight, row.names=rowNames)
    colnames(mc) <- colNames 
    return (mc)
}

statMC <- function(mc) {
    accuracy <- sum(diag(mc)/(sum(rowSums(mc)))) * 100
    sensitivity <- mc[1,1]/(mc[1,1] + mc[2,1]) #True positive rate
    specificity <- mc[2,2]/(mc[1,2] + mc[2,2])
    pos_pred <- mc[1,1]/(mc[1,1] + mc[1,2])
    neg_pred <- mc[2,2]/(mc[2,1] + mc[2,2])
    #arithmetic_mean <- (recall + true_negative)/2
    #geometric_mean <- sqrt(recall * true_negative)
    
    cat("Accuracy: ", accuracy, "\n")
    cat("Sensitivity: ", sensitivity, "\n") #Precision nombre de positive bien classifié
    cat("Specificity: ", specificity, "\n") #nombre de négative bien classifié
    #cat("Recall: ", recall, "\n")
    cat("Pos Pred Value: ", pos_pred, "\n")
    cat("Neg Pred Value: ", neg_pred, "\n")
    #cat("Arithmetic Mean: ", arithmetic_mean, "\n")
    #cat("Geometric Mean: ", geometric_mean, "\n")
    
    #info <- c(false_positive, recall)
    #return(info)
}

treeModel <- rpart(y_train~., data=X_train, method="class")
treePredict <- predict(treeModel, X_test, type = "class")
treeModel
treeModelImp <- data.frame(imp = treeModel$variable.importance)
treeModelImp

ggplot(treeModelImp, aes(x=reorder(row.names(treeModelImp), -imp), y=imp)) + 
  geom_bar(stat="identity") + xlab('Features') + ylab('Importance') + geom_text(aes(label=round(imp,digits=2)), vjust=0)
rpart.plot(treeModel)

validate_test <- predictData(treeModel, X_test, y_test)

mc <- validate_test
infoTreeBase <- statMC(mc)

treeModelPre <- rpart(y_train~., data = X_train, method = "class", 
                   control = rpart.control(cp = 0, maxdepth = 6,minsplit = 100))

treeModelPreImp <- data.frame(imp = treeModelPre$variable.importance)
treeModelPreImp

ggplot(treeModelPreImp, aes(x=reorder(row.names(treeModelPreImp), -imp), y=imp)) + 
  geom_bar(stat="identity") + xlab('Features') + ylab('Importance') + geom_text(aes(label=round(imp,digits=2)), vjust=0)

rpart.plot(treeModelPre)

validate_test_pre <- predictData(treeModelPre, X_test, y_test)

mc <- validate_test_pre
infoTreePre <- statMC(mc)

library(randomForest)

modelRF = randomForest(y_train~., data = X_train, importance = TRUE)

modelRFImp <- data.frame(importance(modelRF))
modelRFImp

ggplot(modelRFImp, aes(x=reorder(row.names(modelRFImp), -MeanDecreaseGini), y=MeanDecreaseGini)) + 
  geom_bar(stat="identity") + xlab('Features') + ylab('Importance')

print(modelRF)

validate_test <- predictData(modelRF, X_test, y_test)

mc <- validate_test
infoRF <- statMC(mc)

encoder <- dummyVars(~., data = X_train)
X_train_test <- data.frame(predict(encoder, newdata = X_train)) 
X_test_test <- data.frame(predict(encoder, newdata = X_test)) 

head(X_train_test)

# Min-Max Scaling function
#MinMaxScaling <- function(x){
#  return((x-min(x))/(max(x)-min(x)))
#}

#for (i in numerical) {
#    X_train_test[[i]] <- MinMaxScaling(X_train_test[[i]])
#}

pr <- knn(X_train_test,X_test_test,cl=y_train,k=10)
 
##create confusion matrix
tab <- table(pr,y_test)

##this function divides the correct predictions by total number of predictions that tell us how accurate teh model is.
infoKNN <- statMC(tab)

i=1
k.optm=1
for (i in 1:28){ 
    knn.mod <- knn(train=X_train_test,test=X_test_test,cl=y_train,k=i)
    k.optm[i] <- 100 * sum(y_test == knn.mod)/NROW(y_test)
    k=i  
    cat(k,'=',k.optm[i],'\n')
}

plot(k.optm, type="b", xlab="K- Value",ylab="Taux d'accuracy")

valDf <- read.table("adult.test", 
                 sep = ',', 
                 fill = FALSE, 
                 strip.white = TRUE)

colnames(valDf) <- c("age", "workclass", "fnlwgt", "education", "education_num", "marital_status", 
                 "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss", 
                 "hours_per_week", "native_country", "salary")

head(valDf)

str(valDf)

cat("Element manqué dans le dataset: ", sum(is.na(valDf)), "\n")

valDf$native_country <- NULL
valDf$capital_gain <- NULL
valDf$capital_loss <- NULL

valDf$workclass[ valDf$workclass == "?" ] <- NA
valDf$workclass = factor(valDf$workclass)

valDf$occupation[ valDf$occupation == "?" ] <- NA
valDf$occupation = factor(valDf$occupation)

colSums(is.na(valDf))

#Supprimer les données null dans le dataset (OPTIONELLE)

cat("Taille de dataset avant la suppresion ", dim(valDf), "\n")

valDf = na.omit(valDf)

cat("Taille de dataset apres la suppresion ", dim(valDf), "\n")

#valDf$native_country[is.na(valDf$native_country)] <- names(which.max(table(valDf$native_country)))
valDf$workclass[is.na(valDf$workclass)] <- names(which.max(table(valDf$workclass)))
valDf$occupation[is.na(valDf$occupation)] <- names(which.max(table(valDf$occupation)))

valDf = subset(valDf, select = -c(relationship, education))

head(valDf)

testDf <- valDf

head(testDf)

X_validation <- testDf[, -10]
y_validation <- testDf[, "salary"]

dim(X_validation)

validate_test_set <- predictData(treeModelPre, X_validation, y_validation)
mc <- validate_test_set
infoDTVal <- statMC(mc)

for (i in colnames(X_train)) {
    levels(X_validation[[i]]) <- levels(X_train[[i]])
}

str(X_validation)

str(X_test)

validate_test_set <- predictData(modelRF, X_validation, y_validation)
mc <- validate_test_set
infoRFVal <- statMC(mc)

X_validation_test <- data.frame(predict(encoder, newdata = X_validation)) 

pr <- knn(X_train_test,X_validation_test,cl=y_train,k=25)
 
##create confusion matrix
tab <- table(pr,y_validation)

##this function divides the correct predictions by total number of predictions that tell us how accurate teh model is.
infoKNN <- statMC(tab)

encoder <- dummyVars(" ~ .", data=df[, names(df) != "salary"])
dfCV <- data.frame(predict(encoder, newdata = df))
dfCV$salary <- df$salary

head(dfCV)

train_control <- trainControl(method="cv", number=10)
# Fit Decision Tree
model <- train(salary~., data=dfCV, trControl=train_control, method="rpart")
# Summarise Results
print(model)

#n <- nrow(dfCV) #Nb observations
#K <- 10 #10-validation croisée
#taille <- n%/%K #Taille chaque bloc
#set.seed(5) #Obtenir la meme séquence tout le temps
#alea <- runif(n) 
#rang <- rank(alea)
#bloc <- (rang-1)%/%taille + 1#Numeroter de bloc
#bloc <- as.factor(bloc)
#print(summary(bloc))

#class(bloc)

#for (k in 1:K) {
#    treeModel <- rpart(salary~., data=dfCV[bloc != k], method="class")
#    treePredict <- predict(treeModel, dfCV[bloc == k], type = "class")
#    matrixConfus <- table(dfCV$salary[bloc==k], treePredict)
#    mc
    #rpart.plot(treeModel)
#}

#dd1 <- data.frame(FP = infoDT[1], TP = infoDT[2])
#dd2 <- data.frame(FP = infoKNN[1], TP = infoKNN[2])

#g <- ggplot() + 
#  geom_line(data = dd1, aes(x = FP, y = TP, color = 'Decision Tree')) + 
#  geom_line(data = dd2, aes(x = FP, y = TP, color = 'K-NN')) +
#  geom_segment(aes(x = 0, xend = 1, y = 0, yend = 1)) +
#  ggtitle('ROC Curve') + 
#  labs(x = 'False Positive Rate', y = 'True Positive Rate') #
#g +  scale_colour_manual(name = 'Classifier', values = c('Decision Tree'='#E69F00', 
#                                               'K-NN'='#56B4E9'))


