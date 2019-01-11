library(dummies)
library(gains)
library(caret)
library(dplyr)
library(e1071)
library(leaps)
library(corrplot)
library("ggpubr")
library(randomForest)
library(MASS)
library(RColorBrewer)
library(rpart)
library(rpart.plot)
library(kernlab)
library(readr)


#Importing dataset
DS_173<- read.csv("churn.csv")
str(DS_173)

#removing range columns
DS_173_WORange = DS_173[,!grepl("*_Range",names(DS_173))]


#Removing columns with more than 55% missing values and Removing variables which are not defined in the data dictionary
#Calculating percentage of missing values in each variable for numeric values
sapply(DS_173_WORange, function(col) sum(is.na(col))/length(col) * 100)
DS_173_WORange = subset(DS_173_WORange, select=-c(crtcount,rmcalls,rmmou, rmrev, REF_QTY, tot_ret, tot_acpt, educ1, retdays,div_type,occu1,mailordr,wrkwoman,mailresp,children,cartype,HHstatin,mailflag,solflag,proptype
                                           ,pcowner,kid0_2,kid3_5,kid6_10,kid11_15,kid16_17,infobase,csa,last_swap,crclscod,ethnic,dualband,hnd_webcap,dwllsize,prizm_social_one,creditcd,marital, car_buy,refurb_new))

#Deleting the missing values
DS_WO_NA <- na.omit(DS_173_WORange)

#Calculate correlation matrix
set.seed(7)
library(mlbench)

DS_COR <- subset(DS_173_WORange,select=-c(area,asl_flag,new_cell,ownrent, dwlltype))
corr_matrix<-cor(DS_COR, method = "pearson", use = "complete.obs")
# summarize the correlation matrix
print(corr_matrix)
library(corrplot)
opar2 <- par(no.readonly = TRUE)
corrplot(corr_matrix,method = "circle",tl.cex = 0.5,tl.col = "black",number.cex = 0.55,bg = "grey14",
         addgrid.col = "gray50", tl.offset = 2,col = colorRampPalette(c("blue1","ivory2","firebrick2"))(100))
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(corr_matrix, cutoff=0.75,names = TRUE)
# print indexes of highly correlated attributes
print(highlyCorrelated)

#removing columns from results from correlation matrix (threshold = 0.75)
DS_After_COR <- subset(DS_WO_NA, select=-c(ovrmou_Mean,ovrrev_Mean,vceovr_Mean,plcd_vce_Mean,comp_vce_Mean,mou_cvce_Mean,mou_rvce_Mean,owylis_vce_Mean,peak_vce_Mean,peak_dat_Mean,
                                    mou_peav_Mean,mou_pead_Mean,opk_vce_Mean,opk_dat_Mean,mou_opkv_Mean,mou_opkd_Mean,drop_blk_Mean,attempt_Mean,complete_Mean,avgrev,
                                    avgmou,avgqty,avg3mou,avg3qty,avg3rev,avg6mou,avg6qty,avg6rev))

#Summary statistics 
DS_Summary <- subset(DS_WO_NA, select=c(eqpdays,age1,totcalls,totrev,rev_Mean,blck_vce_Mean,actvsubs,months,custcare_Mean,drop_vce_Mean))
summary(DS_Summary)

#dividing dataset into training and validation
set.seed(2)
numberOfRows <- nrow(DS_After_COR)
train.index <- sample(numberOfRows, numberOfRows*0.70)
DS_train <- DS_After_COR[train.index,]
DS_valid <- DS_After_COR[-train.index,]


###basic logistic regression with all variables
Logistic_reg<- glm(churn ~., data = DS_train, family = binomial)
summary(Logistic_reg)
pred_test_log<- predict(Logistic_reg, newdata = DS_valid, type ="response")
#Model Accuracy 
#with 0.4 threshold
confusionMatrix(table(predict(Logistic_reg, newdata = DS_valid, type="response") >= 0.4, DS_valid$churn == 1))
#with 0.6 threshold
confusionMatrix(table(predict(Logistic_reg, newdata = DS_valid, type="response") >= 0.6, DS_valid$churn == 1))
#with 0.5 threshold
confusionMatrix(table(predict(Logistic_reg, newdata = DS_valid, type="response") >= 0.5, DS_valid$churn == 1))
##Lift Chart
gain_log <- gains(DS_valid$churn, pred_test_log, groups=10)
# plot lift chart
plot(c(0,gain_log$cume.pct.of.total*sum(DS_valid$churn))~c(0,gain_log$cume.obs), 
     xlab="# cases", ylab="Cumulative", main="", type="l")
lines(c(0,sum(DS_valid$churn))~c(0, dim(DS_valid)[1]), lty=2)
# compute deciles and plot decile-wise chart
heights <- gain_log$mean.resp/mean(DS_valid$churn)
midpoints <- barplot(heights, names.arg = gain_log$depth, ylim = c(0,9), 
                     xlab = "Percentile", ylab = "Mean Response", main = "Decile-wise lift chart")
# add labels to columns
text(midpoints, heights+0.5, labels=round(heights, 1), cex = 0.8)


###Logistic regression by conidering important variables found out using ROC
##important variable calculation
set.seed(7)
# prepare training scheme
control <- trainControl(method="repeatedcv", number=2, repeats=2)
#Converting churn to factor
DS_train_1<-(DS_train)
DS_train_1$churn<-as.factor(DS_train_1$churn)
DS_valid_1<-(DS_valid)
DS_valid_1$churn<-as.factor(DS_valid_1$churn)
# train the model
model <- train(churn ~., data= DS_train_1, method="lvq", preProcess="scale", trControl=control)
# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)
##Logistic regression
Logistic_reg_imp_Var<- glm(churn ~eqpdays + hnd_price + change_mou + totmrc_Mean + mou_Mean + mouiwylisv_Mean + iwylis_vce_Mean + custcare_Mean   +
                             mouowylisv_Mean + ccrndmou_Mean + cc_mou_Mean + lor + age1 + unan_vce_Mean  + recv_vce_Mean   + uniqsubs + inonemin_Mean   +
                             callwait_Mean + asl_flag  + area, data = DS_train_1, family = binomial)
summary(Logistic_reg_imp_Var)
pred_test_log<- predict(Logistic_reg_imp_Var, newdata = DS_valid_1, type ="response")
##Model Accuracy
#with threshold=0.4
confusionMatrix(table(predict(Logistic_reg_imp_Var, newdata = DS_valid_1, type="response") >= 0.4, DS_valid_1$churn == 1))
#with threshold=0.6
confusionMatrix(table(predict(Logistic_reg_imp_Var, newdata = DS_valid_1, type="response") >= 0.6, DS_valid_1$churn == 1))
#with threshold=0.5
confusionMatrix(table(predict(Logistic_reg_imp_Var, newdata = DS_valid_1, type="response") >= 0.5, DS_valid_1$churn == 1))
## Lift Chart 
gain_log <- gains(DS_valid_1$churn, pred_test_log, groups=10)
# plot lift chart
plot(c(0,gain_log$cume.pct.of.total*sum(DS_valid$churn))~c(0,gain_log$cume.obs),
     xlab="# cases", ylab="Cumulative", main="", type="l")
lines(c(0,sum(DS_valid$churn))~c(0, dim(DS_valid)[1]), lty=2)



### Logistic Regression with sTEPWISE- backward selection 
model <- glm(churn ~., data = DS_train, family = binomial) %>%
stepAIC(trace = FALSE)
summary(model)
#prediction
pred_test_Step<- predict(model, newdata = DS_valid, type ="response")
# Model accuracy
#with threshold 0.4
confusionMatrix(table(predict(model, newdata = DS_valid, type="response") >= 0.4, DS_valid$churn == 1))
#with threshold 0.6
confusionMatrix(table(predict(model, newdata = DS_valid, type="response") >= 0.6, DS_valid$churn == 1))
#with threshold 0.5
confusionMatrix(table(predict(model, newdata = DS_valid, type="response") >= 0.5, DS_valid$churn == 1))
##Lift and Gains chart
gain <- gains(DS_valid$churn, pred_test_Step, groups=10)
# plot lift chart
plot(c(0,gain$cume.pct.of.total*sum(DS_valid$churn))~c(0,gain$cume.obs), 
     xlab="# cases", ylab="Cumulative", main="", type="l")
lines(c(0,sum(DS_valid$churn))~c(0, dim(DS_valid)[1]), lty=2)
# compute deciles and plot decile-wise chart
heights <- gain$mean.resp/mean(DS_valid$churn)
midpoints <- barplot(heights, names.arg = gain$depth, ylim = c(0,9), 
                     xlab = "Percentile", ylab = "Mean Response", main = "Decile-wise lift chart")
# add labels to columns
text(midpoints, heights+0.5, labels=round(heights, 1), cex = 0.8)



###Decision tree
# create a classification tree
library(rattle)
# build a deeper classification tree
max.ct <- rpart(churn ~ ., data = DS_train, method = "class", cp = 0, minsplit = 1, maxdepth = 5)
# count number of leaves
length(max.ct$frame$var[max.ct$frame$var == "<leaf>"])
# plot tree
prp(max.ct, type = 1, extra = 1, under = TRUE, split.font = 0.5, varlen = -10, 
    box.col=ifelse(max.ct$frame$var == "<leaf>", 'gray', 'white'))
fancyRpartPlot(max.ct)
# classify records in the validation data.
# set argument type = "class" in predict() to generate predicted class membership.
max.pred1 <- predict(max.ct, DS_valid, type = "class")
# generate confusion matrix for training data
confusionMatrix(max.pred1, as.factor(DS_valid$churn))

# pruning of the tree
printcp(max.ct)
plotcp(max.ct)
ptree<- prune(max.ct, cp= max.ct$cptable[which.min(max.ct$cptable[,"xerror"]),"CP"])
prp(ptree)
fancyRpartPlot(ptree, uniform=TRUE, main="Pruned Classification Tree")
max.pred <- predict(ptree, DS_valid, type = "class")
#generate confusion matrix for training data
confusionMatrix(max.pred, as.factor(DS_valid$churn))
printcp(ptree)


####Random Forest
model_RF_5 <- randomForest(churn~ . , data = DS_train_1,proximity = F, do.trace = T, mtry = 5,ntree=300)
summary(model_RF_5)
print(model_RF_5)
plot(model_RF_5)
which.min(model_RF_5$err.rate[, 1])

#on the basis of above output
model_RF_new_5 <- randomForest(churn~ . , data = DS_train_1,proximity = F, do.trace = T, mtry = 5,ntree=294)
pred_test_RF_5<- predict(model_RF_new_5, newdata = DS_valid_1, type ="response")
table(pred_test_RF_5,DS_valid_1$churn)
confusionMatrix(pred_test_RF_5,DS_valid_1$churn)
varImpPlot(model_RF_new_5)


#random forest with 20 variables at a time
model_RF_20 <- randomForest(churn~ . , data = DS_train_1,proximity = F, do.trace = T, mtry = 20,ntree=500)
summary(model_RF_20)
print(model_RF_20)
plot(model_RF_20)
which.min(model_RF_20$err.rate[, 1])

#on the basis of above output
model_RF_new_20 <- randomForest(churn~ . , data = DS_train_1,proximity = F, do.trace = T, mtry = 20,ntree=414)
pred_test_RF_20<- predict(model_RF_new_20, newdata = DS_valid_1, type ="response")
table(pred_test_RF_20,DS_valid_1$churn)
confusionMatrix(pred_test_RF_20,DS_valid_1$churn)
varImpPlot(model_RF_new_20)




