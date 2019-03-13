library(xgboost)
library(data.table)

cat("Read Data....\n")
train<-fread("../data/application_train.csv",select=c('SK_ID_CURR','TARGET','DAYS_BIRTH','DAYS_EMPLOYED'
                                              ,'AMT_CREDIT','AMT_GOODS_PRICE'
                                              ,'EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3'
                                              ,'NAME_INCOME_TYPE','NAME_EDUCATION_TYPE'
                                              ,'OCCUPATION_TYPE','REGION_RATING_CLIENT'
                                              ,'REGION_RATING_CLIENT_W_CITY','ORGANIZATION_TYPE'
                                              ))
test<-fread("../data/application_test.csv",select=c('SK_ID_CURR','DAYS_BIRTH','DAYS_EMPLOYED'
                                            ,'AMT_CREDIT','AMT_GOODS_PRICE'
                                            ,'EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3'
                                            ,'NAME_INCOME_TYPE','NAME_EDUCATION_TYPE'
                                            ,'OCCUPATION_TYPE','REGION_RATING_CLIENT'
                                            ,'REGION_RATING_CLIENT_W_CITY','ORGANIZATION_TYPE'
))

ind<-1:nrow(train)
train_test<-rbind(train,test,fill=T)

cat("Label Encoding....\n")
for (f in names(train_test)) {
  if (class(train_test[[f]])=="character") {
    #cat("VARIABLE : ",f,"\n")
    levels <- sort(unique(train_test[[f]]))
    train_test[[f]] <- as.integer(factor(train_test[[f]], levels=levels))
  }
}


train<-train_test[ind,]
test<-train_test[-ind,]

features<-setdiff(names(train),c('TARGET','SK_ID_CURR'))

cat("Train Model \n")

xgb_params = list(
  eta = 0.1,
  objective = 'binary:logistic',
  eval_metric='auc',
  colsample_bytree=0.7,
  subsample=0.7,
  min_child_weight=10
)

features<-setdiff(names(train),'TARGET')
dtrainmat = xgb.DMatrix(as.matrix(train[,features,with=FALSE]), label=train$TARGET)
dtestmat = xgb.DMatrix(as.matrix(test[,features,with=FALSE]))

xgbmodel<-xgb.train(xgb_params,dtrainmat,nrounds=125,verbose=2)
pred<-predict(xgbmodel,dtestmat)

cat("Submit Predictions\n")
sub<-data.frame(SK_ID_CURR = test$SK_ID_CURR,TARGET=pred)
write.csv(sub,'baseline.csv',row.names = F)
