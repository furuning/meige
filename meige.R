rm(list = ls())
gc()

load("train_test.Rdata")
#roc

fit<-glm(group~ALB+TBIL+UCR+GGT,family = binomial(link = logit),data=train_data )
train_data$trset<- predict(newdata=train_data,fit,"response")
test_data$Vdset<- predict(newdata=test_data,fit,"response")
library(pROC)
library(ggplot2)

train_roc <- roc(train_data$group,train_data$trset);train_roc
test_roc <- roc(test_data$group,test_data$Vdset);test_roc
roc.test(train_roc,test_roc,method="delong")

roc2<-test_roc
roc1<-train_roc

plot(roc1,print.auc=TRUE, print.auc.x=0.4, print.auc.y=0.5,
     auc.polygon=TRUE, auc.polygon.col="#FFFFFF", 
     max.auc.polygon=FALSE, 
     grid=c(0.5, 0.2), grid.col=c("black", "black"),
     print.thres=TRUE, print.thres.cex=0.9, 
     smooth=F, 
     main="Comparison of trainning set and validation set ROC curves", 
     col="#E64B35", 
     legacy.axes=TRUE)

plot.roc(roc2,
         add=T, 
         col="#4DBBD5", 
         print.thres=TRUE, print.thres.cex=0.9, 
         print.auc=TRUE, print.auc.x=0.4,print.auc.y=0.4,
         smooth = F) 

testobj <- roc.test(roc1,roc2)
text(0.7, 0.2, labels=paste("P value =", format.pval(testobj$p.value)), adj=c(0, .5)) 
legend(0.95,0.20, 
       bty = "n", 
       title="",
       legend=c("training set 0.930 (95%IC,0.913-0.948)", "validation set 0.914 (95%IC,0.885-0.942" ), # 添加分组
       col=c("#E64B35","#4DBBD5"), 
       lwd=2) 

#nomograph
library(regplot)

glm<-glm(group~ALB+GGT+ TBIL +UCR,data=train_data,family = binomial())
regplot(glm,
        observation = train_data[2,],
        points = T,
        center=T,
        odds = F,
        showP = T,
        rank ='sd',
        clickable = F,
        title='Nomgram',
        interval ="confidence"
)


#Calibration curve
dd <- datadist(train_data)
options(datadist="dd")

lrm.fit <- lrm(group ~ ALB+GGT+ TBIL +UCR, 
               data = train_data, x=TRUE,y=TRUE)

lrm.cal <- calibrate(lrm.fit, method = "boot", B=1000)
summary(lrm.cal)

plot(lrm.cal) 

par(mar=c(6,6,3,3)) 
plot(lrm.cal,    
     xlim = c(0,1),
     ylim = c(0,1),
     xlab = "Predicted Probability", 
     ylab="Actual Probability", 
     xaxs = "i", 
     yaxs = "i", 
     legend =FALSE,  
     subtitles = FALSE, 
     cex=1.5, 
     cex.axis=1,
     cex.lab=1 )

abline(0, 1, col="#0066CC", lty=2, lwd=2)
lines(lrm.cal[,c("predy","calibrated.orig")],type="l",lwd=2,col="#009966")
lines(lrm.cal[,c("predy","calibrated.corrected")],type="l",lwd=2,col="#f7921d")
legend(x=0.5,y=0.4,
       legend=c("Ideal","Apparent","Bias-corrected"),
       lty = c(2,1,1),
       lwd = c(2,2,2),
       col = c("#0066CC","#009966","#f7921d"),
       bty="n",
       cex=1)

save(train_data,test_data,file = "train_test.Rdata")



library(tidyverse)
library(rms) 
library(ggplot2)
library(ggsci)
library(dcurves)
# DCA curve
train_data<-read.csv("./zyjtd1.csv")
train_data<-train_data[,c(1,5,7,10,14:15)]

dcurves::dca(formula = group ~ trset, 
             label = list(trset = "train_data"),
             data = train_data) %>%
  plot(smooth = TRUE) + 
  ggplot2::labs(x = "Threshold probability")

test_data<-read.csv("./zyjvd1.csv")
test_data<-test_data[,c(1,5,7,10,14:15)]

dcurves::dca(formula = group ~ Vdset, 
             label = list(trset = "test_data"),
             data = test_data) %>%
  plot(smooth = TRUE) + 
  ggplot2::labs(x = "Threshold probability")





