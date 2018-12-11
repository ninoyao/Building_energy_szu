## Timeserious 
dataori = read.csv("data_complete.csv")
head(dataori)
data_sub1 = dataori[c("Year","Month","Day","Hour","Minute","Cooling_Load")]
install.packages("tseries")
install.packages("forecast")
install.packages("TTR")
library(tseries)
library(forecast)
datacl = data_sub1[c("Cooling_Load")]
##turn the dataset into ts dataset,set strat date and frequency
dataclts <- ts(datacl,start = c(1,2),end = c(12,31),frequency = 48)
class(dataclts)
start(dataclts)
plot(dataclts)

train<- window(dataclts,start = c(1,2),end = c(11,1))
test <- window(dataclts,start = c(11,2),end = c(12,31))
##平均
pred_meanf<- meanf(train,h = 4)
rmse(test, pred_meanf$mean)
##snavie
pred_snavie = snaive(train,h =4)
rmse(pred_snavie$mean,test)
##rwf
pred_rwf<- rwf(train,h = 4)
rmse(pred_rwf$mean,test)
##auto.arima
fit<- auto.arima(train)
accuracy(forecast(fit,h=4),test)
##rmse
rmse = function(pred,test)
{ 
  res<- sqrt(mean((pred-test)^2) )
  res
}
