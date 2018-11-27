#Task: 24-hour cooling load prediction profile
#Considering that only previous 24hr records are useful for prediction
#Two general types: static and dynamic, deciding whether iterative process is needed when doing prediction
#Feature type:
#(1) Raw (both static and dynamic version)
#(2) Statistical features usign max, min, mean, std, for past 48 (both static and dynamic versions) or mean of recent 6hrs with 2-hr window (only dynamic version) 
#(3) Structural features using DFT (both static and dynamic version)
#(4) AE features using autoencoders (both static and dynamic version)
#In total, there will be 10 sets of features (including 1 basic)

library(foreach)
library(doMC)
library(plyr)
registerDoMC(cores = 2)
library(ggplot2)
library(forecast)
library(caret)
library(randomForest)
library(kernlab)
library(glmnet)
library(mxnet)
library(gbm)
library(dummies)
library(xgboost)
library(h2o)
localH2O <- h2o.init(nthreads = -1, max_mem_size = '2g', ip = "127.0.0.1", port = 54321)

####################################################################################
####################################################################################
#Setting parameters for feature extraction
time_lag <- 48
horizon <- 48
window_size <- 4 #For extracting recent statistical features per 2hr

ft_ext <- foreach(i = unique(data_upd$label), .combine = rbind) %dopar% {
  #Select the data considering certain label, i.e., obs are continuous
  data_temp <- data_upd[which(data_upd$label == i),]
  
  ft_temp <- foreach(j = (time_lag+1):(dim(data_temp)[1]), .combine = rbind) %dopar% {
    #Extract the current day_label
    label_current <- data_temp[j,'Day_label'] 
    
    #Static lag
    df_lag <- data_temp[which(data_temp$Day_label %in% c((label_current-(time_lag/48)):(label_current-1))),]
    
    #Dynamic lag
    df_lag_dynamic <- data_temp[(j-time_lag):(j-1),]
    df_lag_dynamic$recent_id <- rep((time_lag/window_size):1, each = window_size)
    df_lag_dynamic$dynamic_id <- rep(1:(time_lag/48), each = 48)
    
    #Current obs
    df_current <- data_temp[j,]
    
    #Static raw 
    ft_raw <- c(df_lag[,'T_Ambient'], df_lag[,'RH_Ambient'], df_lag[,'Cooling_Load'])
    names(ft_raw) <- c(paste0('T_hist_',1:time_lag,'_RAW'),
                       paste0('RH_hist_',1:time_lag,'_RAW'),
                       paste0('Load_hist_',1:time_lag,'_RAW'))
    
    #Dynamic raw 
    ft_raw_dynamic <- c(df_lag_dynamic[,'T_Ambient'], df_lag_dynamic[,'RH_Ambient'], df_lag_dynamic[,'Cooling_Load'])
    names(ft_raw_dynamic) <- c(paste0('T_hist_',1:time_lag,'_RAWdy'),
                               paste0('RH_hist_',1:time_lag,'_RAWdy'),
                               paste0('Load_hist_',1:time_lag,'_RAWdy'))
    
    #Dynamic with last observation
    ft_last <- c(df_lag_dynamic[dim(df_lag_dynamic)[1],'T_Ambient'], 
                 df_lag_dynamic[dim(df_lag_dynamic)[1],'RH_Ambient'], 
                 df_lag_dynamic[dim(df_lag_dynamic)[1],'Cooling_Load'])
    names(ft_last) <- c('T_hist_LASTdy','RH_hist_LASTdy','Load_hist_LASTdy')
    
    #Statistical feature extraction
    #Get a mean value for past 2-hr, 2~4-hr, and 4~6-hr 
    ft_stat <- c(
      as.vector(t(daply(.data = df_lag, .variables = 'Day_label', .fun = function(x) {c(Min = min(x$T_Ambient), 
                                                                                        Max = max(x$T_Ambient),
                                                                                        Mean = mean(x$T_Ambient),
                                                                                        SD = sd(x$T_Ambient))}))),
      as.vector(t(daply(.data = df_lag, .variables = 'Day_label', .fun = function(x) {c(Min = min(x$RH_Ambient), 
                                                                                        Max = max(x$RH_Ambient),
                                                                                        Mean = mean(x$RH_Ambient),
                                                                                        SD = sd(x$RH_Ambient))}))),
      as.vector(t(daply(.data = df_lag, .variables = 'Day_label', .fun = function(x) {c(Min = min(x$Cooling_Load), 
                                                                                        Max = max(x$Cooling_Load),
                                                                                        Mean = mean(x$Cooling_Load),
                                                                                        SD = sd(x$Cooling_Load))})))
    )
    names(ft_stat) <- c(paste0(rep('T_hist_',4*time_lag/48), rep(c('min','max','mean','sd'), time_lag/48),'_',rep(1:(time_lag/48), each = 4),'_STAT'),
                        paste0(rep('RH_hist_',4*time_lag/48), rep(c('min','max','mean','sd'), time_lag/48),'_',rep(1:(time_lag/48), each = 4), '_STAT'),
                        paste0(rep('Load_hist_',4*time_lag/48), rep(c('min','max','mean','sd'), time_lag/48),'_',rep(1:(time_lag/48), each = 4), '_STAT'))
    
    #Dynamic version for STAT, denoted as STATdy
    ft_stat_dynamic <- c(
      as.vector(t(daply(.data = df_lag_dynamic, .variables = 'dynamic_id', .fun = function(x) {c(Min = min(x$T_Ambient),
                                                                                                 Max = max(x$T_Ambient),
                                                                                                 Mean = mean(x$T_Ambient),
                                                                                                 Sd = sd(x$T_Ambient))}))),
      as.vector(t(daply(.data = df_lag_dynamic, .variables = 'dynamic_id', .fun = function(x) {c(Min = min(x$RH_Ambient),
                                                                                                 Max = max(x$RH_Ambient),
                                                                                                 Mean = mean(x$RH_Ambient),
                                                                                                 Sd = sd(x$RH_Ambient))}))),
      as.vector(t(daply(.data = df_lag_dynamic, .variables = 'dynamic_id', .fun = function(x) {c(Min = min(x$Cooling_Load),
                                                                                                 Max = max(x$Cooling_Load),
                                                                                                 Mean = mean(x$Cooling_Load),
                                                                                                 Sd = sd(x$Cooling_Load))})))
    )
    names(ft_stat_dynamic) <- c(paste0(rep('T_hist_',4*time_lag/48), rep(c('min','max','mean','sd'), time_lag/48),'_',rep(1:(time_lag/48), each = 4),'_STATdy'),
                                paste0(rep('RH_hist_',4*time_lag/48), rep(c('min','max','mean','sd'), time_lag/48),'_',rep(1:(time_lag/48), each = 4), '_STATdy'),
                                paste0(rep('Load_hist_',4*time_lag/48), rep(c('min','max','mean','sd'), time_lag/48),'_',rep(1:(time_lag/48), each = 4), '_STATdy'))
    
    #Dynamic statistical features for past 2-hr, 2~4hr, 4-6hr
    ft_stat_dynamic_recent <- c(
      mean(df_lag_dynamic[which(df_lag_dynamic$recent_id == 3), 'T_Ambient']),
      mean(df_lag_dynamic[which(df_lag_dynamic$recent_id == 2), 'T_Ambient']),
      mean(df_lag_dynamic[which(df_lag_dynamic$recent_id == 1), 'T_Ambient']),
      mean(df_lag_dynamic[which(df_lag_dynamic$recent_id == 3), 'RH_Ambient']),
      mean(df_lag_dynamic[which(df_lag_dynamic$recent_id == 2), 'RH_Ambient']),
      mean(df_lag_dynamic[which(df_lag_dynamic$recent_id == 1), 'RH_Ambient']),
      mean(df_lag_dynamic[which(df_lag_dynamic$recent_id == 3), 'Cooling_Load']),
      mean(df_lag_dynamic[which(df_lag_dynamic$recent_id == 2), 'Cooling_Load']),
      mean(df_lag_dynamic[which(df_lag_dynamic$recent_id == 1), 'Cooling_Load'])
    )
    names(ft_stat_dynamic_recent) <- c(paste0('T_mean_',1:3,'_RECENTdy'),paste0('RH_mean_',1:3,'_RECENTdy'),paste0('Load_mean_',1:3,'_RECENTdy'))
    
    #Discrete Fourier Transformation (DFT) features
    ft_dft <- c(
      as.vector(t(daply(.data = df_lag, .variables = 'Day_label', .fun = function(x) {
        mag <- Mod(fft(x$T_Ambient))
        mag_upd <- mag[1:round(length(mag)/2)]
        order(mag_upd, decreasing = T)[1:4]}))),
      as.vector(t(daply(.data = df_lag, .variables = 'Day_label', .fun = function(x) {
        mag <- Mod(fft(x$RH_Ambient))
        mag_upd <- mag[1:round(length(mag)/2)]
        order(mag_upd, decreasing = T)[1:4]}))),
      as.vector(t(daply(.data = df_lag, .variables = 'Day_label', .fun = function(x) {
        mag <- Mod(fft(x$Cooling_Load))
        mag_upd <- mag[1:round(length(mag)/2)]
        order(mag_upd, decreasing = T)[1:4]})))
    )
    names(ft_dft) <- c(
      paste0(rep('T_hist_',4*time_lag/48),'T',rep(1:(time_lag/48), each = 4), '_', rep(1:4, time_lag/48),'_DFT'),
      paste0(rep('RH_hist_',4*time_lag/48),'T',rep(1:(time_lag/48), each = 4), '_', rep(1:4, time_lag/48),'_DFT'),
      paste0(rep('Load_hist_',4*time_lag/48),'T',rep(1:(time_lag/48), each = 4), '_', rep(1:4, time_lag/48),'_DFT'))
    
    #Dynamic version of DFT, denoted as DFTdy
    ft_dft_dynamic <- c(
      as.vector(t(daply(.data = df_lag_dynamic, .variables = 'dynamic_id', .fun = function(x) {
        mag <- Mod(fft(x$T_Ambient))
        mag_upd <- mag[1:round(length(mag)/2)]
        order(mag_upd, decreasing = T)[1:4]}))),
      as.vector(t(daply(.data = df_lag_dynamic, .variables = 'dynamic_id', .fun = function(x) {
        mag <- Mod(fft(x$RH_Ambient))
        mag_upd <- mag[1:round(length(mag)/2)]
        order(mag_upd, decreasing = T)[1:4]}))),
      as.vector(t(daply(.data = df_lag_dynamic, .variables = 'dynamic_id', .fun = function(x) {
        mag <- Mod(fft(x$Cooling_Load))
        mag_upd <- mag[1:round(length(mag)/2)]
        order(mag_upd, decreasing = T)[1:4]})))
    )
    names(ft_dft_dynamic) <- c(
      paste0(rep('T_hist_',4*time_lag/48),'T',rep(1:(time_lag/48), each = 4), '_', rep(1:4, time_lag/48),'_DFTdy'),
      paste0(rep('RH_hist_',4*time_lag/48),'T',rep(1:(time_lag/48), each = 4), '_', rep(1:4, time_lag/48),'_DFTdy'),
      paste0(rep('Load_hist_',4*time_lag/48),'T',rep(1:(time_lag/48), each = 4), '_', rep(1:4, time_lag/48),'_DFTdy'))
    
    #Extract the feature for current time
    ft_current <- as.numeric(df_current[,c(paste0('Month_',1:12), paste0('Day_',1:31), paste0('Hour_',0:23),
                                           paste0('Daytype_',c('Monday','Tuesday','Wednesday','Thursday','Saturday','Sunday')),
                                           'T_Ambient','RH_Ambient')])
    #Combine the features extracted
    ft_com <- c(ft_current, 
                ft_raw, ft_stat, ft_dft, 
                ft_raw_dynamic, ft_stat_dynamic, ft_dft_dynamic, 
                ft_stat_dynamic_recent, ft_last,
                Cooling_Load = df_current[,'Cooling_Load'])
    names(ft_com)[1:75] <- c(paste0('Month_',1:12), paste0('Day_',1:31), paste0('Hour_',0:23),
                             paste0('Daytype_',c('Monday','Tuesday','Wednesday','Thursday','Saturday','Sunday')),
                             'T_Ambient','RH_Ambient')
    return(ft_com)
  }  
  return(ft_temp)
}
colnames(ft_ext)
dim(ft_ext)
class(ft_ext)

ft_ext <- as.data.frame(ft_ext)
colnames(ft_ext)
summary(ft_ext)

#Transform DFT features to factors
colnames(ft_ext)[grep(pattern = '_DFT', x = colnames(ft_ext))]
for(i in grep(pattern = '_DFT', x = colnames(ft_ext))) {
  ft_ext[,i] <- factor(ft_ext[,i])
}
summary(ft_ext)

#Create dummy variables using one-hot encoding
rownames(ft_ext) <- 1:(dim(ft_ext)[1])
ft_ext <- dummy.data.frame(data = ft_ext, omit.constants = F, sep = '_Freq')
dim(ft_ext)
head(ft_ext)
summary(ft_ext)

###################################################################################################
#Select the variables used as inputs
colnames(ft_ext)
colnames(ft_ext)[grep(pattern = '_RAW\\>', x = colnames(ft_ext))]
colnames(ft_ext)[grep(pattern = '_STAT\\>', x = colnames(ft_ext))]
colnames(ft_ext)[grep(pattern = '_DFT_', x = colnames(ft_ext))]

colnames(ft_ext)[grep(pattern = '_RAWdy', x = colnames(ft_ext))]
colnames(ft_ext)[grep(pattern = '_STATdy', x = colnames(ft_ext))]
colnames(ft_ext)[grep(pattern = '_DFTdy', x = colnames(ft_ext))]

colnames(ft_ext)[grep(pattern = '_RECENTdy', x = colnames(ft_ext))]
colnames(ft_ext)[grep(pattern = '_LASTdy', x = colnames(ft_ext))]

#Basic version
var_BASIC <- c(1:which(colnames(ft_ext) == 'RH_Ambient'),
               which(colnames(ft_ext)=='Cooling_Load'))
#Static raw version
var_RAW <- c(1:which(colnames(ft_ext) == 'RH_Ambient'),  
             grep(pattern = '_RAW\\>', x = colnames(ft_ext)),
             which(colnames(ft_ext)=='Cooling_Load'))
#Static statistical version
var_STAT <- c(1:which(colnames(ft_ext) == 'RH_Ambient'),  
              grep(pattern = '_STAT\\>', x = colnames(ft_ext)),
              which(colnames(ft_ext)=='Cooling_Load'))
#Static structural version
var_DFT <- c(1:which(colnames(ft_ext) == 'RH_Ambient'),  
             grep(pattern = '_DFT_', x = colnames(ft_ext)),
             which(colnames(ft_ext)=='Cooling_Load'))
#Dynamic raw version
var_RAWdy <- c(1:which(colnames(ft_ext) == 'RH_Ambient'),  
               grep(pattern = '_RAWdy', x = colnames(ft_ext)),
               which(colnames(ft_ext)=='Cooling_Load'))
#Dynamic statistical version
var_STATdy <- c(1:which(colnames(ft_ext) == 'RH_Ambient'),  
                grep(pattern = '_STATdy', x = colnames(ft_ext)),
                which(colnames(ft_ext)=='Cooling_Load'))
#Dynamic structural version
var_DFTdy <- c(1:which(colnames(ft_ext) == 'RH_Ambient'),  
               grep(pattern = '_DFTdy', x = colnames(ft_ext)),
               which(colnames(ft_ext)=='Cooling_Load'))

#Dynamic recent statistical features
var_RECENTdy <- c(1:which(colnames(ft_ext) == 'RH_Ambient'),  
                  grep(pattern = '_RECENTdy', x = colnames(ft_ext)),
                  which(colnames(ft_ext)=='Cooling_Load'))

#Dynamic last observations
var_LASTdy <- c(1:which(colnames(ft_ext) == 'RH_Ambient'),  
                grep(pattern = '_LASTdy', x = colnames(ft_ext)),
                which(colnames(ft_ext)=='Cooling_Load'))

#Double check
for(i in c('BASIC','RAW','STAT','DFT','RAWdy','STATdy','DFTdy','RECENTdy','LASTdy')) {
  eval(parse(text = paste0('var_temp <- var_',i)))
  print(colnames(ft_ext)[var_temp]) 
}

#Split the data into train, val and test
#Note that the test days are excluded
dim(ft_ext)[1]/48
data_clean <- ft_ext[-ind_test_days,]
dim(data_clean)[1]/48
summary(data_clean)

set.seed(1)
ind_train <- sample(x = 1:dim(data_clean)[1], size = round(0.8*dim(data_clean)[1]), replace = F)
ind_val <- sample(x = c(1:dim(data_clean)[1])[-ind_train], size = round(0.1*dim(data_clean)[1]), replace = F)
ind_test <- c(1:dim(data_clean)[1])[-c(ind_train,ind_val)]

for(i in c('BASIC','RAW','STAT','DFT','RAWdy','STATdy','DFTdy','RECENTdy','LASTdy')) {
  eval(parse(text = paste0('data_train_',i,'_original <- data_clean[ind_train, var_',i,']')))
}

for(i in c('BASIC','RAW','STAT','DFT','RAWdy','STATdy','DFTdy','RECENTdy','LASTdy')) {
  eval(parse(text = paste0('data_val_',i,'_original <- data_clean[ind_val, var_',i,']')))
}

for(i in c('BASIC','RAW','STAT','DFT','RAWdy','STATdy','DFTdy','RECENTdy','LASTdy')) {
  eval(parse(text = paste0('data_test_',i,'_original <- data_clean[ind_test, var_',i,']')))
}

for(i in c('BASIC','RAW','STAT','DFT','RAWdy','STATdy','DFTdy','RECENTdy','LASTdy')) {
  eval(parse(text = paste0('test_days_',i,'_original <- ft_ext[ind_test_days, var_',i,']')))
}

dim(data_train_BASIC_original)
dim(data_val_BASIC_original)
dim(data_test_BASIC_original)
dim(test_days_STAT_original)

#Perform max-min normalization to generate validation and testing data sets
#Note that data_train_names, data_val_names, data_test_names, test_days_names are all normalized data
for(data_set in c('BASIC','RAW','STAT','DFT','RAWdy','STATdy','DFTdy','RECENTdy','LASTdy')) {
  eval(parse(text = paste0('preP_',data_set,' <- preProcess(x = data_train_',data_set,'_original[,-which(colnames(data_train_',data_set,'_original) == \'Cooling_Load\')], 
                           method = c(\'range\'))')))
  eval(parse(text = paste0('data_train_',data_set,' <- cbind.data.frame(
                           predict(preP_',data_set, ', data_train_',data_set,'_original[,-which(colnames(data_train_',data_set,'_original) == \'Cooling_Load\')]),
                           Cooling_Load = data_train_',data_set,'_original$Cooling_Load
                           )')))
  
  eval(parse(text = paste0('data_val_',data_set,' <- cbind.data.frame(
                           predict(preP_',data_set,', data_clean[ind_val,var_',data_set,'[-length(var_',data_set,')]]),
                           Cooling_Load = data_clean[ind_val, var_',data_set,'[length(var_',data_set,')]]
  )')))
  
  eval(parse(text = paste0('data_test_',data_set,' <- cbind.data.frame(
                           predict(preP_',data_set,', data_clean[ind_test,var_',data_set,'[-length(var_',data_set,')]]),
                           Cooling_Load = data_clean[ind_test, var_',data_set,'[length(var_',data_set,')]]
  )')))
  
  eval(parse(text = paste0('test_days_',data_set,' <- cbind.data.frame(
                           predict(preP_',data_set,', ft_ext[ind_test_days,var_',data_set,'[-length(var_',data_set,')]]),
                           Cooling_Load = ft_ext[ind_test_days, var_',data_set,'[length(var_',data_set,')]]
  )')))
}

###################################################################################################
###################################################################################################
#Generate AE features, each time series (i.e., T, RH, Load) in history will be represented as 4 features
res_AE <- foreach(i = c('T_hist','RH_hist','Load_hist')) %do% {
  AE_temp <- h2o.deeplearning(x = grep(pattern = i, x = colnames(data_train_RAW)), 
                              training_frame = as.h2o(data_train_RAW), 
                              validation_frame = as.h2o(data_val_RAW),
                              autoencoder = T, standardize = F, 
                              activation = 'Tanh', 
                              hidden = c(25, 10, 4, 10, 25), 
                              adaptive_rate = T, seed = 1, epochs = 150, 
                              stopping_rounds = 10, stopping_metric = 'MSE', stopping_tolerance = .0001)
  error_temp <- sum(as.vector(h2o.anomaly(AE_temp, data = as.h2o(data_train_RAW))))
  return(list(Model = AE_temp, Error = error_temp))
}
laply(.data = res_AE, .fun = function(x) x$Error)

#Generate the features 
data_train_AE <- cbind.data.frame(
  data_train_RAW[,1:which(colnames(data_train_RAW) == 'RH_Ambient')],
  as.data.frame(h2o.deepfeatures(object = res_AE[[1]]$Model, data = as.h2o(data_train_RAW), layer = 3)),
  as.data.frame(h2o.deepfeatures(object = res_AE[[2]]$Model, data = as.h2o(data_train_RAW), layer = 3)),
  as.data.frame(h2o.deepfeatures(object = res_AE[[3]]$Model, data = as.h2o(data_train_RAW), layer = 3)),
  Cooling_Load = data_train_RAW[,which(colnames(data_train_RAW) == 'Cooling_Load')]
)
dim(data_train_AE)
colnames(data_train_AE)[(which(colnames(data_train_AE) == 'RH_Ambient')+1):(dim(data_train_AE)[2]-1)] <- c(
  paste0('T_hist_AE_',1:4), paste0('RH_hist_AE_',1:4),paste0('Load_hist_AE_',1:4)
) 
head(data_train_AE)

data_val_AE <- cbind.data.frame(
  data_val_RAW[,1:which(colnames(data_val_RAW) == 'RH_Ambient')],
  as.data.frame(h2o.deepfeatures(object = res_AE[[1]]$Model, data = as.h2o(data_val_RAW), layer = 3)),
  as.data.frame(h2o.deepfeatures(object = res_AE[[2]]$Model, data = as.h2o(data_val_RAW), layer = 3)),
  as.data.frame(h2o.deepfeatures(object = res_AE[[3]]$Model, data = as.h2o(data_val_RAW), layer = 3)),
  Cooling_Load = data_val_RAW[,which(colnames(data_val_RAW) == 'Cooling_Load')]
)
dim(data_val_AE)
colnames(data_val_AE)[(which(colnames(data_val_AE) == 'RH_Ambient')+1):(dim(data_val_AE)[2]-1)] <- c(
  paste0('T_hist_AE_',1:4), paste0('RH_hist_AE_',1:4),paste0('Load_hist_AE_',1:4)
) 
head(data_val_AE)

data_test_AE <- cbind.data.frame(
  data_test_RAW[,1:which(colnames(data_test_RAW) == 'RH_Ambient')],
  as.data.frame(h2o.deepfeatures(object = res_AE[[1]]$Model, data = as.h2o(data_test_RAW), layer = 3)),
  as.data.frame(h2o.deepfeatures(object = res_AE[[2]]$Model, data = as.h2o(data_test_RAW), layer = 3)),
  as.data.frame(h2o.deepfeatures(object = res_AE[[3]]$Model, data = as.h2o(data_test_RAW), layer = 3)),
  Cooling_Load = data_test_RAW[,which(colnames(data_test_RAW) == 'Cooling_Load')]
)
dim(data_test_AE)
colnames(data_test_AE)[(which(colnames(data_test_AE) == 'RH_Ambient')+1):(dim(data_test_AE)[2]-1)] <- c(
  paste0('T_hist_AE_',1:4), paste0('RH_hist_AE_',1:4),paste0('Load_hist_AE_',1:4)
) 
head(data_test_AE)

test_days_AE <- cbind.data.frame(
  predict(preP_BASIC, ft_ext[ind_test_days,var_BASIC[-length(var_BASIC)]]),
  as.data.frame(h2o.deepfeatures(object = res_AE[[1]]$Model, data = as.h2o(test_days_RAW), layer = 3)),
  as.data.frame(h2o.deepfeatures(object = res_AE[[2]]$Model, data = as.h2o(test_days_RAW), layer = 3)),
  as.data.frame(h2o.deepfeatures(object = res_AE[[3]]$Model, data = as.h2o(test_days_RAW), layer = 3)),
  Cooling_Load = ft_ext[ind_test_days,'Cooling_Load']
)
dim(test_days_AE)
colnames(test_days_AE)[(which(colnames(test_days_AE) == 'RH_Ambient')+1):(dim(test_days_AE)[2]-1)] <- c(
  paste0('T_hist_AE_',1:4), paste0('RH_hist_AE_',1:4),paste0('Load_hist_AE_',1:4)
)
head(test_days_AE)

res_AEdy <- foreach(i = c('T_hist','RH_hist','Load_hist')) %do% {
  AE_temp <- h2o.deeplearning(x = grep(pattern = i, x = colnames(data_train_RAWdy)), 
                              training_frame = as.h2o(data_train_RAWdy), 
                              validation_frame = as.h2o(data_train_RAWdy),
                              autoencoder = T, standardize = F, 
                              activation = 'Tanh', 
                              hidden = c(25, 10, 4, 10, 25), 
                              adaptive_rate = T, seed = 1, epochs = 100, 
                              stopping_rounds = 10, stopping_metric = 'MSE', stopping_tolerance = .0001)
  error_temp <- sum(as.vector(h2o.anomaly(AE_temp, data = as.h2o(data_train_RAWdy))))
  return(list(Model = AE_temp, Error = error_temp))
}
laply(.data = res_AEdy, .fun = function(x) x$Error)

#Generate the features 
data_train_AEdy <- cbind.data.frame(
  data_train_RAWdy[,1:which(colnames(data_train_RAWdy) == 'RH_Ambient')],
  as.data.frame(h2o.deepfeatures(object = res_AEdy[[1]]$Model, data = as.h2o(data_train_RAWdy), layer = 3)),
  as.data.frame(h2o.deepfeatures(object = res_AEdy[[2]]$Model, data = as.h2o(data_train_RAWdy), layer = 3)),
  as.data.frame(h2o.deepfeatures(object = res_AEdy[[3]]$Model, data = as.h2o(data_train_RAWdy), layer = 3)),
  Cooling_Load = data_train_RAWdy[,which(colnames(data_train_RAWdy) == 'Cooling_Load')]
)
dim(data_train_AEdy)
colnames(data_train_AEdy)[(which(colnames(data_train_AEdy) == 'RH_Ambient')+1):(dim(data_train_AEdy)[2]-1)] <- c(
  paste0('T_hist_AEdy_',1:4), paste0('RH_hist_AEdy_',1:4),paste0('Load_hist_AEdy_',1:4)
) 
head(data_train_AEdy)

data_val_AEdy <- cbind.data.frame(
  data_val_RAWdy[,1:which(colnames(data_val_RAWdy) == 'RH_Ambient')],
  as.data.frame(h2o.deepfeatures(object = res_AEdy[[1]]$Model, data = as.h2o(data_val_RAWdy), layer = 3)),
  as.data.frame(h2o.deepfeatures(object = res_AEdy[[2]]$Model, data = as.h2o(data_val_RAWdy), layer = 3)),
  as.data.frame(h2o.deepfeatures(object = res_AEdy[[3]]$Model, data = as.h2o(data_val_RAWdy), layer = 3)),
  Cooling_Load = data_val_RAWdy[,which(colnames(data_val_RAWdy) == 'Cooling_Load')]
)
dim(data_val_AEdy)
colnames(data_val_AEdy)[(which(colnames(data_val_AEdy) == 'RH_Ambient')+1):(dim(data_val_AEdy)[2]-1)] <- c(
  paste0('T_hist_AEdy_',1:4), paste0('RH_hist_AEdy_',1:4),paste0('Load_hist_AEdy_',1:4)
) 
head(data_val_AEdy)

data_test_AEdy <- cbind.data.frame(
  data_test_RAWdy[,1:which(colnames(data_test_RAWdy) == 'RH_Ambient')],
  as.data.frame(h2o.deepfeatures(object = res_AEdy[[1]]$Model, data = as.h2o(data_test_RAWdy), layer = 3)),
  as.data.frame(h2o.deepfeatures(object = res_AEdy[[2]]$Model, data = as.h2o(data_test_RAWdy), layer = 3)),
  as.data.frame(h2o.deepfeatures(object = res_AEdy[[3]]$Model, data = as.h2o(data_test_RAWdy), layer = 3)),
  Cooling_Load = data_test_RAWdy[,which(colnames(data_test_RAWdy) == 'Cooling_Load')]
)
dim(data_test_AEdy)
colnames(data_test_AEdy)[(which(colnames(data_test_AEdy) == 'RH_Ambient')+1):(dim(data_test_AEdy)[2]-1)] <- c(
  paste0('T_hist_AEdy_',1:4), paste0('RH_hist_AEdy_',1:4),paste0('Load_hist_AEdy_',1:4)
) 
head(data_test_AEdy)

test_days_AEdy <- cbind.data.frame(
  predict(preP_BASIC, ft_ext[ind_test_days,var_BASIC[-length(var_BASIC)]]),
  as.data.frame(h2o.deepfeatures(object = res_AEdy[[1]]$Model, data = as.h2o(test_days_RAWdy), layer = 3)),
  as.data.frame(h2o.deepfeatures(object = res_AEdy[[2]]$Model, data = as.h2o(test_days_RAWdy), layer = 3)),
  as.data.frame(h2o.deepfeatures(object = res_AEdy[[3]]$Model, data = as.h2o(test_days_RAWdy), layer = 3)),
  Cooling_Load = ft_ext[ind_test_days,'Cooling_Load']
)
dim(test_days_AEdy)
colnames(test_days_AEdy)[(which(colnames(test_days_AEdy) == 'RH_Ambient')+1):(dim(test_days_AEdy)[2]-1)] <- c(
  paste0('T_hist_AEdy_',1:4), paste0('RH_hist_AEdy_',1:4),paste0('Load_hist_AEdy_',1:4)
) 
head(test_days_AEdy)

#Up till now, 11 data set have been prepared for comparison
#All data frames' inputs are normalized
dim(data_train_BASIC)
dim(data_train_RAW)
dim(data_train_STAT)
dim(data_train_DFT)
dim(data_train_AE)
dim(data_train_RAWdy)
dim(data_train_STATdy)
dim(data_train_DFTdy)
dim(data_train_AEdy)
dim(data_train_RECENTdy)
dim(data_train_LASTdy)

dim(test_days_BASIC)
dim(test_days_RAW)
dim(test_days_STAT)
dim(test_days_DFT)
dim(test_days_AE)
dim(test_days_RAWdy)
dim(test_days_STATdy)
dim(test_days_DFTdy)
dim(test_days_AEdy)
dim(test_days_RECENTdy)
dim(test_days_LASTdy)

###################################################################################################
###################################################################################################
#(1) Multiple linear regression
res_linear <- foreach(data_set = c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy')) %do% {
  eval(parse(text = paste0('data_train <- data_train_',data_set)))
  eval(parse(text = paste0('data_val <- data_val_',data_set)))
  eval(parse(text = paste0('data_test <- data_test_',data_set)))
  eval(parse(text = paste0('test_days <- test_days_',data_set)))
  
  #Develop the model
  ptm <- proc.time()
  model_temp <- lm(Cooling_Load ~., data = data_train)
  time_model <- (proc.time() - ptm)[3]
  
  #Evaluate the performance based on testing samples
  pred_samples <- predict(model_temp, data_test)
  
  #Evaluate the performance based on testing profiles
  #Note that for 'dy' data set, iterative process is a much
  ptm_pred <- proc.time()
  if(length(grep(pattern = 'dy', x = data_set)) == 1) {
    pred_profiles <- foreach(n = seq(from = 1, to = dim(test_days)[1], by = 48), .combine = c) %do% {
      pred_temp <- c()
      for (m in 1:48) {
        if(m == 1) {
          pred_temp <- as.numeric(predict(model_temp, test_days[n+m-1,]))
        } else {
          #RAWdy
          if(data_set == 'RAWdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAWdy_original))])
            load_hist_raw <- c(load_hist[1:(48-m+1)],pred_temp)
            load_hist_upd <- (load_hist_raw - preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][1,])/(preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][2,] - preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][1,])
            test_days[n+m-1,paste0('Load_hist_',1:48,'_RAWdy')] <- load_hist_upd 
            pred_temp <- c(pred_temp, as.numeric(predict(model_temp, test_days[n+m-1,])))
          }
          #LASTdy
          if(data_set == 'LASTdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAWdy_original))])
            load_hist_raw <- c(load_hist[1:(48-m+1)],pred_temp)
            load_hist_upd <- (load_hist_raw[length(load_hist_raw)] - preP_LASTdy$ranges[,'Load_hist_LASTdy'][1])/(preP_LASTdy$ranges[,'Load_hist_LASTdy'][2] - preP_LASTdy$ranges[,'Load_hist_LASTdy'][1]) 
            test_days[n+m-1,'Load_hist_LASTdy'] <- load_hist_upd 
            pred_temp <- c(pred_temp, as.numeric(predict(model_temp, test_days[n+m-1,])))
          }
          #STATdy
          if(data_set == 'STATdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAW_original))])
            load_hist_raw <- c(load_hist[1:(48-m+1)],pred_temp)
            load_replace <- c(min(load_hist_raw), max(load_hist_raw), mean(load_hist_raw), sd(load_hist_raw))
            preP_min <- preP_STATdy$range[,c('Load_hist_min_1_STATdy','Load_hist_max_1_STATdy',
                                             'Load_hist_mean_1_STATdy','Load_hist_sd_1_STATdy')][1,]
            preP_max <- preP_STATdy$range[,c('Load_hist_min_1_STATdy','Load_hist_max_1_STATdy',
                                             'Load_hist_mean_1_STATdy','Load_hist_sd_1_STATdy')][2,]
            load_replace_upd <- as.numeric((load_replace - preP_min)/(preP_max - preP_min))
            test_days[n+m-1,c('Load_hist_min_1_STATdy','Load_hist_max_1_STATdy',
                              'Load_hist_mean_1_STATdy','Load_hist_sd_1_STATdy')] <- load_replace_upd
            pred_temp <- c(pred_temp, as.numeric(predict(model_temp, test_days[n+m-1,])))
          }
          #DFTdy
          if(data_set == 'DFTdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAWdy_original))])
            load_hist_raw <- c(load_hist[1:(48-m+1)],pred_temp)
            freq <- order(Mod(fft(load_hist_raw))[1:(round(length(Mod(fft(load_hist_raw)))/2))], decreasing = T)[1:4]
            var_to_change <- paste0('Load_hist_T1_',1:4,'_DFTdy_Freq',freq)
            for(q in 1:length(var_to_change)) {
              if(var_to_change[q] %in% colnames(test_days)) {
                test_days[n+m-1,var_to_change[q]] <- 1
              } else {
                next
              }
            }
            pred_temp <- c(pred_temp, as.numeric(predict(model_temp, test_days[n+m-1,])))
          }
          #AEdy
          if(data_set == 'AEdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAWdy_original))])
            load_hist_raw <-c(load_hist[1:(48-m+1)],pred_temp)
            load_hist_upd <- (load_hist_raw - preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][1,])/(preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][2,] - preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][1,])
            frame <- data.frame(matrix(data = load_hist_upd, nrow = 1))
            colnames(frame) <- paste0('Load_hist_',1:48,'_RAWdy')
            ae_get <- as.vector(h2o.deepfeatures(object = res_AEdy[[3]]$Model, data = as.h2o(frame), layer = 3))
            test_days[n+m-1,paste0('Load_hist_AEdy_',1:4)] <- ae_get
            pred_temp <- c(pred_temp, as.numeric(predict(model_temp, test_days[n+m-1,])))
          }
          #RECENT
          if(data_set == 'RECENTdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAWdy_original))])
            load_hist_raw <- c(load_hist[1:(48-m+1)],pred_temp)
            load_replace <- c(mean(load_hist_raw[43:44]), mean(load_hist_raw[45:46]), mean(load_hist_raw[47:48]))
            load_replace_upd <- (load_replace - preP_RECENTdy$range[,paste0('Load_mean_',1:3,'_RECENTdy')][1,])/(preP_RECENTdy$range[,paste0('Load_mean_',1:3,'_RECENTdy')][2,]-preP_RECENTdy$range[,paste0('Load_mean_',1:3,'_RECENTdy')][1,])
            test_days[n+m-1,c('Load_mean_1_RECENTdy','Load_mean_2_RECENTdy','Load_mean_3_RECENTdy')] <- load_replace_upd
            pred_temp <- c(pred_temp, as.numeric(predict(model_temp, test_days[n+m-1,])))
          }
        }
      }
      return(pred_temp)
    }
  } else {
    pred_profiles <- predict(model_temp, test_days)
  }
  time_pred <- (proc.time() - ptm_pred)[3]
  
  return(list(Model = model_temp, 
              pred_test_samples = cbind.data.frame(Pred = pred_samples, Actual = data_test$Cooling_Load), 
              pred_test_profiles = cbind.data.frame(Pred = pred_profiles, Actual = test_days$Cooling_Load),
              Time_model = time_model,
              Time_pred = time_pred))
}

#Get the accuracy on testing samples
df_pred_test <- laply(.data = res_linear, .fun = function(x) accuracy(f = x$pred_test_samples$Pred, x = x$pred_test_samples$Actual))
rownames(df_pred_test) <- c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy')
df_pred_test

#Get the accuracy on testing profiles
df_pred_profile <- laply(.data = res_linear, .fun = function(x) accuracy(f = x$pred_test_profiles$Pred, x = x$pred_test_profiles$Actual))
rownames(df_pred_profile) <- c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy')
df_pred_profile

#Get the time on model development
df_time <- cbind(
  laply(.data = res_linear, .fun = function(x) x$Time_model),
  laply(.data = res_linear, .fun = function(x) x$Time_pred))
rownames(df_time) <- c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy')
df_time

###################################################################################################
###################################################################################################
#(2) Elastic net
res_elnet <- foreach(data_set = c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy')) %do% {
  eval(parse(text = paste0('data_train <- data_train_',data_set)))
  eval(parse(text = paste0('data_val <- data_val_',data_set)))
  eval(parse(text = paste0('data_test <- data_test_',data_set)))
  eval(parse(text = paste0('test_days <- test_days_',data_set)))
  
  res_temp <- foreach(alpha = seq(from = 0, to = 1, by = .1), .combine = rbind.data.frame) %do% {
    temp <- cv.glmnet(x = as.matrix(data_train[,-which(colnames(data_train)=='Cooling_Load')]), 
                      y = data_train$Cooling_Load, 
                      alpha = alpha, standardize = F,
                      nfolds = 3)
    pred_temp <- predict(temp, as.matrix(data_val[,-which(colnames(data_val) == 'Cooling_Load')]), 
                         lambda = temp$lambda.1min)
    c(temp$lambda.min, accuracy(f = as.vector(pred_temp), x = data_val$Cooling_Load)[1:3])
  }
  colnames(res_temp) <- c('lambda_min','ME','RMSE','MAE')
  
  ptm <- proc.time()
  model_temp <- glmnet(x = as.matrix(data_train[,-which(colnames(data_train)=='Cooling_Load')]), 
                       y = data_train$Cooling_Load, family = 'gaussian', 
                       alpha = seq(from = 0, to = 1, by = .1)[which.min(res_temp$RMSE)], 
                       lambda = res_temp[which.min(res_temp$RMSE),'lambda_min'], standardize = F)
  time_model <- (proc.time() - ptm)[3]
  
  #Performance on testing samples
  pred_samples <- as.vector(predict(model_temp, 
                                    as.matrix(data_test[,-which(colnames(data_train)=='Cooling_Load')])))
  
  #Evaluate the performance based on testing profiles
  #Note that for 'dy' data set, iterative process is a much
  ptm_pred <- proc.time()
  if(length(grep(pattern = 'dy', x = data_set)) == 1) {
    pred_profiles <- foreach(n = seq(from = 1, to = dim(test_days)[1], by = 48), .combine = c) %do% {
      pred_temp <- c()
      for (m in 1:48) {
        if(m == 1) {
          pred_temp <- as.vector(predict(model_temp, as.matrix(test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')])))
        } else {
          #RAWdy
          if(data_set == 'RAWdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAWdy_original))])
            load_hist_raw <- c(load_hist[1:(48-m+1)],pred_temp)
            load_hist_upd <- (load_hist_raw - preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][1,])/(preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][2,] - preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][1,])
            test_days[n+m-1,paste0('Load_hist_',1:48,'_RAWdy')] <- load_hist_upd 
            pred_temp <- c(pred_temp, as.vector(predict(model_temp, as.matrix(test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')]))))
          }
          #LASTdy
          if(data_set == 'LASTdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAWdy_original))])
            load_hist_raw <- c(load_hist[1:(48-m+1)],pred_temp)
            load_hist_upd <- (load_hist_raw[length(load_hist_raw)] - preP_LASTdy$ranges[,'Load_hist_LASTdy'][1])/(preP_LASTdy$ranges[,'Load_hist_LASTdy'][2] - preP_LASTdy$ranges[,'Load_hist_LASTdy'][1]) 
            test_days[n+m-1,'Load_hist_LASTdy'] <- load_hist_upd 
            pred_temp <- c(pred_temp, as.vector(predict(model_temp, as.matrix(test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')]))))
          }
          #STATdy
          if(data_set == 'STATdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAW_original))])
            load_hist_raw <- c(load_hist[1:(48-m+1)],pred_temp)
            load_replace <- c(min(load_hist_raw), max(load_hist_raw), mean(load_hist_raw), sd(load_hist_raw))
            preP_min <- preP_STATdy$range[,c('Load_hist_min_1_STATdy','Load_hist_max_1_STATdy',
                                             'Load_hist_mean_1_STATdy','Load_hist_sd_1_STATdy')][1,]
            preP_max <- preP_STATdy$range[,c('Load_hist_min_1_STATdy','Load_hist_max_1_STATdy',
                                             'Load_hist_mean_1_STATdy','Load_hist_sd_1_STATdy')][2,]
            load_replace_upd <- as.numeric((load_replace - preP_min)/(preP_max - preP_min))
            test_days[n+m-1,c('Load_hist_min_1_STATdy','Load_hist_max_1_STATdy',
                              'Load_hist_mean_1_STATdy','Load_hist_sd_1_STATdy')] <- load_replace_upd
            pred_temp <- c(pred_temp, as.vector(predict(model_temp, as.matrix(test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')]))))
          }
          #DFTdy
          if(data_set == 'DFTdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAWdy_original))])
            load_hist_raw <- c(load_hist[1:(48-m+1)],pred_temp)
            freq <- order(Mod(fft(load_hist_raw))[1:(round(length(Mod(fft(load_hist_raw)))/2))], decreasing = T)[1:4]
            var_to_change <- paste0('Load_hist_T1_',1:4,'_DFTdy_Freq',freq)
            for(q in 1:length(var_to_change)) {
              if(var_to_change[q] %in% colnames(test_days)) {
                test_days[n+m-1,var_to_change[q]] <- 1
              } else {
                next
              }
            }
            pred_temp <- c(pred_temp, as.vector(predict(model_temp, as.matrix(test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')]))))
          }
          #AEdy
          if(data_set == 'AEdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAWdy_original))])
            load_hist_raw <-c(load_hist[1:(48-m+1)],pred_temp)
            load_hist_upd <- (load_hist_raw - preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][1,])/(preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][2,] - preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][1,])
            frame <- data.frame(matrix(data = load_hist_upd, nrow = 1))
            colnames(frame) <- paste0('Load_hist_',1:48,'_RAWdy')
            ae_get <- as.vector(h2o.deepfeatures(object = res_AEdy[[3]]$Model, data = as.h2o(frame), layer = 3))
            test_days[n+m-1,paste0('Load_hist_AEdy_',1:4)] <- ae_get
            pred_temp <- c(pred_temp, as.vector(predict(model_temp, as.matrix(test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')]))))
          }
          #RECENT
          if(data_set == 'RECENTdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAWdy_original))])
            load_hist_raw <- c(load_hist[1:(48-m+1)],pred_temp)
            load_replace <- c(mean(load_hist_raw[43:44]), mean(load_hist_raw[45:46]), mean(load_hist_raw[47:48]))
            load_replace_upd <- (load_replace - preP_RECENTdy$range[,paste0('Load_mean_',1:3,'_RECENTdy')][1,])/(preP_RECENTdy$range[,paste0('Load_mean_',1:3,'_RECENTdy')][2,]-preP_RECENTdy$range[,paste0('Load_mean_',1:3,'_RECENTdy')][1,])
            test_days[n+m-1,c('Load_mean_1_RECENTdy','Load_mean_2_RECENTdy','Load_mean_3_RECENTdy')] <- load_replace_upd
            pred_temp <- c(pred_temp, as.vector(predict(model_temp, as.matrix(test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')]))))
          }
        }
      }
      return(pred_temp)
    }
  } else {
    pred_profiles <- as.vector(predict(model_temp, as.matrix(test_days[,-which(colnames(test_days) == 'Cooling_Load')])))
  }
  time_pred <- (proc.time() - ptm_pred)[3]
  
  return(list(Model = model_temp, 
              Opt = res_temp,
              pred_test_samples = cbind.data.frame(Pred = pred_samples, Actual = data_test$Cooling_Load), 
              pred_test_profiles = cbind.data.frame(Pred = pred_profiles, Actual = test_days$Cooling_Load),
              Time_model = time_model,
              Time_pred = time_pred))
}

#Get the accuracy on testing samples
df_pred_test <- laply(.data = res_elnet, .fun = function(x) accuracy(f = x$pred_test_samples$Pred, x = x$pred_test_samples$Actual))
rownames(df_pred_test) <- c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy')
df_pred_test

#Get the accuracy on testing profiles
df_pred_profile <- laply(.data = res_elnet, .fun = function(x) accuracy(f = x$pred_test_profiles$Pred, x = x$pred_test_profiles$Actual))
rownames(df_pred_profile) <- c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy')
df_pred_profile

#Get the time on model development
df_time <- cbind(
  laply(.data = res_elnet, .fun = function(x) x$Time_model),
  laply(.data = res_elnet, .fun = function(x) x$Time_pred))
rownames(df_time) <- c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy')
df_time

###################################################################################################
###################################################################################################
#(3) Random Forests
res_rf <- foreach(data_set = c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy')) %do% {
  eval(parse(text = paste0('data_train <- data_train_',data_set)))
  eval(parse(text = paste0('data_val <- data_val_',data_set)))
  eval(parse(text = paste0('data_test <- data_test_',data_set)))
  eval(parse(text = paste0('test_days <- test_days_',data_set)))
  
  #Note the the computation costs for RF is high, so only select a proportion for parameter tuning
  set.seed(123)
  ind_opt <- sample(x = 1:dim(data_train)[1], size = 1000, replace = F)
  
  #Optimization on parameters
  res_temp <- foreach(m_tree = c(25,30,35), .combine = rbind.data.frame) %do% {
    res_temp_temp <- foreach(n_tree = c(1000), .combine = rbind) %do% {
      temp <- randomForest(x = data_train[ind_opt,-which(colnames(data_train) == 'Cooling_Load')], 
                           y = data_train$Cooling_Load[ind_opt], 
                           xtest = data_val[,-which(colnames(data_val) == 'Cooling_Load')], 
                           ytest = data_val$Cooling_Load, 
                           ntree = n_tree, mtry = m_tree, 
                           replace = T, importance = T)
      c(m_tree, n_tree, accuracy(f = temp$test$predicted, x = data_val$Cooling_Load)[1:3]) 
    }
  }
  colnames(res_temp) <- c('m_tree','n_tree','ME','RMSE','MAE')
  
  #Develop the final model
  #Note that extra sampling is used since the computation time is too large
  set.seed(123)
  ind_use <- sample(x = 1:dim(data_train)[1], size = 3000, replace = F)
  
  ptm <- proc.time()
  model_temp <- randomForest(x = data_train[ind_use,-which(colnames(data_train) == 'Cooling_Load')], 
                             y = data_train$Cooling_Load[ind_use], 
                             xtest = data_test[,-which(colnames(data_test) == 'Cooling_Load')], 
                             ytest = data_test$Cooling_Load, 
                             ntree = res_temp[which.min(res_temp$RMSE),'n_tree'], 
                             mtry = res_temp[which.min(res_temp$RMSE),'m_tree'], 
                             replace = T, importance = T, keep.forest = T)
  time_model <- (proc.time() - ptm)[3]
  
  #Performance on testing samples
  pred_samples <- predict(model_temp, data_test[,-which(colnames(data_test) == 'Cooling_Load')])
  
  #Evaluate the performance based on testing profiles
  #Note that for 'dy' data set, iterative process is a much
  ptm_pred <- proc.time()
  if(length(grep(pattern = 'dy', x = data_set)) == 1) {
    pred_profiles <- foreach(n = seq(from = 1, to = dim(test_days)[1], by = 48), .combine = c) %do% {
      pred_temp <- c()
      for (m in 1:48) {
        if(m == 1) {
          pred_temp <- predict(model_temp, test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')])
        } else {
          #RAWdy
          if(data_set == 'RAWdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAWdy_original))])
            load_hist_raw <- c(load_hist[1:(48-m+1)],pred_temp)
            load_hist_upd <- (load_hist_raw - preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][1,])/(preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][2,] - preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][1,])
            test_days[n+m-1,paste0('Load_hist_',1:48,'_RAWdy')] <- load_hist_upd 
            pred_temp <- c(pred_temp, predict(model_temp, test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')]))
          }
          #LASTdy
          if(data_set == 'LASTdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAWdy_original))])
            load_hist_raw <- c(load_hist[1:(48-m+1)],pred_temp)
            load_hist_upd <- (load_hist_raw[length(load_hist_raw)] - preP_LASTdy$ranges[,'Load_hist_LASTdy'][1])/(preP_LASTdy$ranges[,'Load_hist_LASTdy'][2] - preP_LASTdy$ranges[,'Load_hist_LASTdy'][1]) 
            test_days[n+m-1,'Load_hist_LASTdy'] <- load_hist_upd 
            pred_temp <- c(pred_temp, predict(model_temp, test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')]))
          }
          #STATdy
          if(data_set == 'STATdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAW_original))])
            load_hist_raw <- c(load_hist[1:(48-m+1)],pred_temp)
            load_replace <- c(min(load_hist_raw), max(load_hist_raw), mean(load_hist_raw), sd(load_hist_raw))
            preP_min <- preP_STATdy$range[,c('Load_hist_min_1_STATdy','Load_hist_max_1_STATdy',
                                             'Load_hist_mean_1_STATdy','Load_hist_sd_1_STATdy')][1,]
            preP_max <- preP_STATdy$range[,c('Load_hist_min_1_STATdy','Load_hist_max_1_STATdy',
                                             'Load_hist_mean_1_STATdy','Load_hist_sd_1_STATdy')][2,]
            load_replace_upd <- as.numeric((load_replace - preP_min)/(preP_max - preP_min))
            test_days[n+m-1,c('Load_hist_min_1_STATdy','Load_hist_max_1_STATdy',
                              'Load_hist_mean_1_STATdy','Load_hist_sd_1_STATdy')] <- load_replace_upd
            pred_temp <- c(pred_temp, predict(model_temp, test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')]))
          }
          #DFTdy
          if(data_set == 'DFTdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAWdy_original))])
            load_hist_raw <- c(load_hist[1:(48-m+1)],pred_temp)
            freq <- order(Mod(fft(load_hist_raw))[1:(round(length(Mod(fft(load_hist_raw)))/2))], decreasing = T)[1:4]
            var_to_change <- paste0('Load_hist_T1_',1:4,'_DFTdy_Freq',freq)
            for(q in 1:length(var_to_change)) {
              if(var_to_change[q] %in% colnames(test_days)) {
                test_days[n+m-1,var_to_change[q]] <- 1
              } else {
                next
              }
            }
            pred_temp <- c(pred_temp, predict(model_temp, test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')]))
          }
          #AEdy
          if(data_set == 'AEdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAWdy_original))])
            load_hist_raw <-c(load_hist[1:(48-m+1)],pred_temp)
            load_hist_upd <- (load_hist_raw - preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][1,])/(preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][2,] - preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][1,])
            frame <- data.frame(matrix(data = load_hist_upd, nrow = 1))
            colnames(frame) <- paste0('Load_hist_',1:48,'_RAWdy')
            ae_get <- as.vector(h2o.deepfeatures(object = res_AEdy[[3]]$Model, data = as.h2o(frame), layer = 3))
            test_days[n+m-1,paste0('Load_hist_AEdy_',1:4)] <- ae_get
            pred_temp <- c(pred_temp, predict(model_temp, test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')]))
          }
          #RECENT
          if(data_set == 'RECENTdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAWdy_original))])
            load_hist_raw <- c(load_hist[1:(48-m+1)],pred_temp)
            load_replace <- c(mean(load_hist_raw[43:44]), mean(load_hist_raw[45:46]), mean(load_hist_raw[47:48]))
            load_replace_upd <- (load_replace - preP_RECENTdy$range[,paste0('Load_mean_',1:3,'_RECENTdy')][1,])/(preP_RECENTdy$range[,paste0('Load_mean_',1:3,'_RECENTdy')][2,]-preP_RECENTdy$range[,paste0('Load_mean_',1:3,'_RECENTdy')][1,])
            test_days[n+m-1,c('Load_mean_1_RECENTdy','Load_mean_2_RECENTdy','Load_mean_3_RECENTdy')] <- load_replace_upd
            pred_temp <- c(pred_temp, predict(model_temp, test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')]))
          }
        }
      }
      return(pred_temp)
    }
  } else {
    pred_profiles <- predict(model_temp, test_days[,-which(colnames(test_days) == 'Cooling_Load')])
  }
  time_pred <- (proc.time() - ptm_pred)[3]
  
  return(list(Model = model_temp, 
              Opt = res_temp,
              pred_test_samples = cbind.data.frame(Pred = pred_samples, Actual = data_test$Cooling_Load), 
              pred_test_profiles = cbind.data.frame(Pred = pred_profiles, Actual = test_days$Cooling_Load),
              Time_model = time_model,
              Time_pred = time_pred))
}

#Get the accuracy on testing samples
df_pred_test <- laply(.data = res_rf, .fun = function(x) accuracy(f = x$pred_test_samples$Pred, x = x$pred_test_samples$Actual))
rownames(df_pred_test) <- c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy')
df_pred_test

#Get the accuracy on testing profiles
df_pred_profile <- laply(.data = res_rf, .fun = function(x) accuracy(f = x$pred_test_profiles$Pred, x = x$pred_test_profiles$Actual))
rownames(df_pred_profile) <- c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy')
df_pred_profile

#Get the time on model development
df_time <- cbind(
  laply(.data = res_rf, .fun = function(x) x$Time_model),
  laply(.data = res_rf, .fun = function(x) x$Time_pred))
rownames(df_time) <- c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy')
df_time

###################################################################################################
###################################################################################################
#(4) Gradient boosting trees
res_gbm <- foreach(data_set = c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy')) %do% {
  eval(parse(text = paste0('data_train <- data_train_',data_set)))
  eval(parse(text = paste0('data_val <- data_val_',data_set)))
  eval(parse(text = paste0('data_test <- data_test_',data_set)))
  eval(parse(text = paste0('test_days <- test_days_',data_set)))
  
  #Note that only a proportion of data is used for parameter tuning
  set.seed(123)
  ind_opt <- sample(x = 1:dim(data_train)[1], size = 1000, replace = F)
  
  #Optimization on parameters
  res_temp <- foreach(lr = c(.005, .01, .05), .combine = rbind.data.frame) %do% {
    res_temp_temp <- foreach(n_tree = c(1000), .combine = rbind) %do% {
      temp <- gbm.fit(x = data_train[ind_opt,-which(colnames(data_train) == 'Cooling_Load')], 
                      y = data_train$Cooling_Load[ind_opt], 
                      distribution = 'gaussian', 
                      n.trees = n_tree, 
                      interaction.depth = 8, 
                      shrinkage = lr)
      pred_val <- predict(temp, data_val[,-which(colnames(data_val) == 'Cooling_Load')], n.tree = n_tree)
      c(lr, n_tree, accuracy(f = pred_val, x = data_val$Cooling_Load)[1:3]) 
    }
  }
  colnames(res_temp) <- c('lr','n_tree','ME','RMSE','MAE')
  
  #Develop the final model
  ptm <- proc.time()
  model_temp <- gbm.fit(x = data_train[,-which(colnames(data_train) == 'Cooling_Load')], 
                        y = data_train$Cooling_Load, 
                        distribution = 'gaussian', 
                        interaction.depth = 8, 
                        n.trees = res_temp[which.min(res_temp$RMSE),'n_tree'], 
                        shrinkage = res_temp[which.min(res_temp$RMSE),'lr'])
  time_model <- (proc.time() - ptm)[3]
  
  #Performance on testing samples
  pred_samples <- predict(model_temp, data_test[,-which(colnames(data_test) == 'Cooling_Load')],
                          n.trees = res_temp[which.min(res_temp$RMSE),'n_tree'])
  
  #Evaluate the performance based on testing profiles
  #Note that for 'dy' data set, iterative process is a much
  ptm_pred <- proc.time()
  if(length(grep(pattern = 'dy', x = data_set)) == 1) {
    pred_profiles <- foreach(n = seq(from = 1, to = dim(test_days)[1], by = 48), .combine = c) %do% {
      pred_temp <- c()
      for (m in 1:48) {
        if(m == 1) {
          pred_temp <- predict(model_temp, test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')],
                               n.trees = res_temp[which.min(res_temp$RMSE),'n_tree'])
        } else {
          #RAWdy
          if(data_set == 'RAWdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAWdy_original))])
            load_hist_raw <- c(load_hist[1:(48-m+1)],pred_temp)
            load_hist_upd <- (load_hist_raw - preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][1,])/(preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][2,] - preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][1,])
            test_days[n+m-1,paste0('Load_hist_',1:48,'_RAWdy')] <- load_hist_upd 
            pred_temp <- c(pred_temp, predict(model_temp, test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')],
                                              n.trees = res_temp[which.min(res_temp$RMSE),'n_tree']))
          }
          #LASTdy
          if(data_set == 'LASTdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAWdy_original))])
            load_hist_raw <- c(load_hist[1:(48-m+1)],pred_temp)
            load_hist_upd <- (load_hist_raw[length(load_hist_raw)] - preP_LASTdy$ranges[,'Load_hist_LASTdy'][1])/(preP_LASTdy$ranges[,'Load_hist_LASTdy'][2] - preP_LASTdy$ranges[,'Load_hist_LASTdy'][1]) 
            test_days[n+m-1,'Load_hist_LASTdy'] <- load_hist_upd 
            pred_temp <- c(pred_temp, predict(model_temp, test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')],
                                              n.trees = res_temp[which.min(res_temp$RMSE),'n_tree']))
          }
          #STATdy
          if(data_set == 'STATdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAW_original))])
            load_hist_raw <- c(load_hist[1:(48-m+1)],pred_temp)
            load_replace <- c(min(load_hist_raw), max(load_hist_raw), mean(load_hist_raw), sd(load_hist_raw))
            preP_min <- preP_STATdy$range[,c('Load_hist_min_1_STATdy','Load_hist_max_1_STATdy',
                                             'Load_hist_mean_1_STATdy','Load_hist_sd_1_STATdy')][1,]
            preP_max <- preP_STATdy$range[,c('Load_hist_min_1_STATdy','Load_hist_max_1_STATdy',
                                             'Load_hist_mean_1_STATdy','Load_hist_sd_1_STATdy')][2,]
            load_replace_upd <- as.numeric((load_replace - preP_min)/(preP_max - preP_min))
            test_days[n+m-1,c('Load_hist_min_1_STATdy','Load_hist_max_1_STATdy',
                              'Load_hist_mean_1_STATdy','Load_hist_sd_1_STATdy')] <- load_replace_upd
            pred_temp <- c(pred_temp, predict(model_temp, test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')],
                                              n.trees = res_temp[which.min(res_temp$RMSE),'n_tree']))
          }
          #DFTdy
          if(data_set == 'DFTdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAWdy_original))])
            load_hist_raw <- c(load_hist[1:(48-m+1)],pred_temp)
            freq <- order(Mod(fft(load_hist_raw))[1:(round(length(Mod(fft(load_hist_raw)))/2))], decreasing = T)[1:4]
            var_to_change <- paste0('Load_hist_T1_',1:4,'_DFTdy_Freq',freq)
            for(q in 1:length(var_to_change)) {
              if(var_to_change[q] %in% colnames(test_days)) {
                test_days[n+m-1,var_to_change[q]] <- 1
              } else {
                next
              }
            }
            pred_temp <- c(pred_temp, predict(model_temp, test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')],
                                              n.trees = res_temp[which.min(res_temp$RMSE),'n_tree']))
          }
          #AEdy
          if(data_set == 'AEdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAWdy_original))])
            load_hist_raw <-c(load_hist[1:(48-m+1)],pred_temp)
            load_hist_upd <- (load_hist_raw - preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][1,])/(preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][2,] - preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][1,])
            frame <- data.frame(matrix(data = load_hist_upd, nrow = 1))
            colnames(frame) <- paste0('Load_hist_',1:48,'_RAWdy')
            ae_get <- as.vector(h2o.deepfeatures(object = res_AEdy[[3]]$Model, data = as.h2o(frame), layer = 3))
            test_days[n+m-1,paste0('Load_hist_AEdy_',1:4)] <- ae_get
            pred_temp <- c(pred_temp, predict(model_temp, test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')],
                                              n.trees = res_temp[which.min(res_temp$RMSE),'n_tree']))
          }
          #RECENT
          if(data_set == 'RECENTdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAWdy_original))])
            load_hist_raw <- c(load_hist[1:(48-m+1)],pred_temp)
            load_replace <- c(mean(load_hist_raw[43:44]), mean(load_hist_raw[45:46]), mean(load_hist_raw[47:48]))
            load_replace_upd <- (load_replace - preP_RECENTdy$range[,paste0('Load_mean_',1:3,'_RECENTdy')][1,])/(preP_RECENTdy$range[,paste0('Load_mean_',1:3,'_RECENTdy')][2,]-preP_RECENTdy$range[,paste0('Load_mean_',1:3,'_RECENTdy')][1,])
            test_days[n+m-1,c('Load_mean_1_RECENTdy','Load_mean_2_RECENTdy','Load_mean_3_RECENTdy')] <- load_replace_upd
            pred_temp <- c(pred_temp, predict(model_temp, test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')],
                                              n.trees = res_temp[which.min(res_temp$RMSE),'n_tree']))
          }
        }
      }
      return(pred_temp)
    }
  } else {
    pred_profiles <- predict(model_temp, test_days[,-which(colnames(test_days) == 'Cooling_Load')],
                             n.trees = res_temp[which.min(res_temp$RMSE),'n_tree'])
  }
  time_pred <- (proc.time() - ptm_pred)[3]
  
  return(list(Model = model_temp, 
              Opt = res_temp,
              pred_test_samples = cbind.data.frame(Pred = pred_samples, Actual = data_test$Cooling_Load), 
              pred_test_profiles = cbind.data.frame(Pred = pred_profiles, Actual = test_days$Cooling_Load),
              Time_model = time_model,
              Time_pred = time_pred))
}

#Get the accuracy on testing samples
df_pred_test <- laply(.data = res_gbm, .fun = function(x) accuracy(f = x$pred_test_samples$Pred, x = x$pred_test_samples$Actual))
rownames(df_pred_test) <- c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy')
df_pred_test

#Get the accuracy on testing profiles
df_pred_profile <- laply(.data = res_gbm, .fun = function(x) accuracy(f = x$pred_test_profiles$Pred, x = x$pred_test_profiles$Actual))
rownames(df_pred_profile) <- c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy')
df_pred_profile

#Get the time on model development
df_time <- cbind(
  laply(.data = res_gbm, .fun = function(x) x$Time_model),
  laply(.data = res_gbm, .fun = function(x) x$Time_pred))
rownames(df_time) <- c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy')
df_time

###################################################################################################
###################################################################################################
#(5) Support vector regression
res_svg <- foreach(data_set = c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy')) %do% {
  eval(parse(text = paste0('data_train <- data_train_',data_set)))
  eval(parse(text = paste0('data_val <- data_val_',data_set)))
  eval(parse(text = paste0('data_test <- data_test_',data_set)))
  eval(parse(text = paste0('test_days <- test_days_',data_set)))
  
  #Note that only a proportion of data is used for parameter tuning
  set.seed(123)
  ind_opt <- sample(x = 1:dim(data_train)[1], size = 1000, replace = F)
  
  #Optimization on parameters
  res_temp <- foreach(par_sigma = 2^seq(from = -10, to = 0, by = 1), .combine = rbind.data.frame) %do% {
    res_temp_temp <- foreach(par_C = 2^seq(from = 5, to = 15, by = 1), .combine = rbind) %do% {
      temp <- ksvm(Cooling_Load ~., 
                   kernel = 'rbfdot', type = 'eps-svr', scaled = F, 
                   data = data_train[ind_opt,], 
                   C = par_C,
                   kpar = list(sigma = par_sigma))
      
      pred_val <- as.numeric(predict(temp, data_val[,-which(colnames(data_val) == 'Cooling_Load')]))
      c(par_sigma, par_C, accuracy(f = pred_val, x = data_val$Cooling_Load)[1:3]) 
    }
  }
  colnames(res_temp) <- c('par_sigma','par_C','ME','RMSE','MAE')
  
  #Develop the final model
  set.seed(123)
  ind_use <- sample(x = 1:dim(data_train)[1], size = 5000, replace = F)
  
  ptm <- proc.time()
  model_temp <- ksvm(Cooling_Load ~., 
                     kernel = 'rbfdot', type = 'eps-svr', scaled = F, 
                     data = data_train[ind_use,], 
                     C = res_temp[which.min(res_temp$RMSE), 'par_C'],
                     kpar = list(sigma = res_temp[which.min(res_temp$RMSE), 'par_sigma']))
  time_model <- (proc.time() - ptm)[3]
  
  #Performance on testing samples
  pred_samples <- as.numeric(predict(model_temp, data_test[,-which(colnames(data_test) == 'Cooling_Load')]))
  
  #Evaluate the performance based on testing profiles
  #Note that for 'dy' data set, iterative process is a much
  ptm_pred <- proc.time()
  if(length(grep(pattern = 'dy', x = data_set)) == 1) {
    pred_profiles <- foreach(n = seq(from = 1, to = dim(test_days)[1], by = 48), .combine = c) %do% {
      pred_temp <- c()
      for (m in 1:48) {
        if(m == 1) {
          pred_temp <- as.numeric(predict(model_temp, test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')]))
        } else {
          #RAWdy
          if(data_set == 'RAWdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAWdy_original))])
            load_hist_raw <- c(load_hist[1:(48-m+1)],pred_temp)
            load_hist_upd <- (load_hist_raw - preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][1,])/(preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][2,] - preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][1,])
            test_days[n+m-1,paste0('Load_hist_',1:48,'_RAWdy')] <- load_hist_upd 
            pred_temp <- c(pred_temp, as.numeric(predict(model_temp, test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')])))
          }
          #LASTdy
          if(data_set == 'LASTdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAWdy_original))])
            load_hist_raw <- c(load_hist[1:(48-m+1)],pred_temp)
            load_hist_upd <- (load_hist_raw[length(load_hist_raw)] - preP_LASTdy$ranges[,'Load_hist_LASTdy'][1])/(preP_LASTdy$ranges[,'Load_hist_LASTdy'][2] - preP_LASTdy$ranges[,'Load_hist_LASTdy'][1]) 
            test_days[n+m-1,'Load_hist_LASTdy'] <- load_hist_upd 
            pred_temp <- c(pred_temp, as.numeric(predict(model_temp, test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')])))
          }
          #STATdy
          if(data_set == 'STATdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAW_original))])
            load_hist_raw <- c(load_hist[1:(48-m+1)],pred_temp)
            load_replace <- c(min(load_hist_raw), max(load_hist_raw), mean(load_hist_raw), sd(load_hist_raw))
            preP_min <- preP_STATdy$range[,c('Load_hist_min_1_STATdy','Load_hist_max_1_STATdy',
                                             'Load_hist_mean_1_STATdy','Load_hist_sd_1_STATdy')][1,]
            preP_max <- preP_STATdy$range[,c('Load_hist_min_1_STATdy','Load_hist_max_1_STATdy',
                                             'Load_hist_mean_1_STATdy','Load_hist_sd_1_STATdy')][2,]
            load_replace_upd <- as.numeric((load_replace - preP_min)/(preP_max - preP_min))
            test_days[n+m-1,c('Load_hist_min_1_STATdy','Load_hist_max_1_STATdy',
                              'Load_hist_mean_1_STATdy','Load_hist_sd_1_STATdy')] <- load_replace_upd
            pred_temp <- c(pred_temp, as.numeric(predict(model_temp, test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')])))
          }
          #DFTdy
          if(data_set == 'DFTdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAWdy_original))])
            load_hist_raw <- c(load_hist[1:(48-m+1)],pred_temp)
            freq <- order(Mod(fft(load_hist_raw))[1:(round(length(Mod(fft(load_hist_raw)))/2))], decreasing = T)[1:4]
            var_to_change <- paste0('Load_hist_T1_',1:4,'_DFTdy_Freq',freq)
            for(q in 1:length(var_to_change)) {
              if(var_to_change[q] %in% colnames(test_days)) {
                test_days[n+m-1,var_to_change[q]] <- 1
              } else {
                next
              }
            }
            pred_temp <- c(pred_temp, as.numeric(predict(model_temp, test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')])))
          }
          #AEdy
          if(data_set == 'AEdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAWdy_original))])
            load_hist_raw <-c(load_hist[1:(48-m+1)],pred_temp)
            load_hist_upd <- (load_hist_raw - preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][1,])/(preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][2,] - preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][1,])
            frame <- data.frame(matrix(data = load_hist_upd, nrow = 1))
            colnames(frame) <- paste0('Load_hist_',1:48,'_RAWdy')
            ae_get <- as.vector(h2o.deepfeatures(object = res_AEdy[[3]]$Model, data = as.h2o(frame), layer = 3))
            test_days[n+m-1,paste0('Load_hist_AEdy_',1:4)] <- ae_get
            pred_temp <- c(pred_temp, as.numeric(predict(model_temp, test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')])))
          }
          #RECENT
          if(data_set == 'RECENTdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAWdy_original))])
            load_hist_raw <- c(load_hist[1:(48-m+1)],pred_temp)
            load_replace <- c(mean(load_hist_raw[43:44]), mean(load_hist_raw[45:46]), mean(load_hist_raw[47:48]))
            load_replace_upd <- (load_replace - preP_RECENTdy$range[,paste0('Load_mean_',1:3,'_RECENTdy')][1,])/(preP_RECENTdy$range[,paste0('Load_mean_',1:3,'_RECENTdy')][2,]-preP_RECENTdy$range[,paste0('Load_mean_',1:3,'_RECENTdy')][1,])
            test_days[n+m-1,c('Load_mean_1_RECENTdy','Load_mean_2_RECENTdy','Load_mean_3_RECENTdy')] <- load_replace_upd
            pred_temp <- c(pred_temp, as.numeric(predict(model_temp, test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')])))
          }
        }
      }
      return(pred_temp)
    }
  } else {
    pred_profiles <- as.numeric(predict(model_temp, test_days[,-which(colnames(test_days) == 'Cooling_Load')]))
  }
  time_pred <- (proc.time() - ptm_pred)[3]
  
  return(list(Model = model_temp, 
              Opt = res_temp,
              pred_test_samples = cbind.data.frame(Pred = pred_samples, Actual = data_test$Cooling_Load), 
              pred_test_profiles = cbind.data.frame(Pred = pred_profiles, Actual = test_days$Cooling_Load),
              Time_model = time_model,
              Time_pred = time_pred))
}

#Get the accuracy on testing samples
df_pred_test <- laply(.data = res_svg, .fun = function(x) accuracy(f = x$pred_test_samples$Pred, x = x$pred_test_samples$Actual))
rownames(df_pred_test) <- c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy')
df_pred_test

#Get the accuracy on testing profiles
df_pred_profile <- laply(.data = res_svg, .fun = function(x) accuracy(f = x$pred_test_profiles$Pred, x = x$pred_test_profiles$Actual))
rownames(df_pred_profile) <- c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy')
df_pred_profile

#Get the time on model development
df_time <- cbind(
  laply(.data = res_svg, .fun = function(x) x$Time_model),
  laply(.data = res_svg, .fun = function(x) x$Time_pred))
rownames(df_time) <- c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy')
df_time

###################################################################################################
###################################################################################################
#(6) XGBOOST
res_xgb <- foreach(data_set = c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy')) %do% {
  eval(parse(text = paste0('data_train <- data_train_',data_set)))
  eval(parse(text = paste0('data_val <- data_val_',data_set)))
  eval(parse(text = paste0('data_test <- data_test_',data_set)))
  eval(parse(text = paste0('test_days <- test_days_',data_set)))
  
  data_train_xgb <- xgb.DMatrix(data = as.matrix(data_train[,-which(colnames(data_train) == 'Cooling_Load')]), 
                                label = data_train$Cooling_Load)
  data_val_xgb <- xgb.DMatrix(data = as.matrix(data_val[,-which(colnames(data_val) == 'Cooling_Load')]), 
                              label = data_val$Cooling_Load)
  data_test_xgb <- xgb.DMatrix(data = as.matrix(data_test[,-which(colnames(data_test) == 'Cooling_Load')]), 
                               label = data_test$Cooling_Load)
  
  #Optimization on parameters
  xgb_grid <- expand.grid(max_depth = c(2, 4, 6, 8, 10),
                          eta = c(.01, .1))
  
  res_temp <- foreach(i = 1:dim(xgb_grid)[1], .combine = rbind.data.frame) %do% {
    temp <- xgb.cv(data = data_train_xgb, nfold = 3, 
                   objective = 'reg:linear', 
                   early.stop.round = 10, maximize = F,
                   booster = 'gbtree',
                   nrounds = 200, 
                   max.depth = xgb_grid[i,'max_depth'],
                   eta = xgb_grid[i,'eta'])
    return(c(xgb_grid[i,'max_depth'], xgb_grid[i,'eta'], 
             which.min(as.matrix(temp)[,3]),
             as.numeric(as.matrix(temp)[which.min(as.matrix(temp)[,3]),])))
  }
  colnames(res_temp) <- c('max_depth','eta','iter','train_error','train_std','val_error','val_std')
  
  #Develop the final model
  #Note that extra sampling is used since the computation time is too large
  ptm <- proc.time()
  model_temp <- xgb.train(data = data_train_xgb, 
                          objective = 'reg:linear', 
                          booster = 'gbtree', 
                          nrounds = 1000,
                          max.depth = xgb_grid[which.min(res_temp$val_error),'max_depth'], 
                          eta = xgb_grid[which.min(res_temp$val_error),'eta'])
  time_model <- (proc.time() - ptm)[3]
  
  #Performance on testing samples
  pred_samples <- predict(model_temp, as.matrix(data_test[,-which(colnames(data_test) == 'Cooling_Load')]))
  
  #Evaluate the performance based on testing profiles
  #Note that for 'dy' data set, iterative process is a much
  ptm_pred <- proc.time()
  if(length(grep(pattern = 'dy', x = data_set)) == 1) {
    pred_profiles <- foreach(n = seq(from = 1, to = dim(test_days)[1], by = 48), .combine = c) %do% {
      pred_temp <- c()
      for (m in 1:48) {
        if(m == 1) {
          pred_temp <- predict(model_temp, as.matrix(test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')]))
        } else {
          #RAWdy
          if(data_set == 'RAWdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAWdy_original))])
            load_hist_raw <- c(load_hist[1:(48-m+1)],pred_temp)
            load_hist_upd <- (load_hist_raw - preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][1,])/(preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][2,] - preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][1,])
            test_days[n+m-1,paste0('Load_hist_',1:48,'_RAWdy')] <- load_hist_upd 
            pred_temp <- c(pred_temp, predict(model_temp, as.matrix(test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')])))
          }
          #LASTdy
          if(data_set == 'LASTdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAWdy_original))])
            load_hist_raw <- c(load_hist[1:(48-m+1)],pred_temp)
            load_hist_upd <- (load_hist_raw[length(load_hist_raw)] - preP_LASTdy$ranges[,'Load_hist_LASTdy'][1])/(preP_LASTdy$ranges[,'Load_hist_LASTdy'][2] - preP_LASTdy$ranges[,'Load_hist_LASTdy'][1]) 
            test_days[n+m-1,'Load_hist_LASTdy'] <- load_hist_upd 
            pred_temp <- c(pred_temp, predict(model_temp, as.matrix(test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')])))
          }
          #STATdy
          if(data_set == 'STATdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAW_original))])
            load_hist_raw <- c(load_hist[1:(48-m+1)],pred_temp)
            load_replace <- c(min(load_hist_raw), max(load_hist_raw), mean(load_hist_raw), sd(load_hist_raw))
            preP_min <- preP_STATdy$range[,c('Load_hist_min_1_STATdy','Load_hist_max_1_STATdy',
                                             'Load_hist_mean_1_STATdy','Load_hist_sd_1_STATdy')][1,]
            preP_max <- preP_STATdy$range[,c('Load_hist_min_1_STATdy','Load_hist_max_1_STATdy',
                                             'Load_hist_mean_1_STATdy','Load_hist_sd_1_STATdy')][2,]
            load_replace_upd <- as.numeric((load_replace - preP_min)/(preP_max - preP_min))
            test_days[n+m-1,c('Load_hist_min_1_STATdy','Load_hist_max_1_STATdy',
                              'Load_hist_mean_1_STATdy','Load_hist_sd_1_STATdy')] <- load_replace_upd
            pred_temp <- c(pred_temp, predict(model_temp, as.matrix(test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')])))
          }
          #DFTdy
          if(data_set == 'DFTdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAWdy_original))])
            load_hist_raw <- c(load_hist[1:(48-m+1)],pred_temp)
            freq <- order(Mod(fft(load_hist_raw))[1:(round(length(Mod(fft(load_hist_raw)))/2))], decreasing = T)[1:4]
            var_to_change <- paste0('Load_hist_T1_',1:4,'_DFTdy_Freq',freq)
            for(q in 1:length(var_to_change)) {
              if(var_to_change[q] %in% colnames(test_days)) {
                test_days[n+m-1,var_to_change[q]] <- 1
              } else {
                next
              }
            }
            pred_temp <- c(pred_temp, predict(model_temp, as.matrix(test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')])))
          }
          #AEdy
          if(data_set == 'AEdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAWdy_original))])
            load_hist_raw <-c(load_hist[1:(48-m+1)],pred_temp)
            load_hist_upd <- (load_hist_raw - preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][1,])/(preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][2,] - preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][1,])
            frame <- data.frame(matrix(data = load_hist_upd, nrow = 1))
            colnames(frame) <- paste0('Load_hist_',1:48,'_RAWdy')
            ae_get <- as.vector(h2o.deepfeatures(object = res_AEdy[[3]]$Model, data = as.h2o(frame), layer = 3))
            test_days[n+m-1,paste0('Load_hist_AEdy_',1:4)] <- ae_get
            pred_temp <- c(pred_temp, predict(model_temp, as.matrix(test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')])))
          }
          #RECENT
          if(data_set == 'RECENTdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAWdy_original))])
            load_hist_raw <- c(load_hist[1:(48-m+1)],pred_temp)
            load_replace <- c(mean(load_hist_raw[43:44]), mean(load_hist_raw[45:46]), mean(load_hist_raw[47:48]))
            load_replace_upd <- (load_replace - preP_RECENTdy$range[,paste0('Load_mean_',1:3,'_RECENTdy')][1,])/(preP_RECENTdy$range[,paste0('Load_mean_',1:3,'_RECENTdy')][2,]-preP_RECENTdy$range[,paste0('Load_mean_',1:3,'_RECENTdy')][1,])
            test_days[n+m-1,c('Load_mean_1_RECENTdy','Load_mean_2_RECENTdy','Load_mean_3_RECENTdy')] <- load_replace_upd
            pred_temp <- c(pred_temp, predict(model_temp, as.matrix(test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')])))
          }
        }
      }
      return(pred_temp)
    }
  } else {
    pred_profiles <- predict(model_temp, as.matrix(test_days[,-which(colnames(test_days) == 'Cooling_Load')]))
  }
  time_pred <- (proc.time() - ptm_pred)[3]
  
  return(list(Model = model_temp, 
              Opt = res_temp,
              pred_test_samples = cbind.data.frame(Pred = pred_samples, Actual = data_test$Cooling_Load), 
              pred_test_profiles = cbind.data.frame(Pred = pred_profiles, Actual = test_days$Cooling_Load),
              Time_model = time_model,
              Time_pred = time_pred))
}

#Get the accuracy on testing samples
df_pred_test <- laply(.data = res_xgb, .fun = function(x) accuracy(f = x$pred_test_samples$Pred, x = x$pred_test_samples$Actual))
rownames(df_pred_test) <- c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy')
df_pred_test

#Get the accuracy on testing profiles
df_pred_profile <- laply(.data = res_xgb, .fun = function(x) accuracy(f = x$pred_test_profiles$Pred, x = x$pred_test_profiles$Actual))
rownames(df_pred_profile) <- c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy')
df_pred_profile

#Get the time on model development
df_time <- cbind(
  laply(.data = res_xgb, .fun = function(x) x$Time_model),
  laply(.data = res_xgb, .fun = function(x) x$Time_pred))
rownames(df_time) <- c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy')
df_time

###################################################################################################
###################################################################################################
#(7) MXNET
res_mxnet <- foreach(data_set = c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy')) %do% {
  eval(parse(text = paste0('data_train <- data_train_',data_set)))
  eval(parse(text = paste0('data_val <- data_val_',data_set)))
  eval(parse(text = paste0('data_test <- data_test_',data_set)))
  eval(parse(text = paste0('test_days <- test_days_',data_set)))
  
  #Optimization on parameters
  #Note that the num of hidden units is taken as the mean between input and output variable numbers
  #other choices for activation are 'tanh','sigmoid'
  mxnet_grid <- expand.grid(num_hidden_layer = c(1,2,3,4,5),
                            dropout_input = c(0, .01, .02, .03, .04, .05),
                            dropout_hidden = c(0, .01, .02, .03, .04, .05),
                            act = c('relu','tanh','sigmoid'))
  
  #Define the number of hidden units as the mean of input and output layers
  no_unit <- floor(dim(data_train)[2]/2)
  
  res_temp <- foreach(i = 1:dim(mxnet_grid)[1], .combine = rbind.data.frame) %do% {
    #Define the structure
    mx_input <- mx.symbol.Variable(name = 'data')
    mx_input_drop <- mx.symbol.Dropout(mx_input, p = mxnet_grid[i,'dropout_input'])
    if(mxnet_grid[i,'num_hidden_layer'] == 1){
      mx_hidden_1 <- mx.symbol.FullyConnected(mx_input_drop, num_hidden = no_unit)
      mx_act_1 <- mx.symbol.Activation(mx_hidden_1, act_type = as.character(mxnet_grid[i,'act']))
      mx_drop_final <- mx.symbol.Dropout(mx_act_1, p = mxnet_grid[i,'dropout_hidden'])
      mx_out_layer <- mx.symbol.FullyConnected(mx_drop_final, num_hidden = 1)
      mx_out <- mx.symbol.LinearRegressionOutput(mx_out_layer)
    } else {
      for(j in 1:mxnet_grid[i,'num_hidden_layer']) {
        if(j == 1) {
          mx_hidden_1 <- mx.symbol.FullyConnected(mx_input_drop, num_hidden = no_unit)
          mx_act_1 <- mx.symbol.Activation(mx_hidden_1, act_type = as.character(mxnet_grid[i,'act']))
          mx_drop_1 <- mx.symbol.Dropout(mx_act_1, p = mxnet_grid[i,'dropout_hidden'])
        } else {
          eval(parse(text = paste0('mx_hidden_',j,' <- mx.symbol.FullyConnected(mx_drop_',j-1,', num_hidden = no_unit)')))
          eval(parse(text = paste0('mx_act_',j,' <- mx.symbol.Activation(mx_hidden_',j,', act_type = as.character(mxnet_grid[i,\'act\']))')))
          eval(parse(text = paste0('mx_drop_',j,' <- mx.symbol.Dropout(mx_act_',j,', p = mxnet_grid[i,\'dropout_hidden\'])')))
        }
      }
      eval(parse(text = paste0('mx_out_layer <- mx.symbol.FullyConnected(mx_drop_',mxnet_grid[i,'num_hidden_layer'],', num_hidden = 1)')))
      mx_out <- mx.symbol.LinearRegressionOutput(mx_out_layer)
    }
    
    temp <- mx.model.FeedForward.create(
      symbol = mx_out, 
      X = data.matrix(data_train[,-which(colnames(data_train) == 'Cooling_Load')]), 
      y = data_train$Cooling_Load, 
      eval.data = list(data = data.matrix(data_val[,-which(colnames(data_val) == 'Cooling_Load')]), label = data_val$Cooling_Load),
      array.layout = 'rowmajor',
      ctx = mx.cpu(), num.round = 50, array.batch.size=100, 
      learning.rate=1e-6, momentum=0.9, eval.metric=mx.metric.rmse)
    
    pred_val <- as.vector(predict(temp, data.matrix(data_val[,-which(colnames(data_val) == 'Cooling_Load')]), array.layout = 'rowmajor'))
    
    return(c(mxnet_grid[i,'num_hidden_layer'], 
             no_unit, 
             mxnet_grid[i,'dropout_input'],
             mxnet_grid[i,'dropout_hidden'],
             accuracy(pred_val, data_val$Cooling_Load)[1:3]))
  }
  colnames(res_temp) <- c('num_hidden_layer','hidden_unit','dropout_input','dropout_hidden','ME','RMSE','MAE')
  res_temp$act <- mxnet_grid[,'act']
  
  #Develop the final model
  #Define the optimum structure
  ind_opt_parameter <- which.min(res_temp$RMSE)
  mx_input <- mx.symbol.Variable(name = 'data')
  mx_input_drop <- mx.symbol.Dropout(mx_input, p = res_temp[ind_opt_parameter,'dropout_input'])
  if(res_temp[ind_opt_parameter,'num_hidden_layer'] == 1){
    mx_hidden_1 <- mx.symbol.FullyConnected(mx_input_drop, num_hidden = no_unit)
    mx_act_1 <- mx.symbol.Activation(mx_hidden_1, act_type = as.character(mxnet_grid[i,'act']))
    mx_drop_final <- mx.symbol.Dropout(mx_act_1, p = mxnet_grid[i,'dropout_hidden'])
    mx_out_layer <- mx.symbol.FullyConnected(mx_drop_final, num_hidden = 1)
    mx_out_final <- mx.symbol.LinearRegressionOutput(mx_out_layer)
  } else {
    for(j in 1:res_temp[ind_opt_parameter,'num_hidden_layer']) {
      if(j == 1) {
        mx_hidden_1 <- mx.symbol.FullyConnected(mx_input_drop, num_hidden = no_unit)
        mx_act_1 <- mx.symbol.Activation(mx_hidden_1, act_type = as.character(res_temp[ind_opt_parameter,'act']))
        mx_drop_1 <- mx.symbol.Dropout(mx_act_1, p = res_temp[ind_opt_parameter,'dropout_hidden'])
      } else {
        eval(parse(text = paste0('mx_hidden_',j,' <- mx.symbol.FullyConnected(mx_drop_',j-1,', num_hidden = no_unit)')))
        eval(parse(text = paste0('mx_act_',j,' <- mx.symbol.Activation(mx_hidden_',j,', act_type = as.character(res_temp[ind_opt_parameter,\'act\']))')))
        eval(parse(text = paste0('mx_drop_',j,' <- mx.symbol.Dropout(mx_act_',j,', p = res_temp[ind_opt_parameter,\'dropout_hidden\'])')))
      }
    }
    eval(parse(text = paste0('mx_out_layer <- mx.symbol.FullyConnected(mx_drop_',res_temp[ind_opt_parameter,'num_hidden_layer'],', num_hidden = 1)')))
    mx_out_final <- mx.symbol.LinearRegressionOutput(mx_out_layer)
  }
  
  ptm <- proc.time()
  logger <- mx.metric.logger$new()
  model_temp <- mx.model.FeedForward.create(
    symbol = mx_out_final, 
    X = data.matrix(data_train[,-which(colnames(data_train) == 'Cooling_Load')]), 
    y = data_train$Cooling_Load, 
    eval.data = list(data = data.matrix(data_val[,-which(colnames(data_val) == 'Cooling_Load')]), label = data_val$Cooling_Load),
    array.layout = 'rowmajor',
    ctx = mx.cpu(), num.round = 500, array.batch.size=100, 
    learning.rate=1e-6, momentum=0.9, eval.metric=mx.metric.rmse,
    batch.end.callback = mx.callback.log.train.metric(period = 100, logger = logger),
    epoch.end.callback = mx.callback.save.checkpoint(prefix = data_set, period = 1))
  time_model <- (proc.time() - ptm)[3]
  
  #Get the best iteration
  best_iter <- which.min(logger$eval)
  
  #Get the best model
  model_temp <- mx.model.load(prefix = data_set, iteration = best_iter)
  
  #Performance on testing samples
  pred_samples <- as.vector(predict(model_temp, data.matrix(data_test[,-which(colnames(data_test) == 'Cooling_Load')]), array.layout = 'rowmajor'))
  
  #Evaluate the performance based on testing profiles
  #Note that for 'dy' data set, iterative process is a much
  ptm_pred <- proc.time()
  if(length(grep(pattern = 'dy', x = data_set)) == 1) {
    pred_profiles <- foreach(n = seq(from = 1, to = dim(test_days)[1], by = 48), .combine = c) %do% {
      pred_temp <- c()
      for (m in 1:48) {
        if(m == 1) {
          pred_temp <- as.vector(predict(model_temp, data.matrix(test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')]), array.layout = 'rowmajor'))
        } else {
          #RAWdy
          if(data_set == 'RAWdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAWdy_original))])
            load_hist_raw <- c(load_hist[1:(48-m+1)],pred_temp)
            load_hist_upd <- (load_hist_raw - preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][1,])/(preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][2,] - preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][1,])
            test_days[n+m-1,paste0('Load_hist_',1:48,'_RAWdy')] <- load_hist_upd 
            pred_temp <- c(pred_temp, as.vector(predict(model_temp, data.matrix(test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')]), array.layout = 'rowmajor')))
          }
          #LASTdy
          if(data_set == 'LASTdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAWdy_original))])
            load_hist_raw <- c(load_hist[1:(48-m+1)],pred_temp)
            load_hist_upd <- (load_hist_raw[length(load_hist_raw)] - preP_LASTdy$ranges[,'Load_hist_LASTdy'][1])/(preP_LASTdy$ranges[,'Load_hist_LASTdy'][2] - preP_LASTdy$ranges[,'Load_hist_LASTdy'][1]) 
            test_days[n+m-1,'Load_hist_LASTdy'] <- load_hist_upd 
            pred_temp <- c(pred_temp, as.vector(predict(model_temp, data.matrix(test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')]), array.layout = 'rowmajor')))
          }
          #STATdy
          if(data_set == 'STATdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAW_original))])
            load_hist_raw <- c(load_hist[1:(48-m+1)],pred_temp)
            load_replace <- c(min(load_hist_raw), max(load_hist_raw), mean(load_hist_raw), sd(load_hist_raw))
            preP_min <- preP_STATdy$range[,c('Load_hist_min_1_STATdy','Load_hist_max_1_STATdy',
                                             'Load_hist_mean_1_STATdy','Load_hist_sd_1_STATdy')][1,]
            preP_max <- preP_STATdy$range[,c('Load_hist_min_1_STATdy','Load_hist_max_1_STATdy',
                                             'Load_hist_mean_1_STATdy','Load_hist_sd_1_STATdy')][2,]
            load_replace_upd <- as.numeric((load_replace - preP_min)/(preP_max - preP_min))
            test_days[n+m-1,c('Load_hist_min_1_STATdy','Load_hist_max_1_STATdy',
                              'Load_hist_mean_1_STATdy','Load_hist_sd_1_STATdy')] <- load_replace_upd
            pred_temp <- c(pred_temp, as.vector(predict(model_temp, data.matrix(test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')]), array.layout = 'rowmajor')))
          }
          #DFTdy
          if(data_set == 'DFTdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAWdy_original))])
            load_hist_raw <- c(load_hist[1:(48-m+1)],pred_temp)
            freq <- order(Mod(fft(load_hist_raw))[1:(round(length(Mod(fft(load_hist_raw)))/2))], decreasing = T)[1:4]
            var_to_change <- paste0('Load_hist_T1_',1:4,'_DFTdy_Freq',freq)
            for(q in 1:length(var_to_change)) {
              if(var_to_change[q] %in% colnames(test_days)) {
                test_days[n+m-1,var_to_change[q]] <- 1
              } else {
                next
              }
            }
            pred_temp <- c(pred_temp, as.vector(predict(model_temp, data.matrix(test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')]), array.layout = 'rowmajor')))
          }
          #AEdy
          if(data_set == 'AEdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAWdy_original))])
            load_hist_raw <-c(load_hist[1:(48-m+1)],pred_temp)
            load_hist_upd <- (load_hist_raw - preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][1,])/(preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][2,] - preP_RAWdy$range[,paste0('Load_hist_',1:48,'_RAWdy')][1,])
            frame <- data.frame(matrix(data = load_hist_upd, nrow = 1))
            colnames(frame) <- paste0('Load_hist_',1:48,'_RAWdy')
            ae_get <- as.vector(h2o.deepfeatures(object = res_AEdy[[3]]$Model, data = as.h2o(frame), layer = 3))
            test_days[n+m-1,paste0('Load_hist_AEdy_',1:4)] <- ae_get
            pred_temp <- c(pred_temp, as.vector(predict(model_temp, data.matrix(test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')]), array.layout = 'rowmajor')))
          }
          #RECENT
          if(data_set == 'RECENTdy') {
            load_hist <- as.numeric(test_days_RAWdy_original[n+m-1,grep(pattern = 'Load_hist', x = colnames(test_days_RAWdy_original))])
            load_hist_raw <- c(load_hist[1:(48-m+1)],pred_temp)
            load_replace <- c(mean(load_hist_raw[43:44]), mean(load_hist_raw[45:46]), mean(load_hist_raw[47:48]))
            load_replace_upd <- (load_replace - preP_RECENTdy$range[,paste0('Load_mean_',1:3,'_RECENTdy')][1,])/(preP_RECENTdy$range[,paste0('Load_mean_',1:3,'_RECENTdy')][2,]-preP_RECENTdy$range[,paste0('Load_mean_',1:3,'_RECENTdy')][1,])
            test_days[n+m-1,c('Load_mean_1_RECENTdy','Load_mean_2_RECENTdy','Load_mean_3_RECENTdy')] <- load_replace_upd
            pred_temp <- c(pred_temp, as.vector(predict(model_temp, data.matrix(test_days[n+m-1,-which(colnames(test_days) == 'Cooling_Load')]), array.layout = 'rowmajor')))
          }
        }
      }
      return(pred_temp)
    }
  } else {
    pred_profiles <- as.vector(predict(model_temp, data.matrix(test_days[,-which(colnames(test_days) == 'Cooling_Load')]), array.layout = 'rowmajor'))
  }
  time_pred <- (proc.time() - ptm_pred)[3]
  
  return(list(Model = model_temp, 
              Opt = res_temp,
              pred_test_samples = cbind.data.frame(Pred = pred_samples, Actual = data_test$Cooling_Load), 
              pred_test_profiles = cbind.data.frame(Pred = pred_profiles, Actual = test_days$Cooling_Load),
              Time_model = time_model,
              Time_pred = time_pred))
}

#Get the accuracy on testing samples
df_pred_test <- laply(.data = res_mxnet, .fun = function(x) accuracy(f = x$pred_test_samples$Pred, x = x$pred_test_samples$Actual))
rownames(df_pred_test) <- c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy')
df_pred_test

#Get the accuracy on testing profiles
df_pred_profile <- laply(.data = res_mxnet, .fun = function(x) accuracy(f = x$pred_test_profiles$Pred, x = x$pred_test_profiles$Actual))
rownames(df_pred_profile) <- c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy')
df_pred_profile

#Get the time on model development
df_time <- cbind(
  laply(.data = res_mxnet, .fun = function(x) x$Time_model),
  laply(.data = res_mxnet, .fun = function(x) x$Time_pred))
rownames(df_time) <- c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy')
df_time

###################################################################################################
###################################################################################################
#Output the prediction results for summarization
out_com <- foreach(i = c('linear','elnet','rf','gbm','svg','xgb','mxnet')) %do% {
  eval(parse(text = paste0('df_temp_samples <- as.data.frame(t(laply(.data = res_',i,', .fun = function(x) x$pred_test_samples$Pred)))')))
  colnames(df_temp_samples) <- paste0(i,'_',c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy'))
  
  eval(parse(text = paste0('df_temp_profiles <- as.data.frame(t(laply(.data = res_',i,', .fun = function(x) x$pred_test_profiles$Pred)))')))
  colnames(df_temp_profiles) <- paste0(i,'_',c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy'))
  
  return(list(samples = df_temp_samples, profiles = df_temp_profiles))
}

#Output 1: Test samples
out_com_samples <- do.call(what = cbind.data.frame, args = llply(.data = out_com, .fun = function(x) x$samples))
out_com_samples$Actual <- res_mxnet[[1]]$pred_test_samples$Actual
dim(out_com_samples)
head(out_com_samples)
write.csv(x = out_com_samples, file = '/Users/Fan/Desktop/JCIT_Prediction/Profile_Refactor/test_samples.csv', row.names = F)

#Output 2: Test profiles
out_com_profiles <- do.call(what = cbind.data.frame, args = llply(.data = out_com, .fun = function(x) x$profiles))
out_com_profiles$Actual <- res_mxnet[[1]]$pred_test_profiles$Actual
dim(out_com_profiles)
head(out_com_profiles)
write.csv(x = out_com_profiles, file = '/Users/Fan/Desktop/JCIT_Prediction/Profile_Refactor/test_profiles.csv', row.names = F)

aaply(.data = as.matrix(out_com_samples[,-dim(out_com_samples)[2]]), .margins = 2, .fun = function(x) accuracy(x, out_com_samples$Actual))

res_profiles <- adply(.data = as.matrix(out_com_profiles[,-dim(out_com_samples)[2]]), 
                      .margins = 2, .fun = function(x) accuracy(x, out_com_profiles$Actual))
res_profiles <- res_profiles[with(res_profiles, order(RMSE, decreasing = F)),]
res_profiles

#Output 3: H2o model for AE and AEdy
h2o.saveModel(object = res_AEdy[[3]]$Model, path = '/Users/Fan/Desktop/JCIT_Prediction/Profile_Refactor/')
h2o.saveModel(object = res_AE[[3]]$Model, path = '/Users/Fan/Desktop/JCIT_Prediction/Profile_Refactor/')

res_AE[[3]]$Model <- h2o.loadModel(path = '/Users/Fan/Desktop/JCIT_Prediction/Profile_Refactor/DeepLearning_model_R_AE')
res_AEdy[[3]]$Model <- h2o.loadModel(path = '/Users/Fan/Desktop/JCIT_Prediction/Profile_Refactor/DeepLearning_model_R_AEdy')

#Output 4: Prediction on training data
train_linear <- foreach(i = c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy'), .combine = cbind.data.frame) %do% {
  ind <- which(c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy') == i)
  eval(parse(text = paste0('pred <- as.vector(predict(res_linear[[ind]]$Model, data_train_',i,'[,-which(colnames(data_train_',i,') == \'Cooling_Load\')]))')))
  return(pred)
}
dim(train_linear)
colnames(train_linear) <- paste0('linear_',c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy'))
summary(train_linear)

train_elnet <- foreach(i = c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy'), .combine = cbind.data.frame) %do% {
  ind <- which(c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy') == i)
  eval(parse(text = paste0('pred <- as.vector(predict(res_elnet[[ind]]$Model, as.matrix(data_train_',i,'[,-which(colnames(data_train_',i,') == \'Cooling_Load\')])))')))
  return(pred)
}
dim(train_elnet)
colnames(train_elnet) <- paste0('elnet_',c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy'))
summary(train_elnet)

train_rf <- foreach(i = c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy'), .combine = cbind.data.frame) %do% {
  ind <- which(c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy') == i)
  eval(parse(text = paste0('pred <- as.vector(predict(res_rf[[ind]]$Model, data_train_',i,'[,-which(colnames(data_train_',i,') == \'Cooling_Load\')]))')))
  return(pred)
}
dim(train_rf)
colnames(train_rf) <- paste0('rf_',c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy'))
summary(train_rf)

train_gbm <- foreach(i = c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy'), .combine = cbind.data.frame) %do% {
  ind <- which(c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy') == i)
  eval(parse(text = paste0('pred <- as.vector(predict(res_gbm[[ind]]$Model, data_train_',i,'[,-which(colnames(data_train_',i,') == \'Cooling_Load\')], n.trees = res_gbm[[ind]]$Opt[which.min(res_gbm[[ind]]$Opt$RMSE),\'n_tree\']))')))
  return(pred)
}
dim(train_gbm)
colnames(train_gbm) <- paste0('gbm_',c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy'))
summary(train_gbm)

train_svg <- foreach(i = c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy'), .combine = cbind.data.frame) %do% {
  ind <- which(c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy') == i)
  eval(parse(text = paste0('pred <- as.vector(predict(res_svg[[ind]]$Model, data_train_',i,'[,-which(colnames(data_train_',i,') == \'Cooling_Load\')]))')))
  return(pred)
}
dim(train_svg)
colnames(train_svg) <- paste0('svg_',c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy'))
summary(train_svg)

train_xgb <- foreach(i = c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy'), .combine = cbind.data.frame) %do% {
  ind <- which(c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy') == i)
  eval(parse(text = paste0('pred <- as.vector(predict(res_xgb[[ind]]$Model, as.matrix(data_train_',i,'[,-which(colnames(data_train_',i,') == \'Cooling_Load\')])))')))
  return(pred)
}
dim(train_xgb)
colnames(train_xgb) <- paste0('xgb_',c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy'))
summary(train_xgb)

train_mxnet <- foreach(i = c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy'), .combine = cbind.data.frame) %do% {
  ind <- which(c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy') == i)
  eval(parse(text = paste0('pred <- as.vector(predict(res_mxnet[[ind]]$Model, as.matrix(data_train_',i,'[,-which(colnames(data_train_',i,') == \'Cooling_Load\')]), array.layout = \'rowmajor\'))')))
  return(pred)
}
dim(train_mxnet)
colnames(train_mxnet) <- paste0('mxnet_',c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy'))
summary(train_mxnet)

train_com <- cbind.data.frame(train_linear, train_elnet, train_rf, train_gbm, train_svg, train_xgb, train_mxnet, Actual = data_train_BASIC$Cooling_Load)
dim(train_com)
head(train_com)

aaply(.data = as.matrix(train_com[,-dim(train_com)[2]]), .margins = 2, .fun = function(x) accuracy(x, train_com$Actual))
write.csv(x = train_com, file = '/Users/Fan/Desktop/JCIT_Prediction/Profile_Refactor/train_samples.csv', row.names = F)

#Output 5: Prediction on validation data
val_linear <- foreach(i = c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy'), .combine = cbind.data.frame) %do% {
  ind <- which(c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy') == i)
  eval(parse(text = paste0('pred <- as.vector(predict(res_linear[[ind]]$Model, data_val_',i,'[,-which(colnames(data_val_',i,') == \'Cooling_Load\')]))')))
  return(pred)
}
dim(val_linear)
colnames(val_linear) <- paste0('linear_',c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy'))
summary(val_linear)

val_elnet <- foreach(i = c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy'), .combine = cbind.data.frame) %do% {
  ind <- which(c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy') == i)
  eval(parse(text = paste0('pred <- as.vector(predict(res_elnet[[ind]]$Model, as.matrix(data_val_',i,'[,-which(colnames(data_val_',i,') == \'Cooling_Load\')])))')))
  return(pred)
}
dim(val_elnet)
colnames(val_elnet) <- paste0('elnet_',c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy'))
summary(val_elnet)

val_rf <- foreach(i = c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy'), .combine = cbind.data.frame) %do% {
  ind <- which(c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy') == i)
  eval(parse(text = paste0('pred <- as.vector(predict(res_rf[[ind]]$Model, data_val_',i,'[,-which(colnames(data_val_',i,') == \'Cooling_Load\')]))')))
  return(pred)
}
dim(val_rf)
colnames(val_rf) <- paste0('rf_',c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy'))
summary(val_rf)

val_gbm <- foreach(i = c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy'), .combine = cbind.data.frame) %do% {
  ind <- which(c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy') == i)
  eval(parse(text = paste0('pred <- as.vector(predict(res_gbm[[ind]]$Model, data_val_',i,'[,-which(colnames(data_val_',i,') == \'Cooling_Load\')], n.trees = res_gbm[[ind]]$Opt[which.min(res_gbm[[ind]]$Opt$RMSE),\'n_tree\']))')))
  return(pred)
}
dim(val_gbm)
colnames(val_gbm) <- paste0('gbm_',c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy'))
summary(val_gbm)

val_svg <- foreach(i = c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy'), .combine = cbind.data.frame) %do% {
  ind <- which(c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy') == i)
  eval(parse(text = paste0('pred <- as.vector(predict(res_svg[[ind]]$Model, data_val_',i,'[,-which(colnames(data_val_',i,') == \'Cooling_Load\')]))')))
  return(pred)
}
dim(val_svg)
colnames(val_svg) <- paste0('svg_',c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy'))
summary(val_svg)

val_xgb <- foreach(i = c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy'), .combine = cbind.data.frame) %do% {
  ind <- which(c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy') == i)
  eval(parse(text = paste0('pred <- as.vector(predict(res_xgb[[ind]]$Model, as.matrix(data_val_',i,'[,-which(colnames(data_val_',i,') == \'Cooling_Load\')])))')))
  return(pred)
}
dim(val_xgb)
colnames(val_xgb) <- paste0('xgb_',c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy'))
summary(val_xgb)

val_mxnet <- foreach(i = c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy'), .combine = cbind.data.frame) %do% {
  ind <- which(c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy') == i)
  eval(parse(text = paste0('pred <- as.vector(predict(res_mxnet[[ind]]$Model, as.matrix(data_val_',i,'[,-which(colnames(data_val_',i,') == \'Cooling_Load\')]), array.layout = \'rowmajor\'))')))
  return(pred)
}
dim(val_mxnet)
colnames(val_mxnet) <- paste0('mxnet_',c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy'))
summary(val_mxnet)

val_com <- cbind.data.frame(val_linear, val_elnet, val_rf, val_gbm, val_svg, val_xgb, val_mxnet, Actual = data_val_BASIC$Cooling_Load)
dim(val_com)
head(val_com)

aaply(.data = as.matrix(val_com[,-dim(val_com)[2]]), .margins = 2, .fun = function(x) accuracy(x, val_com$Actual))
write.csv(x = val_com, file = '/Users/Fan/Desktop/JCIT_Prediction/Profile_Refactor/val_samples.csv', row.names = F)

#Output 6: Optimization grid
opt <- foreach(i = c('elnet','rf','gbm','svg','xgb','mxnet')) %do% {
  opt_temp <- foreach(j = c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy'), .combine = rbind.data.frame) %do% {
    ind <- which(c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy') == j)
    temp <- eval(parse(text = paste0('res_',i,'[[',ind,']]$Opt')))
    temp$Label <- paste0(i,'_',j)
    return(temp)
  }
  write.csv(x = opt_temp, file = paste0('/Users/Fan/Desktop/JCIT_Prediction/Profile_Refactor/Opt_',i,'.csv'), row.names = F)
  return(opt_temp)
}

#Output 7: The workspace
save.image(file = '/Users/Fan/Desktop/JCIT_Prediction/Profile_Refactor/Profile_Refactor.RData')

###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
#Ensembles
#Check the accuracy on testing samples
dim(out_com_samples)
head(out_com_samples)
res_test <- adply(.data = as.matrix(out_com_samples[,-dim(out_com_samples)[2]]), .margins = 2, .fun = function(x) accuracy(f = x, x = out_com_samples$Actual))
res_test <- res_test[with(res_test, order(RMSE, decreasing = F)),]
res_test

#Check the accuracy on testing profiles
dim(out_com_profiles)
head(out_com_profiles)
res_profiles <- adply(.data = as.matrix(out_com_profiles[,-dim(out_com_profiles)[2]]), .margins = 2, .fun = function(x) accuracy(f = x, x = out_com_profiles$Actual))
res_profiles <- res_profiles[with(res_profiles, order(RMSE, decreasing = F)),]
res_profiles

#Develop ensembles using ridge regression
head(val_com)

model_ensemble_cv <- cv.glmnet(x = as.matrix(train_com[,-which(colnames(train_com) == 'Actual')]), nfolds = 3, 
                               y = train_com$Actual, family = 'gaussian', alpha = 0, standardize = T)
model_ensemble_cv$lambda.min

model_ensemble <- glmnet(x = as.matrix(val_com[,-which(colnames(val_com) == 'Actual')]), 
                         y = val_com$Actual, family = 'gaussian', 
                         alpha = 0, lambda = model_ensemble_cv$lambda.min, standardize = T)

#Ensemble performance on testing samples
pred_samples_ensemble <- as.vector(predict(object = model_ensemble, as.matrix(out_com_samples[,-dim(out_com_samples)[2]])))
accuracy(f = pred_samples_ensemble, x = out_com_samples$Actual)

#Ensemble performance on testing profiles
pred_profiles_ensemble <- as.vector(predict(object = model_ensemble, as.matrix(out_com_profiles[,-dim(out_com_profiles)[2]])))
accuracy(f = pred_profiles_ensemble, x = out_com_profiles$Actual)

###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
#Visualization 
head(out_com_profiles)
library(reshape2)
df_vis <- melt(data = out_com_profiles)
df_vis$X <- rep(1:(dim(df_vis)[1]/length(unique(df_vis$variable))), 2)
head(df_vis)

ggplot(data = df_vis[which(df_vis$variable %in% c('xgb_AEdy','Actual')),]) + 
  geom_line(aes(x = X, y = value, color = variable, linetype = variable)) +
  xlab('Time (30-min)') + ylab('Cooling Load (kW)') +
  theme(panel.background = element_rect(fill = 'white', colour = 'black'),
        legend.position = 'top',
        legend.title = element_blank())

#Check the accuracy on profiles for each method
res_profiles[grep(pattern = 'svg',x = res_profiles$X1),]

#Check the accuracy on test samples for each method
res_test[grep(pattern = 'mxnet',x = res_test$X1),]

#Check the computation time for model development
time_model <- foreach(i = c('linear','elnet','rf','gbm','svg','xgb','mxnet'), .combine = rbind.data.frame) %do% {
  time_model_temp <- foreach(j = c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy'), .combine = c) %do% {
    ind <- which(c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy') == j)
    eval(parse(text = paste0('res_',i,'[[',ind,']]$Time_model')))
  }
  return(time_model_temp)
}
colnames(time_model) <- c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy')    
rownames(time_model) <- c('linear','elnet','rf','gbm','svg','xgb','mxnet')
time_model

#Check the computation time for model prediction
time_pred <- foreach(i = c('linear','elnet','rf','gbm','svg','xgb','mxnet'), .combine = rbind.data.frame) %do% {
  time_pred_temp <- foreach(j = c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy'), .combine = c) %do% {
    ind <- which(c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy') == j)
    eval(parse(text = paste0('res_',i,'[[',ind,']]$Time_pred')))
  }
  return(time_pred_temp)
}
colnames(time_pred) <- c('BASIC','RAW','STAT','DFT','AE','RAWdy','STATdy','DFTdy','AEdy','RECENTdy','LASTdy')    
rownames(time_pred) <- c('linear','elnet','rf','gbm','svg','xgb','mxnet')
time_pred

###################################################################################################
###################################################################################################



