library(tidyverse)
library(reshape2)

housing = read.csv('D:/RAG/housing.csv')

head(housing)
summary(housing)

par(mfrow=c(2,5)) #设置图形布局为2行5列，用于后续绘制多个直方图。

colnames(housing)#查看数据框的列名
ggplot(data = melt(housing), mapping = aes(x = value)) + 
  geom_histogram(bins = 30) + facet_wrap(~variable, scales = 'free_x')
#melt(housing)将数据框从宽格式转换为长格式，便于绘制直方图。
# aes(x = value)：指定x轴为变量的值。
#geom_histogram(bins = 30)：绘制直方图，设置30个区间。
#facet_wrap(~variable, scales = 'free_x')：按变量分面绘制直方图，x轴的尺度自由调整。

housing$total_bedrooms[is.na(housing$total_bedrooms)] = median(housing$total_bedrooms , na.rm = TRUE)
#将total_bedrooms列中的缺失值（NA）替换为该列的中位数。


housing$mean_bedrooms = housing$total_bedrooms/housing$households
housing$mean_rooms = housing$total_rooms/housing$households
head(housing)

drops = c('total_bedrooms', 'total_rooms')
housing = housing[ , !(names(housing) %in% drops)]#删除total_bedrooms和total_rooms列，因为它们已被新特征替代
head(housing)

categories = unique(housing$ocean_proximity)
cat_housing = data.frame(ocean_proximity = housing$ocean_proximity)
#提取ocean_proximity列的唯一值作为分类变量。
#创建一个新的数据框cat_housing，用于存储分类变量的独热编码。
table(cat_housing)

for(cat in categories){
  cat_housing[,cat] = rep(0, times= nrow(cat_housing))
}
head(cat_housing) 
# 遍历ocean_proximity的所有唯一类别（存储在categories中）。
#为每个类别创建一个新列，并初始化为0。


for(i in 1:length(cat_housing$ocean_proximity)){
  cat = as.character(cat_housing$ocean_proximity[i])
  cat_housing[,cat][i] = 1
}
head(cat_housing)
#为每个分类创建一个新的列，并初始化为0。
#遍历ocean_proximity列，将对应的分类列设置为1，完成独热编码。

cat_columns = names(cat_housing)
keep_columns = cat_columns[cat_columns != 'ocean_proximity']
cat_housing = select(cat_housing,one_of(keep_columns))
#删除ocean_proximity列，保留独热编码后的列。
tail(cat_housing)#查看独热编码后的数据框的最后6行。

colnames(housing)

drops = c('ocean_proximity','median_house_value')
housing_num =  housing[ , !(names(housing) %in% drops)]

head(housing_num)#删除ocean_proximity和median_house_value列，保留数值型特征

scaled_housing_num = scale(housing_num)#对数值型特征进行标准化（Z-score标准化），使其均值为0，标准差为1
head(scaled_housing_num)

cleaned_housing = cbind(cat_housing, scaled_housing_num, median_house_value=housing$median_house_value)
head(cleaned_housing)
#将独热编码后的分类变量、标准化后的数值特征和目标变量median_house_value合并成一个新的数据框cleaned_housing


set.seed(1738) # Set a random seed so that same sample can be reproduced in future runs

sample = sample.int(n = nrow(cleaned_housing), size = floor(.8*nrow(cleaned_housing)), replace = F)
train = cleaned_housing[sample, ] #just the samples
test  = cleaned_housing[-sample, ] #everything but the samples
#设置随机种子以保证结果可重复。
#随机抽取80%的数据作为训练集，剩余20%作为测试集

head(train)
nrow(train) + nrow(test) == nrow(cleaned_housing)

library('boot')

?cv.glm # note the K option for K fold cross validation

glm_house = glm(median_house_value~median_income+mean_rooms+population, data=cleaned_housing)
k_fold_cv_error = cv.glm(cleaned_housing , glm_house, K=5)
#使用广义线性模型（GLM）拟合目标变量median_house_value与median_income、mean_rooms和population的关系。
k_fold_cv_error$delta
#对GLM模型进行5折交叉验证，计算模型的误差。


glm_cv_rmse = sqrt(k_fold_cv_error$delta)[1]
glm_cv_rmse #off by about $83,000... it is a start
#计算交叉验证的均方根误差（RMSE）

names(glm_house) #what parts of the model are callable?
glm_house$coefficients
#查看GLM模型的系数。

install.packages("randomForest")
library('randomForest')

names(train)

set.seed(1738)

train_y = train[,'median_house_value']
train_x = train[, names(train) !='median_house_value']
#取训练集的目标变量train_y和特征变量train_x

head(train_y)
head(train_x)

#some people like weird r format like this... I find it causes headaches
#rf_model = randomForest(median_house_value~. , data = train, ntree =500, importance = TRUE)
rf_model = randomForest(train_x, y = train_y , ntree = 500, importance = TRUE)
names(rf_model)
#使用随机森林模型训练数据，设置500棵树，并计算特征重要性。

rf_model$importance
#查看随机森林模型中每个特征的重要性



oob_prediction = predict(rf_model) #leaving out a data source forces OOB predictions

#you may have noticed that this is avaliable using the $mse in the model options.
#but this way we learn stuff!
train_mse = mean(as.numeric((oob_prediction - train_y)^2))
oob_rmse = sqrt(train_mse)
oob_rmse
#使用袋外数据（OOB）预测训练集，并计算均方根误差（RMSE）
test_y = test[,'median_house_value']
test_x = test[, names(test) !='median_house_value']


y_pred = predict(rf_model , test_x)
test_mse = mean(((y_pred - test_y)^2))
test_rmse = sqrt(test_mse)
test_rmse
##使用随机森林模型对测试集进行预测，并计算测试集的均方根误差（RMSE）


# 假设模型保存为文件
rf_model <- readRDS("randomForest_model.rds")

# 假设有一个新的数据框 `new_data`，其中包含与训练集相同的特征
new_predictions <- predict(rf_model, new_data)
importance <- rf_model$importance

# 保存随机森林模型
saveRDS(rf_model, "randomForest_model.rds")

# 保存GLM模型
saveRDS(glm_house, "glm_model.rds")

# 加载随机森林模型
rf_model <- readRDS("randomForest_model.rds")

# 加载GLM模型
glm_house <- readRDS("glm_model.rds")