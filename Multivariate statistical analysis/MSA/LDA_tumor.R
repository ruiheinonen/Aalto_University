setwd("~/Desktop/Classes and Learning material /Multivariate Statistical Analysis 19/Project")

data = read.csv("data-2.csv", header = TRUE)
data = data[,2:32] # Drop first column because its an ID


library(rmarkdown)

# The first 10 columns represent mean values, 
# The following 10 columns represent standard error of the mean values
# The following 10 columns represent "worst" or largest mean value for x 

summary(data) 

# By examining the summary table, we see that  
# The means and medians for different mean values vary a lot
# There seems to be only a little variation in different standard error values
# The means and medians for different worst valeus vary a lot 
# The data seems to be somewhat clustered as there is a lot of variation in terms of 
# location and scatter between groups of variables. For example, perimeter_se and 
# area_se seem to have have higher median and mean values than other _se variables. 
# In addition, the max values for these variables are much greater than the 
# corresponding mean and median values giving indication of outliers and scatterness.


# UNIVARIATE ANALYSIS 
##########################################################################################

# To further examine the numerical explanatory variables, I will generate box plots for 
# benign and malignant tumorsin three groups: first the “mean” values (Figure 1.), 
# secondly the “standard error” values (Figure 2.) and lastly the “worst” values (Figure 3.). 
# Before plotting the box plots, I will center and scale the explanatory variables to 
# address the broad difference of location and scatter within the variables, as seen from 
# the summary table. 

library(reshape2)
library(ggplot2)

scale_x1 = scale(x = data[2:11], center = TRUE, scale = TRUE) # center = - mean, scale = / std
data_box1 = cbind(data[1], scale_x1)
box1 = melt(data_box1, id.var = "diagnosis")

plot_box1 = ggplot(data = box1 , aes(x=variable, y=value, fill=diagnosis)) + 
  geom_boxplot(aes(fill=diagnosis)) +
  scale_fill_manual(values = c("grey0", "grey100"))

plot_box1 + theme(axis.text.x = element_text(angle = 90, face = "bold"), 
                  panel.background = element_rect(fill = "white", colour = "grey50"))

# The median for fractal_dimension_mean is almost equal for Malignant and Benign tumors
# Thus this variable does not give good information for classification 

scale_x2 = scale(x = data[12:21], center = TRUE, scale = TRUE)
data_box2 = cbind(data[1], scale_x2)
box2 = melt(data_box2, id.var = "diagnosis")

plot_box2 = ggplot(data = box2 , aes(x=variable, y=value)) + 
  geom_boxplot(aes(fill=diagnosis)) +
  scale_fill_manual(values = c("grey0", "grey100"))

plot_box2 + theme(axis.text.x = element_text(angle = 90, face = "bold"),
                  panel.background = element_rect(fill = "white", colour = "grey50"))

# The medians for texture se, smoothness_se, symmetry_se and fractal_dimension_se are 
# almost equal. Thus these variables do not give good information for classification 

scale_x3 = scale(x = data[22:31], center = TRUE, scale = TRUE)
data_box3 = cbind(data[1], scale_x3)
box3 = melt(data_box3, id.var = "diagnosis")

plot_box3 = ggplot(data = box3 , aes(x=variable, y=value)) + 
  geom_boxplot(aes(fill=diagnosis)) + 
  scale_fill_manual(values = c("grey0", "grey100"))

plot_box3 + theme(axis.text.x = element_text(angle = 90, face = "bold"),
                  panel.background = element_rect(fill = "white", colour = "grey50"))

# The medians between Malignant and Benign tumor across variables are quite dissimilar 
# Thus these varibles are so far good to go for the model 


# By examining the box plots, I am able drop insignificant features, which include 
# *fractal_dimension_mean*, *texture se*, *smoothness_se*, *symmetry_se* and 
# *fractal_dimension_se*. These variables do not give good information for classification as 
# their medians for both benign and malignant tumors are almost equal, and hence 
# I will exclude these variables from further examination of the data. As the nature of the 
# features is somewhat similar, one could also explore the variances of the variables and 
# select features based on those variables that exhibit most variation because high variation 
# helps to distinguish patterns in the data. However, I will not apply this approach in this 
# study, instead, next I will perform feature selection based on examination of the pairwise 
# correlation coefficients.

##############################################################################################

# BIVARIATE ANALYSIS
##############################################################################################
library(corrplot)

data2 = cbind(data[1], scale_x1, scale_x2, scale_x3)
data2 = data2[-which(names(data2) %in% c("fractal_dimension_mean", "texture_se",
                                         "smoothness_se", "symmetry_se",
                                         "fractal_dimension_se"))]
head(data2)
plot_corr = data2[2:26]
corr = cor(plot_corr)

corrplot(corr, method = "color", col = gray.colors(100), type = "upper", order = "hclust", 
         number.cex = .5, addCoef.col = "black", tl.col = "black", tl.srt = 90, 
         insig = "blank", diag = FALSE)

# Exploration of the correlation matrix allows us to identify high pairwise correlations.
# In order to select the most important features for the model, I follow Ferrar and Glauber 
# (1967) and constrain pairwise correlations between any two explanatory variables to be less 
# than 0,9. Hence, will exclude *texture_worst*, *radius_se*, *perimeter_se*, *radius_mean*, 
# *perimeter_mean*, *area_worst*, *radius_worst*, *perimeter_worst* and *concave.points_mean* 
# from the data on this point forward.

# DROP 
data2 = data2[-which(names(data2) %in% c("texture_worst", "radius_se", "perimeter_se",
                                         "radius_mean", "perimeter_mean", "area_worst",
                                         "radius_worst", "perimeter_worst",
                                         "concave.points_mean"))]

# Alternative way to drop features according to a given cut-off point
# drop_these = colnames(data2)[findCorrelation(corr, cutoff = 0.9, verbose = FALSE)]
# drop_these # remove high pairwise correlations 

# MODEL BUILDING 
##############################################################################################

library(MASS)

# SPlit into tarining and testing data sets 
set.seed(101) 
sample = sample.int(n = nrow(data2), size = floor(.75*nrow(data2)), replace = F)
train = data[sample, ]
test = data[-sample, ]

data2_lda = lda(diagnosis~. ,data = train)
data2_predict_lda = predict(data2_lda, newdata = test)

# MODEL EVALUATION
##############################################################################################

library(ROCR)
library(gplots)
library(caret)

# CONFUSION MATRIX 
data2_cv = lda(diagnosis~. ,data = data2, CV = T)
result = data.frame(Actual_class = data2[,1], Predcited_class = data2_cv$class)
tab = table(result)
tab

# This matrix shows the misclassification rates of benign and malignant tumors. 
# According to this table, the LDA model misclassifies 26 malignant tumors as benign 
# and 1 benign tumor as malignant.

# Accuracy in the test set 
ct = table(test$diagnosis, data2_predict_lda$class)
diag(prop.table(ct, 1))
sum(diag(prop.table(ct)))

# Accuracy in teh training set 
data2_predict_lda_train = predict(data2_lda, newdata = train)
ct = table(train$diagnosis, data2_predict_lda_train$class)
diag(prop.table(ct, 1))
sum(diag(prop.table(ct)))

# AUROC 
prediction = prediction(as.numeric(data2_predict_lda$class), as.numeric(test$diagnosis))
roc_performance = performance(prediction, measure = "tpr", x.measure = "fpr")
auc = performance(prediction, measure = "auc")
auc = auc@y.values

# This figure reports the true positive rate against the false positive rate and the AUC score 
# (0,906). According to the AUC score, the suggested LDA model will be able to distinguish 
# positive and negative classes with a 90.6% probability. The linear line represents an 
# indicative function that does not differ from a simple 50-50 guess.

plot(roc_performance)
abline(a=0, b= 1)
text(x = .25, y = .65 ,paste("AUC = ", round(auc[[1]],3), sep = ""))

##############################################################################################

# APPENDIX

library(purrr)
library(tidyr)

#dev.new()
data %>%
  keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram()
#dev.off()


