# Objective: Analyze the emotional significance of raga mayamalavagoulva w.r.t
# others raga
# Input dataset: Emotion class prediciton from the Adaboost model for 5 songs within
# each raga
# Total obs: 192

setwd("D:/PhD Program/Final Research/Dissertation")

library(lsmeans)
library(multcomp)
library(multcompView)
library(psych)
library(moments)
library(FSA)
library(rcompanion)
library(car)

# read the model outcome data
file_path = "D:/PhD Program/Final Research/Dissertation/carnatic_raga_MER_dataset.csv"

mer_data_raw = read.csv(file_path, header = T, stringsAsFactors = F)
mer_data = mer_data_raw[mer_data_raw$Quadrant=='Q1',]

# Step 1: Summary & Basic plots
summary(mer_data)
mer_data$perc <- mer_data$perc + 1.0 # to avoid 0 values
mer_data$Emotion <- factor(mer_data$Emotion)
mer_data$Quadrant <- factor(mer_data$Quadrant)
mer_data$raga <- factor(mer_data$raga)

Summarize(perc~raga, data=mer_data, digits = 3)

boxplot(perc ~ raga,
        data = mer_data)

groupwiseMean(perc ~ raga, 
              data=mer_data, 
              conf = 0.95, 
              digits = 3, 
              traditional = F,
              percentile = T)

y = mer_data$perc
y_log = log(mer_data$perc+10)

# normality tests
# Shapiro test

shapiro.test(y)
shapiro.test(y_log)

# Agostino test - skewness 

agostino.test(y)
agostino.test(y_log)

# anscombe test - kurtosis

anscombe.test(y)
anscombe.test(y_log)

# equilatiy of variance

bartlett.test(y,mer_data$raga)
bartlett.test(y_log,mer_data$raga)

tapply(y_log,mer_data$raga, var)

# Krusal test
kruskal.test(y~raga, data = mer_data)
kruskal.test(y_log~raga, data = mer_data)

# Step 2: One-way ANOVA
# raga - linear model
aov_model_raga <- aov(perc~raga-1, data = mer_data)
summary(aov_model_raga)
qqnorm(aov_model_raga$residuals)

Anova(aov_model_raga, type='II')
marginal = lsmeans(aov_model_raga,
                   ~ raga)

pairs_df = pairs(marginal, adjust = 'tukey')
#write.csv(pairs_df,"test_csv.csv")


CLD = cld(marginal,
          alpha   = 0.05,
          Letters = letters,         ###  Use lowercase letters for .group
          adjust  = "tukey") 
CLD

TukeyHSD(aov_model_raga)
Anova(aov_model_raga, type='2')