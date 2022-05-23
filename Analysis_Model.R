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
#mer_data = mer_data_raw[mer_data_raw$Quadrant=='Q1',]
mer_data = mer_data_raw

# Step 1: Summary & Basic plots
summary(mer_data)
mer_data$perc <- mer_data$perc + 1.0 # to avoid percentages with '0.00' as value
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
aov_model_raga <- aov(perc~raga, data = mer_data)
summary(aov_model_raga)
qqnorm(aov_model_raga$residuals)

Anova(aov_model_raga, type='II')
marginal = lsmeans(aov_model_raga,
                   ~ raga)

pairs_df = pairs(marginal, adjust = 'tukey')
#write.csv(pairs_df,"test_csv.csv")
pairs_df


CLD = cld(marginal,
          alpha   = 0.05,
          Letters = letters,         ###  Use lowercase letters for .group
          adjust  = "tukey") 
CLD

TukeyHSD(aov_model_raga)
Anova(aov_model_raga, type='2')


# Quadrant - linear model
aov_model_quadrant <- aov(perc~raga, data = mer_data)
summary(aov_model_quadrant)
qqnorm(aov_model_quadrant$residuals)

Anova(aov_model_quadrant, type='II')
marginal = lsmeans(aov_model_quadrant,
                   ~ raga)

pairs_df = pairs(marginal, adjust = 'tukey')
#write.csv(pairs_df,"test_csv.csv")
pairs_df


CLD = cld(marginal,
          alpha   = 0.05,
          Letters = letters,         ###  Use lowercase letters for .group
          adjust  = "tukey") 
CLD

TukeyHSD(aov_model_quadrant)
Anova(aov_model_quadrant, type='2')

# 2-Way ANOVA

aov_model <- aov(perc~raga+Quadrant, data = mer_data)
summary(aov_model)
qqnorm(aov_model$residuals)

Anova(aov_model, type='II')
marginal = lsmeans(aov_model,
                   ~ raga)

pairs_df = pairs(marginal, adjust = 'tukey')
#write.csv(pairs_df,"test_csv.csv")
pairs_df


CLD = cld(marginal,
          alpha   = 0.05,
          Letters = letters,         ###  Use lowercase letters for .group
          adjust  = "tukey") 
CLD

TukeyHSD(aov_model)
Anova(aov_model, type='2')

# 2-Way ANOVA no intercept

# Fixed effects
aov_model_ni <- aov(perc~raga+Quadrant, data = mer_data)

# Interaction effect
aov_model_ni <- aov(perc~raga+Quadrant+raga:Quadrant, data = mer_data)

#OR
aov_model_ni <- aov(perc~raga*Quadrant-1, data = mer_data)
summary(aov_model_ni)
qqnorm(aov_model_ni$residuals)

Anova(aov_model_ni, type='II')
marginal = lsmeans(aov_model_ni,
                   ~ raga)

pairs_df = pairs(marginal, adjust = 'tukey')
#write.csv(pairs_df,"test_csv.csv")
pairs_df


CLD = cld(marginal,
          alpha   = 0.05,
          Letters = letters,         ###  Use lowercase letters for .group
          adjust  = "tukey") 
CLD

tt <- TukeyHSD(aov_model_ni)
Anova(aov_model_ni, type='2')

pairwise.t.test(mer_data$perc,mer_data$Emotion,p.adj="bonferroni")
pairwise.t.test(mer_data$perc,mer_data$raga,p.adj="bonferroni")


TukeyHSD(x = aov_model_ni,
         ordered = FALSE,
         which = "Quadrant",
         conf.level = 0.95)

summary(aov_model_ni)

aov_model_ni <- aov(perc~Quadrant+(1/raga), data = mer_data)


aov_model_ni <- aov(perc~raga+Quadrant-1, data = mer_data)

summary(aov_model_ni)
qqnorm(aov_model_ni$residuals)

Anova(aov_model_ni, type='II')

TukeyHSD(x = aov_model_ni,
         ordered = FALSE,
         which = "Quadrant",
         conf.level = 0.95)

interaction.plot(mer_data$raga,mer_data$Quadrant,mer_data$perc)
mer_data_q1 = mer_data_raw[mer_data_raw$raga %in% c('Mayamalavagaula','Bhairavi'),]

pairwise.t.test(mer_data_q1$perc, mer_data_q1$raga, adjust.method="BH")
