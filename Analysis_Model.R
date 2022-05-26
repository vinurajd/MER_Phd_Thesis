# Objective: Analyze the emotional significance of raga mayamalavagoulva w.r.t
# others raga
# Input dataset: Emotion class prediciton from the Adaboost model for 5 songs within
# each raga
# Total obs: 192

setwd("D:/PhD Program/Final Research/Dissertation")
library(HH)
library(lsmeans)
library(multcomp)
library(multcompView)
library(psych)
library(moments)
library(FSA)
library(rcompanion)
library(car)
library(coin)
library(FSA)

# read the model outcome data
file_path = "D:/PhD Program/Final Research/Dissertation/carnatic_raga_MER_dataset.csv"

mer_data_raw = read.csv(file_path, header = T, stringsAsFactors = F)
#mer_data = mer_data_raw[mer_data_raw$Quadrant=='Q1',]

# Step 1: Summary & Basic plots
summary(mer_data_raw)
mer_data_raw$perc <- mer_data_raw$perc + 1.0 # to avoid percentages with '0.00' as value
mer_data_raw$Emotion <- factor(mer_data_raw$Emotion)
mer_data_raw$Quadrant <- factor(mer_data_raw$Quadrant)
mer_data_raw$raga <- factor(mer_data_raw$raga)
Summarize(perc~raga, data=mer_data, digits = 3)


mer_data = mer_data_raw[mer_data_raw$Quadrant=='Q1',]

#dev.new(width=5, height=4, unit="in")

jpeg('Q1-emotion-perc.jpg', width=800, height=400)
boxplot(perc ~ raga,
        data = mer_data,main="Box plot - Raga significance Q1", 
        xlab="raga", ylab="Emotion %",las = 1,ylim=c(0,100),
        cex.names = 1, cex.axis=0.8,cex.main=1)

dev.off()

mer_data = mer_data_raw[mer_data_raw$Quadrant=='Q2',]

jpeg('Q2-emotion-perc.jpg', width=800, height=400)
boxplot(perc ~ raga,
        data = mer_data,main="Box plot - Raga significance Q2", 
        xlab="raga", ylab="Emotion %",las = 1,ylim=c(0,100),
        cex.names = 1, cex.axis=0.8,cex.main=1)

dev.off()

mer_data = mer_data_raw[mer_data_raw$Quadrant=='Q3',]

jpeg('Q3-emotion-perc.jpg', width=800, height=400)
boxplot(perc ~ raga,
        data = mer_data,main="Box plot - Raga significance Q3", 
        xlab="raga", ylab="Emotion %",las = 1,ylim=c(0,100),
        cex.names = 1, cex.axis=0.8,cex.main=1)
dev.off()

mer_data = mer_data_raw[mer_data_raw$Quadrant=='Q4',]

jpeg('Q4-emotion-perc.jpg', width=800, height=400)
boxplot(perc ~ raga,
        data = mer_data,main="Box plot - Raga significance Q4", 
        xlab="raga", ylab="Emotion %",las = 1,ylim=c(0,100),
        cex.names = 1, cex.axis=0.8,cex.main=1)

dev.off()

mer_data = mer_data_raw

groupwiseMean(perc ~ raga, 
              data=mer_data, 
              conf = 0.95, 
              digits = 3, 
              traditional = F,
              percentile = T)


y = mer_data$perc
y_log = log(mer_data$perc+10)

plot(density(y))
plot(density(y_log))

qqnorm(y_log)
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

# Krusal test - overall
mer_data = mer_data_raw
kt_quadrant <- kruskal.test(perc~Quadrant, data = mer_data)
kt_quadrant
# mer_data = mer_data_raw[mer_data_raw$Quadrant=='Q3',]
# median_test(perc~raga, data = mer_data)
median_test(perc~Quadrant, data = mer_data)

#options(scipen=999)

# perform dunnet test as post hoc for Kruskal Wallis
dunnTest_res <- dunnTest(perc~Quadrant, data = mer_data)
format(dunnTest_res$res,scientific=F)
dunnTest_res <- dunnTest(perc~Quadrant, data = mer_data, 
                         method= 'bonferroni')
format(dunnTest_res$res,scientific=F)
# Pairwise test for comparison
pairwise.wilcox.test(mer_data$perc,mer_data$Quadrant,exact=F)

#
# Krusal test - raga
mer_data = mer_data_raw[mer_data_raw$raga%in% c('Mayamalavagaula',
                                                'Kalyani',
                                                'Mohanam',
                                                'Bhairavi',
                                                'Todi'),]


mer_data = mer_data[mer_data$Quadrant %in% c('Q1','Q2'),]
mer_data$Quadrant <- factor(mer_data$Quadrant,levels=unique(mer_data$Quadrant))
kt_quadrant <- kruskal.test(perc~Quadrant, data = mer_data)
kt_quadrant
rownames(mer_data) = seq(length=nrow(mer_data))
# median_test(perc~raga, data = mer_data)
median_test(perc~Quadrant, data = mer_data)

# perform dunnet test as post hoc for Kruskal Wallis
dunnTest(perc~Quadrant, data = mer_data,two.sided = T)
dunn.test::dunn.test(mer_data$perc,mer_data$Quadrant)

interaction.plot(mer_data$Quadrant,mer_data$raga,mer_data$perc, main="", 
                 ylab = "",
                 xlab = "")
ci.plot(lm(perc~Quadrant, data = mer_data))
