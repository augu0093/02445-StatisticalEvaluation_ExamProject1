#proj1
setwd("C:/Users/Karlu/Desktop/Proj1")

# THE DATA NEEDS TO BE IMPORTED MANUALLY INTO THE VARIABLE "armdata"

#plot of every trajectory for exp9 person 1 
l = c(1,2,1,3,2,3)
names = c("x","y","x","z", "y", "z")
titles = c("The trajectory from above", "The trajectory from the side","The trajectory from behind")
par(mfrow=c(1,3),pty="s")
for (k in c(1:3)){
  for (i in 1:3){
    d = matrix(0,ncol = 3, nrow=100)
    for (j in 1:10){
      d = d+armdata[[9]][[k]][[j]]*(1/10)
    }
    x <- d[,l[i*2-1]][seq(100)[c(F,F,T)]]
    y <- d[,l[i*2]][seq(100)[c(F,F,T)]]
    
    plot(x, y, title = titles[i], xlab = names[i*2-1],ylab = names[i*2], main = titles[i],asp = 1)
  }
}



armdata[[9]][[1]][[j]]

#Making a 100obs X 300 feature data matrix
#armdata[[9]] is the experiment
pre_data <- armdata[[9]]
x <- c()

for (person in 1:10){
  for (repetition in 1:10){
    D <-  pre_data[[person]][[repetition]]
    
    r <- c()
    for (feature in 1:3){
      r <- append(r, D[,feature])
    }
    x <- rbind(x, r)

  }
}

#finnishing up x data
x <- x[1:100,1:300]
x <- as.data.frame.matrix(x)
rownames(x) <- c(1:100)
write.table(x,file<-"x9.csv",row.names = FALSE,col.names = FALSE)

#y data
y <- c()
for (i in 1:10){y <- append(y,rep(i,10))}
write.table(y,file<-"y9.csv",row.names = FALSE,col.names = FALSE)


# ANOVA
#getting the data
person_mean_exp <- matrix(0, nrow = 10, ncol = 16)
for (exp in 1:16){
  for (per in 1:10){
    
    temp_mean <- c()
    for (rep in 1:10){
      results <- na.omit(as.data.frame(armdata[[exp]][[per]][[rep]][]))
      temp_mean <- append (temp_mean,  mean(sapply(results, mean, na.rm = TRUE)))
    }
    if (is.na(mean(temp_mean))){print(c(exp,per,rep))}
    
    person_mean_exp[per,exp] = mean(temp_mean)
  }
}

#getting factors for data frame
exp = c()
for (i in 1:10){exp = append(exp, c(1:16))}

per <- c()
for (i in 1:10){per = append(per,rep(i,16))}

person_mean_exp_vector <- as.vector(t(person_mean_exp))

D <- data.frame(
  y=person_mean_exp_vector,
  experiment=factor(exp),
  person=factor(per))

#box plot of person and experiment
par(mfrow=c(1,2))
plot(D$experiment, D$y, xlab="Experiment", ylab=" ")
plot(D$person, D$y, xlab="Subject", ylab=" ")

# ANOVA
fit <- (lm(y ~ experiment + person, data=D))
anova(fit)

#check for normality
par(mfrow=c(1,3))
qqnorm(fit$residuals,main= "")
qqline(fit$residuals,main= "")
#and same variance:
plot(D$experiment, fit$residuals, xlab="Experiment", ylab="Residuals")
plot(D$person, fit$residuals, xlab="Subject", ylab="Residuals")

plot(fit)

# Matrix of paried t-test for all esperiments. 
p_values <- matrix(0, 16, 16)
for (i in 1:16){
  for (j in 1:16){
   p_values[i,j] <- t.test(person_mean_exp[,i],person_mean_exp[,j], paired = TRUE)$p.value
    
  }
}
#Matrix from hell 16 x 16
round(p_values, digits =2) 
  
# List of CI from paired t-test between exp 9 and all other.
CI <- c()
for (i in 1:16){
  CI <- append(CI, t.test(person_mean_exp[,i],person_mean_exp[,9], paired = TRUE)$conf[1:2])
}
write.table(CI,file<-"CI.csv",row.names = FALSE,col.names = FALSE)

#exp = c('exp 1','exp 2','exp 3','exp 4','exp 5','exp 6','exp 7','exp 8','exp 9','exp10','exp11','exp12','exp13','exp14','exp15','exp16')
#p_val_dataframe = as.data.frame((round(p_values, digits =2)))
#colnames(p_val_dataframe) <- exp
#rownames(p_val_dataframe) <- exp
#write.table(p_val_dataframe,file<-"exp_CI.csv",row.names = FALSE,col.names = FALSE)
#p_val_dataframe
#write.table(as.data.frame(person_mean_exp),file<-"exp_data.csv",row.names = FALSE,col.names = FALSE)
