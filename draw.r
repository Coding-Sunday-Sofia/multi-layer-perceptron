r01 <- read.csv(file="~/Desktop/r01.csv", header=TRUE, sep=",")
m01 <- read.csv(file="~/Desktop/m01.csv", header=TRUE, sep=",")

boxplot(t(r01), main="MLP with Regular Training", col="red", xlab="Training Cycles", ylab="Total MLP Error")

boxplot(t(m01), main="MLP with Manipulated Training", col="green", xlab="Training Cycles", ylab="Total MLP Error")

plot(rowSums(m01),type="l",col="green",main="ANN Training Convergence", xlab="Training Cycles", ylab="Average MLP Error")
lines(rowSums(r01),col="red")
legend("topright",legend=c("MLP with Regular Training", "MLP with Manipulated Training"),col=c("red", "green"),lty=1)
