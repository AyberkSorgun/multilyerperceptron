setwd("~/Desktop/hw03")
data_set <- read.csv("hw03_data_set_images.csv",header = FALSE)
Y_truth_label <- read.csv("hw03_data_set_labels.csv",header = FALSE)
Y_truth_label <- Y_truth_label$V1

safelog <- function(x) {
  return (log(x + 1e-100))
}

sigmoid <- function(a) {
  return (1 / (1 + exp(-a)))
}

softmax <- function(Z, v) {
  scores <- exp(Z %*% t(v))
  scores <- scores / matrix(rowSums(scores), nrow(scores), ncol(scores), byrow = FALSE)
  return (scores)
}

train_a <- data_set[1:25,]
train_b <- data_set[40:64,]
train_c <- data_set[79:103,]
train_d <- data_set[118:142,]
train_e <- data_set[157:181,]
test_a <- data_set[26:39,]
test_b <- data_set[65:78,]
test_c <- data_set[104:117,]
test_d <- data_set[143:156,]
test_e <- data_set[182:195,]

training_set <- rbind(train_a,train_b,train_c,train_d,train_e)
test_set <- rbind(test_a,test_b,test_c,test_d,test_e)

K <- 5L
D <- ncol(training_set)
N <- length(Y_truth_label)

Y_truth <- matrix(0, N, K)
Y_truth[cbind(1:N, Y_truth_label)] <- 1

Y_truth_test <- cbind(Y_truth[26:39,],Y_truth[65:78,],Y_truth[104:117,],Y_truth[143:156,],Y_truth[182:195,])
dim(Y_truth_test) <- c(70,5)
Y_truth_train <- cbind(Y_truth[1:25,],Y_truth[40:64,],Y_truth[79:103,],Y_truth[118:142,],Y_truth[157:181,])
dim(Y_truth_train) <- c(125,5)

training_set <- data.matrix(training_set)
test_set <- data.matrix(test_set)

eta <- 0.005
epsilon <- 1e-3
H <- 20
max_iteration <- 200
set.seed(521)

W <- matrix(runif((D + 1) * H, min = -0.01, max = 0.01), D + 1, H)
v <- matrix(runif(100, min = -0.01, max = 0.01),5,20)

Z <- sigmoid(cbind(1, training_set) %*% W)
y_predicted <- softmax(Z,v) 
objective_values <- -sum(Y_truth_train * safelog(y_predicted))

iteration <- 1
while (1) {
  
  for (i in sample(125)) {
    # calculate hidden nodes
    Z[i,] <- sigmoid(c(1, training_set[i,]) %*% W)
    # calculate output node
    y_predicted[i,] <- softmax(Z[i,],v)  
    
    v <- v + eta * (Y_truth_train[i,] - y_predicted[i,]) * c(1, Z[i,])
    for (h in 1:20) {
      W[,h] <- W[,h] + eta * sum((Y_truth_train[i,] - y_predicted[i,]) * v[,h]) * Z[i, h] * (1 - Z[i, h]) * c(1, training_set[i,])
    }
  }
  
  Z <- sigmoid(cbind(1, training_set) %*% W)
  y_predicted <- softmax(Z,v) 
  objective_values <- c(objective_values, -sum(Y_truth_train * safelog(y_predicted)))
  
  if (abs(objective_values[iteration + 1] - objective_values[iteration]) < epsilon || iteration >= max_iteration) {
    break
  }
  
  iteration <- iteration + 1
}
print(W)
print(v)

# plot objective function during iterations
plot(1:(iteration + 1), objective_values,
     type = "l", lwd = 2, las = 1,
     xlab = "Iteration", ylab = "Error")

conf_truth_train <- rowSums(sapply(X=1:5, FUN=function(c) {Y_truth_train[,c]*c}))
y_predicted <- apply(y_predicted, 1, which.max)
confusion_matrix_train <- table(y_predicted, conf_truth_train)
print(confusion_matrix_train)

Z <- sigmoid(cbind(1, test_set) %*% W)
Y_predicted_test <- softmax(Z,v)

conf_truth_test <- rowSums(sapply(X=1:5, FUN=function(c) {Y_truth_test[,c]*c}))
Y_predicted_test <- apply(Y_predicted_test, 1, which.max)
confusion_matrix_test <- table(Y_predicted_test, conf_truth_test)
print(confusion_matrix_test)
