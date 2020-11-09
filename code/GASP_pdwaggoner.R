# Philip Waggoner (https://pdwaggoner.github.io/), University of Chicago
# Nov 6, 2020 - GASP Talk

# Building Neural Networks in R with External Machine Learning Engines

# Agenda: 
#
#   1.    Overview external ML engines and APIs
#   2.    Overview neural network architecture (ANN, autoencoder, deep)
#   3.    To the code! Application with ANES and Congress data


###
### Notes
###


# Preliminaries: 
#   1. Visit the repo for this talk for the two data sets we will use for the examples: https://github.com/pdwaggoner/gasp2020
#   2. Make sure you have the latest version of the JDK (https://www.oracle.com/java/technologies/javase/javase-jdk8-downloads.html) installed on your machine for H2O to work. If not, no worries, just follow along.


# External ML Engines

# ML engines are sets of tools that include repositories of source code for many model architectures, allowing computational power and ability to extend far beyond a single package or basic, local, statistical programming. 

# Via some entry point, e.g., an API, the engines offer access to this repository of models, allowing for development and implementation of virtually any machine learning model, locally from anywhere in the world. 

# 1. Tensorflow (Keras)
# 2. H2O

# Neural Network Architecture

# A neural network is interested in building a model that emulates the human brain's approach to information processing: 

#   1. receive raw information 
#   2. Process that raw information
#   3. Iteratively revisit and consider the data to look for trends, and thus learn
#   4. Offer a conclusion based on how the information was processed

## (Major) Hyperparameters

#   1. Hidden layers
#   2. Neurons in the hidden layers
#   3. Activation function
#   4. Learning rate
#   5. Epochs
#   6. Stopping

# --> EXAMPLE 1 <-- #

# Autoencoders

# Two major steps, built on a neural network architecture: encode and decode

# These seek to also learn from data, but in an unsupervised way. Take raw unlabeled inputs, encode them (force information loss) based on natural patterns in the input space, and thus project onto a lower dimension. Then, reconstruct (decode) the original input space, but this time based only on the encoding from the first step. 

# The goal: minimize reconstruction error. And also, use deep features via "feature extraction".

# --> EXAMPLE 2 <-- #



###
### Code for applications
###


#
# Example 1: Artificial neural network via Keras/Tensorflow
#

# load some libraries
library(keras)
library(tensorflow)
library(tidyverse)
library(patchwork)
library(skimr)
library(naniar)
library(recipes)
library(here)
library(tictoc)

#install_keras() # might need this if you're new to keras/TF

# load the data
cong <- read_csv(here("data", "congress_109_110.csv"))

data <- cong %>% 
  select(elected, votepct, dwnom1, seniority, les) %>% 
  glimpse()

skim(data) 

# a few observations seem to be missing; what's the pattern?
data %>%
  gg_miss_upset()

# impute
congress_recipe <- recipe(les ~ elected + votepct + dwnom1 + seniority, 
                          data = data) %>%
  step_meanimpute(dwnom1) %>% # impute using feature mean
  step_knnimpute(all_predictors()) # use kNN for ALL others: elected, seniority, votepct

congress_imputed <- prep(congress_recipe) %>% 
  juice()

# how does it look?
{
  summary(data$dwnom1)
summary(congress_imputed$dwnom1) # pretty good!

summary(data$elected)
summary(congress_imputed$elected) 

summary(data$seniority)
summary(congress_imputed$seniority) 

summary(data$votepct)
summary(congress_imputed$votepct)
}

# now construction begins
set.seed(1234)

## first, create train test split (80/20)
congress_imputed_split <- sample(1:nrow(congress_imputed), 0.8 * nrow(congress_imputed))
train <- congress_imputed[congress_imputed_split, ]
test <- congress_imputed[-congress_imputed_split, ]

# next, normalize (first train, then test BY train)
train <- train %>% 
  scale() 

# now test set
train_mean <- attr(train, "scaled:center") 
train_sd <- attr(train, "scaled:scale")
test <- scale(test, center = train_mean, scale = train_sd)

skim(train)
skim(test)

# separate response from input features to fit the model
train_labels <- train[ , "les"]
train_data <- train[ , 1:4]

test_labels <- test[ , "les"]
test_data <- test[ , 1:4]

# construct the model
model <- keras_model_sequential() 

model %>% 
  layer_dense(units = 4,
              input_shape = dim(train_data)[2]) %>%
  layer_dense(units = 8,
              activation = "relu") %>%
  layer_dense(units = 1) %>% 
  compile(
    loss = "mse",
    optimizer = optimizer_rmsprop(),
    metrics = "mae")

epochs <- 500

# fit the model (be sure to store the results for visualization)
{
  tic()
out <- model %>% fit(
  train_data,
  train_labels,
  epochs = epochs,
  validation_split = 0.2,
  verbose = FALSE
); out 
toc()} 

# visualize the training process
plot(out) +
  theme_minimal() +
  labs(title = "Predicting LES in Congress with Neural Network",
       x = "Epoch",
       y = "Metric") 

# predictions
test_predictions <- model %>% 
  predict(test_data)

# compare with LES from training set
preds <- qplot(test_predictions, xlab = "Predicted LES")
peak_preds <- ggplot_build(preds)[[1]][[1]]
x1 <- mean(unlist(peak_preds[which.max(peak_preds$ymax), c("xmin", "xmax")]))
preds <- preds + 
  geom_vline(xintercept=x1, col="red", lty=2, lwd=1) + 
  geom_vline(xintercept=0, col="blue", lty=1, lwd=1) + 
  theme_minimal()

trained <- qplot(train_labels, xlab = "Training LES")
peak_tr <- ggplot_build(trained)[[1]][[1]]
x2 <- mean(unlist(peak_tr[which.max(peak_tr$ymax), c("xmin", "xmax")]))
trained <- trained + 
  geom_vline(xintercept=x2, col="red", lty=2, lwd=1) + 
  geom_vline(xintercept=0, col="blue", lty=1, lwd=1) + 
  theme_minimal()

preds + 
  trained + plot_annotation(title = "Comparing Test and Train LES")



#
# Example 2: Autoencoders via H2O
#

# load a few (new) libraries
library(h2o)
library(doParallel)

# read in data
NESdta <- read_csv(here("data", "anes_2016.csv"))

# Munging and engineering
NES_new <- NESdta %>%
  dplyr::select(pid3, starts_with("ft")) %>%
  mutate(democrat = ifelse(pid3 == 1, 1, 0),
         fttrump = replace(fttrump, fttrump > 100, NA),
         ftobama = replace(ftobama, ftobama > 100, NA),
         ftblack = replace(ftblack, ftblack > 100, NA),
         ftwhite = replace(ftwhite, ftwhite > 100, NA),
         fthisp = replace(fthisp, fthisp > 100, NA),
         ftgay = replace(ftgay, ftgay > 100, NA),
         ftjeb = replace(ftjeb, ftjeb > 100, NA),
         ftcarson = replace(ftcarson, ftcarson > 100, NA),
         fthrc = replace(fthrc, fthrc > 100, NA),
         ftrubio = replace(ftrubio, ftrubio > 100, NA),
         ftcruz = replace(ftcruz, ftcruz > 100, NA),
         ftsanders = replace(ftsanders, ftsanders > 100, NA),
         ftfiorina = replace(ftfiorina, ftfiorina > 100, NA),
         ftpolice = replace(ftpolice, ftpolice > 100, NA),
         ftfem = replace(ftfem, ftfem > 100, NA),
         fttrans = replace(fttrans, fttrans > 100, NA),
         ftmuslim = replace(ftmuslim, ftmuslim > 100, NA),
         ftsci = replace(ftsci, ftsci > 100, NA))

# drop pid3 for fitting in a bit
NES_new <- NES_new %>% 
  dplyr::select(-pid3) %>% 
  glimpse()

# skim for NAs
NES_new %>%   
  skim() # almost good, but some missing

# search and impute
NES_new %>%
  gg_miss_upset()

# impute
anes_recipe <- recipe(democrat ~ ., 
                      data = NES_new) %>%
  step_knnimpute(all_predictors()) 

anes_imputed <- prep(anes_recipe) %>% 
  juice()

# now check (indiv)
{
  summary(NES_new$ftcarson)
summary(anes_imputed$ftcarson) 

summary(NES_new$ftcruz)
summary(anes_imputed$ftcruz)

summary(NES_new$ftsanders)
summary(anes_imputed$ftsanders) 

summary(NES_new$ftrubio)
summary(anes_imputed$ftrubio) 

summary(NES_new$ftfiorina)
summary(anes_imputed$ftfiorina)}

# one final check to make sure:
anes_imputed %>%   
  skim()

theme_set(theme_minimal())

# Let's get to know the data, and specifically the response feature, "democrat"
ggplot(anes_imputed, aes(ftobama, fill = factor(democrat))) + # divisive
  geom_boxplot(alpha = 0.5) + 
  labs(fill = "Democrat")

ggplot(anes_imputed, aes(fttrump, fill = factor(democrat))) + # odd
  geom_boxplot(alpha = 0.5) + 
  labs(fill = "Democrat")

ggplot(anes_imputed, aes(ftjeb, fill = factor(democrat))) + # midling and even
  geom_boxplot(alpha = 0.5) + 
  labs(fill = "Democrat")

ggplot(anes_imputed, aes(fthrc, fill = factor(democrat))) + # much more divisive
  geom_boxplot(alpha = 0.5) + 
  labs(fill = "Democrat")

ggplot(anes_imputed, aes(ftsanders, fill = factor(democrat))) + # slightly less divisive than O or HRC
  geom_boxplot(alpha = 0.5) + 
  labs(fill = "Democrat")

# can also view the histograms for both parties 
## for obama
ggplot(anes_imputed, aes(x = ftobama)) +
  geom_histogram(color = "dodger blue", 
                 fill = "dodger blue",
                 alpha = 0.5,
                 bins = 30) +
  facet_wrap( ~ democrat)

## for trump
ggplot(anes_imputed, aes(x = fttrump)) +
  geom_histogram(color = "dodger blue", 
                 fill = "dodger blue", 
                 alpha = 0.5, 
                 bins = 30) +
  facet_wrap( ~ democrat)

## and so on...

# finally, let's take a look at the distribution of party 
anes_imputed %>%
  ggplot(aes(x = democrat)) +
  geom_bar(color = "dark green", 
           fill = "dark green", 
           alpha = 0.7) +
  labs(title = "Party Distribution",
       x = "Democrat",
       y = "Count")

set.seed(1234)

# First, convert party to factor for modeling 
anes_imputed$democrat <- factor(anes_imputed$democrat)

# initializing the H2O cluster/session
my_h2o <- h2o.init()

# Detecting the available number of cores; set up cluster for parallel session
cores <- detectCores() - 1 # leave one for the rest of the computer to process normally
cluster <- makeCluster(cores, setup_timeout = 0.5)
registerDoParallel(cluster) # takes about 5 minutes to set up the first time

# Create an H2O dataframe
anes_h2o <- anes_imputed %>% 
  as.h2o()

# Create train (0.60), validation (0.20), and test (0.20) sets 
split_frame <- h2o.splitFrame(anes_h2o, 
                              ratios = c(0.6, 0.2), 
                              seed = 1234)   

# a quick look
split_frame %>% 
  glimpse()

train <- split_frame[[1]]
validation <- split_frame[[2]]
test <- split_frame[[3]]

# Store response and predictors separately (per h2o syntax)
response <- "democrat"

predictors <- setdiff(colnames(train), response)

# Construct vanilla autoencoder with tanh activation and a single hidden layer with 4 nodes
autoencoder <- h2o.deeplearning(x = predictors, 
                                training_frame = train,
                                autoencoder = TRUE,
                                seed = 1234,
                                hidden = c(4), 
                                epochs = 500,
                                activation = "Tanh",
                                validation_frame = test)

# Make "predictions" on test set
preds <- h2o.predict(autoencoder, test)

# Let's now extract the codings/features from fit AE
codings <- h2o.deepfeatures(autoencoder, train, layer = 1) %>% # "layer" is referring to the number of hidden layers (1 in our case)
  as.data.frame() %>%
  mutate(democrat = as.vector(train[ , 19]))

## Note: these are read as, e.g., DF.L1.C1 - "data frame, layer number, column number"

# Numeric inspection of the "codes" (or, "scores")
codings %>% 
  head(10)

# Visual inspection (separate by party?)
ggplot(codings, aes(x = DF.L1.C1, y = DF.L1.C2, 
                    color = factor(democrat))) +
  geom_point(alpha = 0.5) + 
  labs(title = "First Two Deep Features",
       subtitle = "Shallow Autoencoder with 1 Hidden Layer (4 nodes)",
       color = "Democrat")

# Now, check the next two features (3 and 4)
ggplot(codings, aes(x = DF.L1.C3, y = DF.L1.C4, 
                    color = factor(democrat))) +
  geom_point(alpha = 0.5) +
  labs(title = "Last Two Deep Features",
       subtitle = "Shallow Autoencoder with 1 Hidden Layer (4 nodes)",
       color = "Democrat")


## First, let's build a new model to predict party affiliation as a function of the deep features. With our deep features from the AE, train a deep neural net (2 HL, total of 10 nodes per the rule)

codings_val <- h2o.deepfeatures(autoencoder, validation, layer = 1) %>%
  as.data.frame() %>%
  mutate(democrat = as.factor(as.vector(validation[ , 19]))) %>%
  as.h2o()

# Store using new codings_val object
deep_features <- setdiff(colnames(codings_val), response)

deep_net <- h2o.deeplearning(y = response,
                             x = deep_features,
                             training_frame = codings_val,
                             reproducible = TRUE, 
                             ignore_const_cols = FALSE,
                             seed = 1234,
                             hidden = c(5, 5), 
                             epochs = 500,
                             activation = "Tanh")

## Make predictions & classify
test_3 <- h2o.deepfeatures(autoencoder, test, layer = 1)
test_pred <- h2o.predict(deep_net, test_3, type = "response")%>%
  as.data.frame() %>%
  mutate(Truth = as.vector(test[, 19]))

# Visualize predictions
test_pred %>% 
  head(25)

# Summarize predictions
print(h2o.predict(deep_net, test_3) %>%
        as.data.frame() %>%
        mutate(truth = as.vector(test[, 19])) %>%
        group_by(truth, predict) %>%
        summarise(n = n()) %>%
        mutate(freq = n / sum(n)))


## Feature Importance
fimp <- as.data.frame(h2o.varimp(deep_net)) %>% 
  arrange(desc(relative_importance))

fimp %>% 
  ggplot(aes(x = relative_importance, 
             y = reorder(variable, -relative_importance))) +
  geom_point(color = "dark red", 
             fill = "dark red", 
             alpha = 0.5) +
  labs(title = "Feature Importance",
       subtitle = "Deep Neural Network (2 hidden layers with 10 neurons)",
       x = "Relative Importance",
       y = "Feature") 


# Shut down h2o and stop the parallel cluster when finished with the session
h2o.shutdown()
stopCluster(cluster)


#
# Example 3: Autoencoder via Keras/Tensorflow
#

# Here is a great tutorial on how to implement an autoencoder in Keras/TF: http://gradientdescending.com/pca-vs-autoencoders-for-dimensionality-reduction/ 

# The logic is very similar to everything we have covered, so it should be manageable for you to adapt this code to apply to our political data. Good luck!
