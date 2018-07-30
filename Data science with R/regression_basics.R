library(ggplot2)
library(dplyr)
library(moderndive)

# Load data of professor rankings and different attributes
load(url("http://www.openintro.org/stat/data/evals.RData"))

# Choose a subset of the data for analysis (using the dplyr library)
evals <- evals %>%
  select(score, bty_avg)

# score: Numerical variable of the average teaching 
# score based on students evaluations between 1 and 5. 
# This is the outcome variable y of interest.
# 
# bty_avg: Numerical variable of average “beauty” rating 
# based on a panel of 6 students’ scores between 1 and 10. 
# This is the numerical explanatory variable x of interest.

# Compute summary statistics
evals %>%
  summary()


# Evaluate the correlation between beauty and score
cor(evals$score, evals$bty_avg)


# Visualize the data using a scatterplot
ggplot(evals, aes(x = evals$bty_avg, y = score )) +
  geom_point() + 
  labs(x = "Beauty Score", y = "Teaching Score", title = "Relation between beauty and teaching score")

# To avoid overplotting we use geom_jitter and alpha
ggplot(evals, aes(x = evals$bty_avg, y = score )) +
  geom_jitter(alpha = 0.5) + 
  labs(x = "Beauty Score", y = "Teaching Score", title = "Relation between beauty and teaching score")


# Add regression line to plot with standard error bands 
ggplot(evals, aes(x = evals$bty_avg, y = score )) +
  geom_jitter(alpha = 0.5) + 
  labs(x = "Beauty Score", y = "Teaching Score", title = "Relation between beauty and teaching score") + 
  geom_smooth(method = "lm")


# Create a regression table using the linear model
lm(score ~ bty_avg, data = evals) %>%
  get_regression_table(digits= 2)


# Calculate the residual error and predicted values (score hat) with regression points
lm(score ~ bty_avg, data = evals) %>%
  get_regression_points()



# Create a similar exploration using age as the studied parameter
load(url("http://www.openintro.org/stat/data/evals.RData"))

# Choose a subset of the data for analysis (using the dplyr library)
evals <- evals %>%
  select(score, age)

# get summary statistics
evals %>%
  summary()

# Visualize data using scatter plot
ggplot(data = evals, aes(x = age, y = score)) + 
  geom_jitter()


# Create a regression line of the data
ggplot(data = evals, aes(x = age, y = score)) +
  geom_jitter() + 
  geom_smooth(method  = 'lm')


# Create a regression table
lm(score ~age, evals) %>%
  get_regression_table()

# Create a regression points
regr_points = lm(score ~age, evals) %>%
  get_regression_points()

# Histogram of residual
ggplot(regr_points, aes(x = residual)) + 
  geom_histogram(bins = 40)
# Histogram is skewed to the right



