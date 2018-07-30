library(nycflights13)
library(knitr)
library(dplyr)
library(ggplot2)

all_alaska_flights <- flights %>%
  filter(carrier == "AS")  


# Create scatterplot of the flights data
ggplot(all_alaska_flights, aes(x = dep_delay, y = arr_delay)) +
  geom_point()


# Avoiding overplotting ~ cluttering plot with too many points plotted on top of each other. Using alpha areas with high degree
# Of overlap will appear darker and vice versa
ggplot(all_alaska_flights, aes(x = dep_delay, y = arr_delay)) +
  geom_point(alpha = 0.20)


# Same with all flights
ggplot(flights, aes(x = dep_delay, y = arr_delay)) +
  geom_point(alpha = 0.20)


# Another way of handling overplotting is by using jittering, which inserts a bit randomness to the plot 
# And moves the points on the plot by a small degree
  # Parameters width and height correspond to the amount that points are shaken
ggplot(all_alaska_flights, aes(x = dep_delay, y = arr_delay)) + 
  geom_jitter(width = 30, height = 20)




# Looking at LINEGRAPHS next
?weather

early_january_weather <- weather %>% 
  filter(origin == "EWR" & month == 1 & day <=15)


# Plot the weather to a lineplot
ggplot(early_january_weather, aes(x = time_hour, y = temp)) +
  geom_line()

ggplot(weather, aes(x = time_hour, y = temp)) + 
  geom_line() + 
  facet_grid(~origin)


# HISTOGRAMS

# First plot weather data using a scatter plot with alpha parameter to show which values are more frequent
ggplot(weather, aes(x = temp, y = 0)) + 
  geom_point(alpha = 0.008)


# Next plot the same data using histogram
ggplot(weather, aes(x = temp)) + 
  geom_histogram(bins = 60, color = "white")

# We can also specify the binwidth instead of the amount of bins
ggplot(weather, aes(x = temp)) + 
  geom_histogram(binwidth = 5, color = "white")

# FACETTING
# facet the histograms so that we have a histogram of each month
ggplot(weather, aes(x = temp)) +
  geom_histogram(binwidth = 3, color= "white") + 
  facet_wrap(~month, nrow = 3)

# Boxplot - make the same plot using a boxplot, where we get a five-number-summary of the distribution
ggplot(weather, aes(x = factor(month), y = temp)) + 
  geom_boxplot()


# Using bar plot
# All the flights leaving from NY airport in 2013
ggplot(flights, aes(x = carrier)) + 
  geom_bar()

# Notice that histogram is used for numerical values and bar is used for categorical variables (for example character variables)

kable(airlines)


# Create flights table
flights_table <- flights %>%
  group_by(carrier) %>%
  summarize(number = n())

# Join flights data with airports data using the origin/faa as a key
flights_namedports <- flights %>%
  inner_join(airports, by = c("origin" = "faa"))


# Next we use barplot to distinguish the leaving planes from different airports
ggplot(flights_namedports, aes(x = carrier, fill = name)) + 
  geom_bar()
# To avoid stacking the plot we can use keyword "Dodge" to have the bars side-by-side
ggplot(flights_namedports, aes(x = carrier, fill = name)) + 
  geom_bar(position = "dodge")
