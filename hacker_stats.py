import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def perform_bernoulli_trials(n, p):
    """Perform n Bernoulli trials with success probability p
    and return number of successes."""
    # Initialize number of successes: n_success
    n_success = 0

    # Perform trials
    for i in range(n):
        # Choose random number between zero and one: random_number
        random_number = np.random.random()

        # If less than p, it's a success so add one to n_success
        if random_number < p:
            n_success += 1

    return n_success


def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n + 1) / n

    return x, y


# Seed the random number generator
np.random.seed(42)

# Initialize random numbers: random_numbers
random_numbers = np.empty(100000)

# Generate random numbers by looping over range(100000)
for it in range(100000):
    random_numbers[it] = np.random.random()

# Plot a histogram
_ = plt.hist(random_numbers)

# Show the plot
plt.show()

# How many defaults might we expect?
# Let's say a bank made 100 mortgage loans. It is possible that anywhere between 0
# and 100 of the loans will be defaulted upon. You would like to know the probability of getting a given number of
# defaults, given that the probability of a default is p = 0.05. To investigate this, you will do a simulation. You
# will perform 100 Bernoulli trials using the perform_bernoulli_trials() function you wrote in the previous exercise
# and record how many defaults we get. Here, a success is a default. (Remember that the word "success" just means
# that the Bernoulli trial evaluates to True, i.e., did the loan recipient default?) You will do this for another 100
# Bernoulli trials. And again and again until we have tried it 1000 times. Then, you will plot a histogram describing
# the probability of the number of defaults.

# Seed random number generator
np.random.seed(42)

# Initialize the number of defaults: n_defaults
n_defaults = np.empty(1000)

# Compute the number of defaults
for i in range(1000):
    n_defaults[i] = perform_bernoulli_trials(100, 0.05)

# Plot the histogram with default number of bins; label your axes
_ = plt.hist(n_defaults, density=True)
_ = plt.xlabel('number of defaults out of 100 loans')
_ = plt.ylabel('probability')

# Show the plot
plt.show()

print(stats.describe(n_defaults))

# Compute ECDF: x, y
x, y = ecdf(n_defaults)

# Plot the ECDF with labeled axes
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.xlabel("defaults")
_ = plt.ylabel("ECDF")

# Show the plot
plt.show()

# If interest rates are such that the bank will lose money if 10 or more of its loans are defaulted upon, what is the
# probability that the bank will lose money?
# Compute the number of 100-loan simulations with 10 or more defaults: n_lose_money
n_lose_money = np.sum(n_defaults >= 10)

# Compute and print probability of losing money
print('Probability of losing money =', n_lose_money / len(n_defaults))

# The number r of successes in n Bernoulli trials with probability p of success, is Binomially distributed
# np.random.binomial(100, 0.05, size=10000)

# Plotting the binomial PMF using a histogram The trick is setting up the edges of the bins to pass to plt.hist() via
# the bins keyword argument. We want the bins centered on the integers. So, the edges of the bins should be -0.5,
# 0.5, 1.5, 2.5, ... up to max(n_defaults) + 1.5. You can generate an array like this using np.arange() and then
# subtracting 0.5 from the array.
# Compute bin edges: bins
bins = np.arange(min(n_defaults), max(n_defaults) + 2) - 0.5
print(bins)

# Generate histogram
_ = plt.hist(n_defaults, density=True, bins=bins)

# Label axes
_ = plt.xlabel("number of successes (defaults)")
_ = plt.ylabel("probability")

# Show the plot
plt.show()

# The Poisson distribution is a limit of the Binomial distribution for rare events.
# This is just like the Poisson story we discussed in the video, where we get on average 6 hits on a website per hour.
# Draw 10,000 samples out of Poisson distribution: samples_poisson
samples_poisson = np.random.poisson(10, size=10000)

# Print the mean and standard deviation
print('Poisson:     ', np.mean(samples_poisson), np.std(samples_poisson))

# Specify values of n and p to consider for Binomial: n, p
n = [20, 100, 1000]
p = [0.5, 0.1, 0.01]

# Draw 10,000 samples for each n,p pair: samples_binomial
for i in range(3):
    samples_binomial = np.random.binomial(n[i], p[i], size=10000)

    # Print results
    # The standard deviation of the Binomial distribution gets closer and closer to that of the Poisson distribution
    # as the probability p gets lower and lower.
    print('n =', n[i], 'Binom:', np.mean(samples_binomial), np.std(samples_binomial))

# 1990 and 2015 featured the most no-hitters of any season of baseball (there were seven). Given that there are on
# average 251/115 no-hitters per season, what is the probability of having seven or more in a season?
# Mean/average: There were 251 no-hitter games (a team that batted recorded no hits in 9 innings) in 115 seasons.
# Draw 10,000 samples out of Poisson distribution: n_nohitters
n_nohitters = np.random.poisson(251/115, size=10000)

# Compute number of samples that are seven or greater: n_large
n_large = np.sum(n_nohitters >= 7)

# Compute probability of getting seven or more: p_large
p_large = n_large / 10000

# Print the result
print('Probability of seven or more no-hitters:', p_large)
