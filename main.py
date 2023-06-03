import numpy as np
import numba as nb
import timeit
import matplotlib.pyplot as plt

# Function with logarithmic time complexity O(log(n))
def logarithmic_time(n):
    return np.log(n)

@nb.jit
def logarithmic_time_numba(n):
    return np.log(n)

# Function with linear time complexity O(n)
def linear_time(n):
    return np.sum(np.arange(n))

@nb.jit
def linear_time_numba(n):
    return np.sum(np.arange(n))

# Function with exponential time complexity O(2^n)
def exponential_time(n):
    return np.sum(2 ** np.arange(n))

@nb.jit
def exponential_time_numba(n):
    return np.sum(2 ** np.arange(n))


# Range of input sizes
input_sizes = np.logspace(1, 6, 10, dtype=np.int32)

print("input sizes: ", input_sizes, "\n")


python_logarithmic_execution_times = []
python_linear_execution_times = []
python_exponential_execution_times = []
numba_logarithmic_execution_times = []
numba_linear_execution_times = []
numba_exponential_execution_times = []

# Run every numba function once to compile
logarithmic_time_numba(input_sizes[-1])
linear_time_numba(input_sizes[-1])
exponential_time_numba(input_sizes[-1])

num_tests = 1000

# Perform the test for each input size
for size in input_sizes:
    python_logarithmic_time_sum = 0
    python_linear_time_sum = 0
    python_exponential_time_sum = 0
    numba_logarithmic_time_sum = 0
    numba_linear_time_sum = 0
    numba_exponential_time_sum = 0

    for _ in range(num_tests):
        # Measure Python execution time
        start = timeit.default_timer()
        logarithmic_time(size)
        stop = timeit.default_timer()
        python_logarithmic_time_sum += stop - start

        start = timeit.default_timer()
        linear_time(size)
        stop = timeit.default_timer()
        python_linear_time_sum += stop - start

        start = timeit.default_timer()
        exponential_time(size)
        stop = timeit.default_timer()
        python_exponential_time_sum += stop - start

        # Measure Numba execution time
        start = timeit.default_timer()
        logarithmic_time_numba(size)
        stop = timeit.default_timer()
        numba_logarithmic_time_sum += stop - start

        start = timeit.default_timer()
        linear_time_numba(size)
        stop = timeit.default_timer()
        numba_linear_time_sum += stop - start

        start = timeit.default_timer()
        exponential_time_numba(size)
        stop = timeit.default_timer()
        numba_exponential_time_sum += stop - start

        #print every 10 percent done
        if _ % (num_tests / 10) == 0:
            print("Progress: ", _ / num_tests * 100, "%", " of ", size, " tests done")

    python_logarithmic_execution_times.append((python_logarithmic_time_sum / num_tests)*1000)
    python_linear_execution_times.append((python_linear_time_sum / num_tests)*1000)
    python_exponential_execution_times.append((python_exponential_time_sum / num_tests)*1000)
    numba_logarithmic_execution_times.append((numba_logarithmic_time_sum / num_tests)*1000)
    numba_linear_execution_times.append((numba_linear_time_sum / num_tests)*1000)
    numba_exponential_execution_times.append((numba_exponential_time_sum / num_tests)*1000)

# Plot the results
fig1, ax1 = plt.subplots()
fig1.set_size_inches(12, 6)
ax1.plot(input_sizes, python_logarithmic_execution_times, label='Python logarithmic time')
ax1.plot(input_sizes, python_linear_execution_times, label='Python linear time')
ax1.plot(input_sizes, python_exponential_execution_times, label='Python exponential time')
ax1.plot(input_sizes, numba_logarithmic_execution_times, label='Numba logarithmic time')
ax1.plot(input_sizes, numba_linear_execution_times, label='Numba linear time')
ax1.plot(input_sizes, numba_exponential_execution_times, label='Numba exponential time')
plt.title('Execution time for different time complexities')
ax1.set_xscale('log')
ax1.set_xticks(input_sizes)
ax1.set_yscale('log')
ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax1.ticklabel_format(axis='x', style='plain')
ax1.set_xlabel('Input size')
ax1.set_ylabel('Execution time (ms)')
ax1.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
plt.legend(loc='upper left')
plt.box(on=True)
plt.show()
