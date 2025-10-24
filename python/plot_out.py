import matplotlib.pyplot as plt

# Read output values dumped during sim (for example via $fwrite)
with open("outputs.txt") as f:
    values = [int(line.strip()) for line in f]

q15_to_float = lambda x: x / 2**15
values_f = list(map(q15_to_float, values))

plt.plot(values_f)
plt.title("Mamba Output Over Time")
plt.xlabel("Cycle")
plt.ylabel("Output Value")
plt.grid(True)
plt.show()
