import matplotlib.pyplot as plt
import math

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 11}
plt.rc('font', **font)

a = math.sqrt(10)
# Naive Bayes data
nb_pos = [1, 6]
nb_means = [0.953, 0.946]
nb_errors = [0.032/a, 0.018/a]

# JAGS dnorm data
norm_pos = [2, 7]
norm_means = [0.98, 0.987]
norm_errors = [0.032/a, 0.01/a]

# JAGS ddexp data
dexp_pos = [3, 8]
dexp_means = [0.973, 0.986]
dexp_errors = [0.064/a, 0.012/a] # Calculate

fig, ax = plt.subplots()
ax.yaxis.grid(True)
ax.set_axisbelow(True)
ax.bar(nb_pos, nb_means, yerr=nb_errors, align='center', alpha=1, ecolor='black', capsize=10, color='#e54c19')
ax.bar(norm_pos, norm_means, yerr=norm_errors, align='center', alpha=1, ecolor='black', capsize=10, color='#11c123')
ax.bar(dexp_pos, dexp_means, yerr=dexp_errors, align='center', alpha=1, ecolor='black', capsize=10, color='#2f51d8')
ax.set_ylabel('Accuracy')
ax.set_xticks(norm_pos)
ax.set_xticklabels(['Iris', 'Letter-Recognition'])
ax.set_ylim([0.85, 1.04])
ax.legend(["Naive Bayes", "Normal Priors", "Double Exp. Priors"], loc=9)

plt.savefig('results_bar_plot.png', dpi=110)

