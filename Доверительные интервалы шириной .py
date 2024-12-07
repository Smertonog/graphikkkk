"""import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Параметры нормального распределения
mean = 0    # Среднее значение (x̄)
std_dev = 1 # Стандартное отклонение (σ)

# Генерация значений x для графиков
x = np.linspace(-4, 4, 1000)
y = norm.pdf(x, mean, std_dev)

# Создание фигуры
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Функция для настройки графика
def configure_plot(ax, title, fill_range):
    ax.plot(x, y, 'k-', label='f(x)')
    ax.fill_between(x, y, where=fill_range, color='gray', alpha=0.5)
    ax.set_title(title, fontsize=12)
    ax.axvline(mean, color='black', linestyle='--', linewidth=0.8)
    ax.axvline(mean - std_dev * (1 if "(a)" in title else 2), color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(mean + std_dev * (1 if "(a)" in title else 2), color='gray', linestyle='--', linewidth=0.8)
    ax.text(-3.8, max(y), 'f(x)', fontsize=12, verticalalignment='top', horizontalalignment='left')
    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)  # Убираем числа на осях

# График (a): доверительный интервал ±σ
configure_plot(axes[0], "(a) ±σ (68%)", (x >= mean - std_dev) & (x <= mean + std_dev))

# График (б): доверительный интервал ±2σ
configure_plot(axes[1], "(б) ±2σ (95%)", (x >= mean - 2*std_dev) & (x <= mean + 2*std_dev))

# Настройка и отображение
plt.tight_layout()
plt.show()"""

#ВАРИАНТ 2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Параметры нормального распределения
mean = 0    # Среднее значение (x̄)
std_dev = 1 # Стандартное отклонение (σ)

# Генерация значений x для графиков
x = np.linspace(-4, 4, 1000)
y = norm.pdf(x, mean, std_dev)

# Создание фигуры
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# График (a): доверительный интервал ±σ
axes[0].plot(x, y, 'k-', label='f(x)')
axes[0].fill_between(x, y, where=(x >= mean - std_dev) & (x <= mean + std_dev), color='gray', alpha=0.5)
axes[0].set_title("(a) ±σ (68%)", fontsize=12)
axes[0].set_xlabel("x")
axes[0].set_ylabel("f(x)")
axes[0].axvline(mean, color='black', linestyle='--', linewidth=0.8)
axes[0].axvline(mean - std_dev, color='gray', linestyle='--', linewidth=0.8)
axes[0].axvline(mean + std_dev, color='gray', linestyle='--', linewidth=0.8)
axes[0].grid()

# График (б): доверительный интервал ±2σ
axes[1].plot(x, y, 'k-', label='f(x)')
axes[1].fill_between(x, y, where=(x >= mean - 2*std_dev) & (x <= mean + 2*std_dev), color='gray', alpha=0.5)
axes[1].set_title("(б) ±2σ (95%)", fontsize=12)
axes[1].set_xlabel("x")
axes[1].set_ylabel("f(x)")
axes[1].axvline(mean, color='black', linestyle='--', linewidth=0.8)
axes[1].axvline(mean - 2*std_dev, color='gray', linestyle='--', linewidth=0.8)
axes[1].axvline(mean + 2*std_dev, color='gray', linestyle='--', linewidth=0.8)
axes[1].grid()

# Настройка и отображение
plt.tight_layout()
plt.show()