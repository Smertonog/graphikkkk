import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Настройки
mean = 0  # Среднее значение для всех распределений
std_devs = [1, 2, 3, 4]  # Разные значения стандартного отклонения
x = np.linspace(-10, 10, 1000)  # Диапазон значений для оси X

# Визуализация
plt.figure(figsize=(10, 6))

for std_dev in std_devs:
    y = norm.pdf(x, mean, std_dev)  # Вычисляем плотность вероятности для данного sigma
    plt.plot(x, y, label=f'σ = {std_dev}')

plt.xlabel("Значение")
plt.ylabel("Плотность вероятности")
plt.legend(title="Стандартное отклонение")
plt.grid(True)
plt.show()