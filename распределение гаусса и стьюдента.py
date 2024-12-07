import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t

# Диапазон значений для оси X
x = np.linspace(-5, 5, 1000)

# Параметры распределений
mean = 0  # Среднее значение для нормального распределения
std_dev = 1  # Стандартное отклонение для нормального распределения

# Расчет плотности вероятности
y_gauss = norm.pdf(x, mean, std_dev)  # Нормальное распределение
y_student_10 = t.pdf(x, df=10)  # Распределение Стьюдента с 10 степенями свободы
y_student_1 = t.pdf(x, df=1)  # Распределение Стьюдента с 1 степенью свободы

# Визуализация
plt.figure(figsize=(10, 6))

# Построение кривых
plt.plot(x, y_gauss, label="Распределение Гаусса (σ=1)", color="blue")
plt.plot(x, y_student_10, label="Распределение Стьюдента (N=10)", color="green")
plt.plot(x, y_student_1, label="Распределение Стьюдента (N=1)", color="red", linestyle='--', linewidth=1)

# Настройки графика
plt.xlabel("Значение измеряемой величины")
plt.ylabel("Плотность вероятности")
plt.legend()
plt.grid(True)
plt.show()