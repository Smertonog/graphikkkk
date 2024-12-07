import matplotlib.pyplot as plt
import numpy as np

# Данные для всех графиков
true_value = 100
n_measurements = 10
x = np.arange(n_measurements)  # Номер измерения

# Параметры погрешностей
k = 5  # Случайная погрешность для data_D
instrumental_error_C = np.random.normal(0, 2, n_measurements)  # Небольшая случайная погрешность для data_C
instrumental_error_D = 0  # Приборная погрешность для data_D

# Данные для каждого графика
data_A = np.full(n_measurements, true_value)
data_B = np.full(n_measurements, true_value + 5)
data_C = true_value + np.linspace(0, 10, n_measurements) + instrumental_error_C
data_D = true_value + np.random.normal(k, k, n_measurements) + instrumental_error_D
data_F = true_value + np.random.normal(0, 10, n_measurements)

# Настройки цвета и меток
colors = ['skyblue', 'lightgreen', 'orchid', 'skyblue', 'skyblue']
labels = ['A', 'Б', 'В', 'Г', 'Д']
data_list = [data_A, data_B, data_C, data_D, data_F]

# Построение графиков
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for i, ax in enumerate(axes.flat[:5]):
    ax.scatter(x, data_list[i], color=colors[i], label='Измерения', s=50)  # Увеличен размер точек
    ax.axhline(true_value, color='red', linestyle='--', label='Истинное значение')
    ax.set_title(labels[i], loc='center', fontsize=12, pad=10)  # Центрирование заголовка
    ax.set_xlabel('Номер измерения', fontsize=10)
    ax.set_ylabel('Измеренное значение (см)', fontsize=10)
    ax.set_xlim(-0.5, n_measurements - 0.5)
    ax.set_ylim(70, 130)
    ax.legend(fontsize=9)

# Удаление лишнего пустого графика
fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.show()
