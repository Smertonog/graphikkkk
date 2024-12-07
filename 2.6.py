import numpy as np
import matplotlib.pyplot as plt

# Генерация случайных данных
np.random.seed(42)
x_values = np.array([1, 2, 3, 4, 5, 6, 7, 8])
true_value = 5
y_values = true_value + np.random.normal(0, 0.1, size=len(x_values))  # малый разброс
y_values_large = true_value + np.random.normal(0, 0.5, size=len(x_values))  # большой разброс

# Погрешности измерений
delta_x = 0.1  # Погрешность одного измерения для графика (а)
delta_x_large = 0.1  # Погрешность одного измерения для графика (б)
sigma_x = np.std(x_values)  # Стандартное отклонение для большой погрешности

# Создаем фигуру и оси
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# График (а) - Разброс меньше погрешности одного измерения
axs[0].errorbar(x_values, y_values, xerr=delta_x, fmt='o', color='black', label='Измеренные данные')
axs[0].hlines(true_value, x_values.min(), x_values.max(), color='red', linestyle='--', label='Истинное значение')
axs[0].set_title('Разброс меньше погрешности одного измерения')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].legend()
axs[0].grid(True)

# График (б) - Разброс больше погрешности одного измерения
axs[1].errorbar(x_values, y_values_large, xerr=sigma_x, fmt='o', color='black', label='Измеренные данные')
axs[1].hlines(true_value, x_values.min(), x_values.max(), color='red', linestyle='--', label='Истинное значение')
axs[1].set_title('Разброс больше погрешности одного измерения')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')
axs[1].legend()
axs[1].grid(True)

# Показываем график
plt.tight_layout()
plt.show()
