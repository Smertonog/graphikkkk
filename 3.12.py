import numpy as np
import matplotlib.pyplot as plt

# Данные
x = np.linspace(-2, 2, 100)  # Диапазон x
y = -x**2 + 4  # Основная парабола
y_upper = -x**2 + 4 + 0.5  # Верхняя граница
y_lower = -x**2 + 4 - 0.5  # Нижняя граница

# Точки на графике
x_points = np.array([-1, 0, 1])
y_points = -x_points**2 + 4
errors_x = [0.5, 0.5, 0.5]  # Погрешности по x
errors_y = [0.5, 0.5, 0.5]  # Погрешности по y

# Цвета для точек
colors = ['red', 'blue', 'green']  # Список цветов для каждой точки

# Создание графика
fig, ax = plt.subplots(figsize=(6, 7))  # Создаем фигуру и оси

# Основная кривая и пунктирные линии с новыми цветами
ax.plot(x, y, color='blue', linewidth=2)  # Основная кривая (синий)
ax.plot(x, y_upper, color='green', linestyle='--')  # Верхняя граница (зеленый)
ax.plot(x, y_lower, color='green', linestyle='--')  # Нижняя граница (красный)

# Погрешности
ax.errorbar(x_points, y_points, xerr=errors_x, yerr=errors_y, fmt='o', color='gray', ecolor='gray', capsize=3)

# Вертикальные линии
ax.axvline(-1, linestyle='--', color='k')
ax.axvline(1, linestyle='--', color='k')
ax.text(-1.3, -1, r'$x_M - \Delta x$', fontsize=12, va='center')
ax.text(0.7, -1, r'$x_M + \Delta x$', fontsize=12, va='center')

# Горизонтальные линии
ax.axhline(4.5, linestyle='--', color='k')
ax.axhline(3.5, linestyle='--', color='k')
ax.text(-2.6, 4.5, r'$y_M + \Delta y$', fontsize=12, ha='center', va='center')
ax.text(-2.6, 3.5, r'$y_M - \Delta y$', fontsize=12, ha='center', va='center')
ax.text(-2.6, 4, r'$y_M$', fontsize=12, ha='center', va='center')

# Максимальная точка
ax.axhline(4, color='k')  # Горизонтальная линия через максимум
ax.axvline(0, linestyle='-', color='k')  # Вертикальная линия через максимум

# Включение сетки
ax.grid(True)  # Включаем сетку

# Скрытие меток осей (делаем их прозрачными)
ax.tick_params(axis='both', which='both', labelcolor='white', length=0)

# Подписи осей по центру
ax.text(2.05, -1.05, '$x$', fontsize=14, ha='center')  # Подпись оси x (внизу)
ax.text(-2.5, 4.9, '$y$', fontsize=14, va='center', rotation=90)  # Подпись оси y (слева)

# Показ графика
plt.show()
