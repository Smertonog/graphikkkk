import numpy as np
import matplotlib.pyplot as plt

# Данные для графика
x = np.array([1, 2, 3, 4, 5])  # Координаты x
y = np.array([1.2, 2.3, 3.1, 4.4, 5.3])  # Координаты y
x_error = np.array([0.2, 0.3, 0.2, 0.3, 0.2])  # Горизонтальная погрешность
y_error = np.array([0.3, 0.4, 0.3, 0.4, 0.3])  # Вертикальная погрешность

# Основная линия (пример наилучшей прямой)
slope, intercept = np.polyfit(x, y, 1)
line_x = np.linspace(0.5, 5.5, 100)
line_y = slope * line_x + intercept

# Дополнительные линии (вспомогательные прямые)
line1_y = slope * x + intercept - 0.2  # Смещение вниз для примера
line2_y = slope * x + intercept + 0.2  # Смещение вверх для примера

# Создание графика
plt.figure(figsize=(8, 6))
plt.errorbar(x, y, xerr=x_error, yerr=y_error, fmt='o', color='black', label="Экспериментальные точки")
plt.plot(line_x, line_y, color='gray', label="Наилучшая прямая")
plt.plot(x, line1_y, '--', color='gray', alpha=0.5)
plt.plot(x, line2_y, '--', color='gray', alpha=0.5)

# Настройка осей и текста
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$y$', fontsize=14)
plt.xticks(ticks=x, labels=[f'$x_{i}$' for i in range(1, len(x) + 1)])
plt.yticks(ticks=[1, 2, 3, 4, 5], labels=[f'$y_{i}$' for i in range(1, len(x) + 1)])
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

plt.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()
