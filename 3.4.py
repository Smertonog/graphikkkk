import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Данные для экспериментальных точек (их можно менять)
x_data = [0.5, 1.35, 2.0, 2.6, 3.2, 3.8, 4.4]
y_data = [1.8, 3, 4.2, 5.4, 6.7, 7.9, 8.9]
x_errors = [0.2, 0.4, 0.2, 0.3, 0.2, 0.1, 0.2]
y_errors = [0.5, 0.5, 0.5, 0.6, 0.5, 0.8, 0.5]

# Коэффициенты для прямой y = kx + b
k = 1.7  # Угол наклона
b = 1

x_line = np.linspace(0, 5, 100)
y_line = k * x_line + b

# Произвольные точки на прямой (их можно менять)
x1, x2 = 1.3, 3.5  # Меняемые координаты x1 и x2
y1, y2 = k * x1 + b, k * x2 + b  # Вычисляем y1 и y2 на основе прямой

# Координаты для линии deltaYmax
x_dymax = 3.7
y_dymax_start = 7.4
y_dymax_end = 8.7

# Верхняя граница погрешности 6-й точки
y_intersection = y_data[5] + y_errors[5]
y_intersection_1 = y_data[5] + y_errors[5]

# Настройка зеленой горизонтальной линии DeltaXmax
x_dxmax_start = 1.16
x_dxmax_end = 1.78
y_dxmax = 2.75

# Верхняя и нижняя границы погрешности 2-й точки
x_error_left = x_data[1] - x_errors[1]
x_error_right = x_data[1] + x_errors[1]

# Координаты подписей осей
x_label_position = (4.7, -0.3)  # Позиция подписи оси x
y_label_position = (-0.1, 9.0)  # Позиция подписи оси y

# Настройка графика
fig, ax = plt.subplots(figsize=(10, 7))

# Построение экспериментальных точек с погрешностями
ax.errorbar(x_data, y_data, xerr=x_errors, yerr=y_errors, fmt='o', color='black', ecolor='black', capsize=3)

# Построение прямоугольников для погрешностей
for x, y, x_err, y_err in zip(x_data, y_data, x_errors, y_errors):
    rect = Rectangle((x - x_err, y - y_err), 2 * x_err, 2 * y_err, linewidth=0.2, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

# Построение наилучшей прямой
ax.plot(x_line, y_line, color='blue', label='Best fit line')  # Линия сделана синей

# Подписи точек x1, y1 и x2, y2
ax.plot(x1, y1, 'o', color='white', markersize=10, markeredgecolor='black', label='$x_1, y_1$')
ax.plot(x2, y2, 'o', color='white', markersize=10, markeredgecolor='black', label='$x_2, y_2$')

# Горизонтальная ось на высоте 1
ax.axhline(1, linestyle="--", color='black', linewidth=0.8)

# Вертикальные и горизонтальные пунктирные линии для x1, x2, y1, y2
ax.plot([x1, x1], [0, y1], linestyle='--', color='orange', linewidth=0.8)  # Линии сделаны оранжевыми
ax.plot([x2, x2], [0, y2], linestyle='--', color='orange', linewidth=0.8)  # Линии сделаны оранжевыми
ax.plot([0, x1], [y1, y1], linestyle='--', color='orange', linewidth=0.8)
ax.plot([0, x2], [y2, y2], linestyle='--', color='orange', linewidth=0.8)

# Пунктирные линии от y1, y2 до оси y
ax.plot([0, 0], [1, y1], linestyle='--', color='orange', linewidth=0.8)
ax.plot([0, 0], [1, y2], linestyle='--', color='orange', linewidth=0.8)

# Новая стрелка и подпись для b
ax.annotate('', xy=(0.2, 1), xytext=(0.2, 0), arrowprops=dict(facecolor='red', color='red', arrowstyle='<->'))  # Красная стрелка
ax.text(0.25, 0.5, '$b$', fontsize=12, color='red', va='center')  # Подпись b, красного цвета

# Зеленая линия для deltaYmax
ax.annotate('', xy=(x_dymax, y_dymax_end), xytext=(x_dymax, y_dymax_start),
            arrowprops=dict(facecolor='green', color='green', arrowstyle='<->'))
plt.text(x_dymax - 0.4, (y_dymax_start + y_dymax_end) / 2, r'$\Delta y_{\text{max}}$', fontsize=14, color='green', va='center')

# Зеленый пунктир до пересечения с погрешностью 6-й точки
ax.plot([x_dymax, x_data[5]], [y_dymax_end, y_intersection], linestyle='--', color='green', linewidth=0.8)
ax.plot([x_dymax, x_data[5]], [y_dymax_end-1.22, y_intersection_1-1.22], linestyle='--', color='green', linewidth=0.8)

# Зеленая линия для DeltaXmax
ax.annotate('', xy=(x_dxmax_end, y_dxmax), xytext=(x_dxmax_start, y_dxmax),
            arrowprops=dict(facecolor='green', color='green', arrowstyle='<->'))
ax.text((x_dxmax_start + x_dxmax_end+0.2) / 2, y_dxmax - 0.5, r'$\Delta x_{\text{max}}$', fontsize=14, color='green', ha='center')

# Зеленые пунктирные линии до пересечения с погрешностью 2-й точки
ax.plot([1.16, 1.16], [y_dxmax, y_data[1]], linestyle='--', color='green', linewidth=0.8)
ax.plot([1.75, 1.75], [y_dxmax, y_data[1]], linestyle='--', color='green', linewidth=0.8)

# Подпись y1, y2, x1, x2
ax.text(-0.05, y1, '$y_1$', fontsize=14, va='center', ha='right', color='orange')  # Цвет подписи y1
ax.text(-0.05, y2, '$y_2$', fontsize=14, va='center', ha='right', color='orange')  # Цвет подписи y2
ax.text(x1, -0.3, '$x_1$', fontsize=14, ha='center', color='orange')  # Цвет подписи x1
ax.text(x2, -0.3, '$x_2$', fontsize=14, ha='center', color='orange')  # Цвет подписи x2

# Подпись осей с учетом координат
ax.text(*x_label_position, '$x$', fontsize=14, ha='center', va='center')
ax.text(*y_label_position, '$y$', fontsize=14, ha='center', va='center')

# Скрытие меток на осях
ax.set_xticks([0])  # Оставляем только метку 0 на оси x
ax.set_yticks([0])  # Оставляем только метку 0 на оси y
ax.set_xlim(0, 5)
ax.set_ylim(0, 10)

plt.tight_layout()
plt.show()
