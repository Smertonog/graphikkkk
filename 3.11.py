import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Задаем диапазон значений x
x = np.linspace(-2, 2, 500)
x0, y0 = 0, 0  # центральная точка (x0, y0)
dx, dy = 0.4, 0.4  # отклонения по x и y

# Основная кривая
y = np.tanh(1.1*x)

# Увеличиваем угол наклона пунктирных линий
y_upper = y + 0.27   # Добавили наклон для верхней линии
y_lower = y - 0.27   # Добавили наклон для нижней линии

# Настройка графика
plt.figure(figsize=(8, 6))
plt.plot(x, y, linewidth=2, color='blue', label='Основная кривая')  # Основная кривая
plt.plot(x, y_upper, 'b--', linewidth=1)  # Верхняя пунктирная кривая
plt.plot(x, y_lower, 'b--', linewidth=1)  # Нижняя пунктирная кривая

# Пунктирные линии по оси X и Y
plt.axhline(y=y0, color='k', linestyle='--', linewidth=0.8)  # Горизонтальная линия через y0
plt.axhline(y=y0 + dy-0.045, color='k', linestyle='--', linewidth=0.8)  # Горизонтальная линия через y0 + dy
plt.axhline(y=y0 - dy+0.045, color='k', linestyle='--', linewidth=0.8)  # Горизонтальная линия через y0 - dy
plt.axvline(x=x0, color='k', linestyle='--', linewidth=0.8)  # Вертикальная линия через x0
plt.axvline(x=x0 + dx-0.157, color='k', linestyle='--', linewidth=0.8)  # Вертикальная линия через x0 + dx
plt.axvline(x=x0 - dx+0.157, color='k', linestyle='--', linewidth=0.8)  # Вертикальная линия через x0 - dx

# Точки A, B, C и D
plt.plot(x0, y0, 'ko')  # Точка A
plt.text(x0 + 0.05, y0, ' A', ha='right', va='top', fontsize=14)

plt.plot(x0, np.tanh(1.5*x0)+0.275, 'ko')  # Точка B
plt.text(x0, np.tanh(1.5*x0)+0.275, ' B', ha='right', va='bottom', fontsize=14)

# Смещаем точку C на верхний кончик эллипса
y_C = y0 + dy + 0.8 * dy  # Расположим точку C на верхней части эллипса
plt.plot(x0 + dx - 0.16, y_C - 0.46, 'ko')  # Точка C
plt.text(x0 + dx - 0.23, y_C - 0.46, ' C', ha='left', va='bottom', fontsize=14)

# Смещаем точку D так, чтобы она была внутри пунктирных линий
plt.plot(x0 + dx - 0.16, y0, 'ko')  # Точка D
plt.text(x0 + dx - 0.16, y0 + 0.008, ' D', ha='left', va='top', fontsize=14)

# Коридор погрешностей (область между верхней и нижней границей)
plt.fill_between(x, y_upper, y_lower, color='gray', alpha=0.3, label="Коридор погрешностей")

# Эллипс погрешности с поворотом на 60 градусов
ellipse = Ellipse(
    (x0, y0), width=2*dx, height=0.8*dy, edgecolor='red', facecolor='none', 
    linestyle='--', label="Эллипс погрешности", angle=60  # Угол наклона эллипса 60 градусов
)
plt.gca().add_patch(ellipse)

# Настройка осей
plt.xlim(x0 - 2 * dx, x0 + 2 * dx)
plt.ylim(y0 - 2 * dy, y0 + 2 * dy)

# Подписи для осей X и Y с возможностью изменения их расположения
# Задаем позиции для подписей осей
x_label_pos = (x0 + 0.75, y0 - dy-0.45)
y_label_pos = (x0 - 2 * dx-0.03, y0 + 0.7)  # Для оси Y

# Отображаем подписи осей X и Y в указанных позициях
plt.text(x_label_pos[0], x_label_pos[1], r'$x$', fontsize=14, ha='center', va='center')
plt.text(y_label_pos[0], y_label_pos[1], r'$y$', fontsize=14, ha='center', va='center')

# Подписи осей X и Y
plt.xticks([x0 - dx+0.157, x0, x0 + dx-0.157], [r'$x_0 - \Delta x$', r'$x_0$', r'$x_0 + \Delta x$'], fontsize=14)
plt.yticks([y0 - dy+0.045, y0, y0 + dy-0.045], [r'$y_0 - \Delta y$', r'$y_0$', r'$y_0 + \Delta y$'], fontsize=14)

# Показ графика
plt.grid(False)  # Отключаем сетку для чистоты изображения

plt.show()
