import numpy as np
import matplotlib.pyplot as plt

# Задаем диапазон значений x
x = np.linspace(-2, 2, 500)
x0, y0 = 0, 0  # центральная точка (x0, y0)
dx, dy = 0.4, 0.4  # отклонения по x и y

# Установка шрифта на Times New Roman с размером 16pt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

# Основная кривая
y = np.tanh(1.5*x)

# Пунктирные кривые, ограничивающие область
y_upper = y + 0.3
y_lower = y - 0.3

# Настройка графика
plt.figure(figsize=(8, 6))
plt.plot(x, y, linewidth=2, color='blue', label='Main curve')  # Основная кривая
plt.plot(x, y_upper, 'b--', linewidth=1)  # Верхняя пунктирная кривая
plt.plot(x, y_lower, 'b--', linewidth=1)  # Нижняя пунктирная кривая

# Линии, параллельные осям, через (x0 ± dx) и (y0 ± dy)
plt.axhline(y=y0, color='k', linestyle='--', linewidth=0.8)
plt.axhline(y=y0 + dy, color='k', linestyle='--', linewidth=0.8)
plt.axhline(y=y0 - dy, color='k', linestyle='--', linewidth=0.8)
plt.axvline(x=x0, color='k', linestyle='--', linewidth=0.8)
plt.axvline(x=x0 + dx, color='k', linestyle='--', linewidth=0.8)
plt.axvline(x=x0 - dx, color='k', linestyle='--', linewidth=0.8)

# Точки A, B, C и D
plt.plot(x0, y0, 'ko')  # Точка A
plt.text(x0, y0, ' A', ha='right', va='top')

plt.plot(x0, np.tanh(1.5*x0)+0.3, 'ko')  # Точка B
plt.text(x0, y0 + dy, ' B', ha='right', va='bottom')

plt.plot(x0 + dx, np.tanh(1.5*(x0 + dx)), 'ko')  # Точка C
plt.text(x0 + dx, y0 + dy, ' C', ha='left', va='bottom')

plt.plot(x0 + dx, y0, 'ko')  # Точка D
plt.text(x0 + dx, y0, ' D', ha='left', va='top')

# Настройка осей
plt.xlim(x0 - 2 * dx, x0 + 2 * dx)
plt.ylim(y0 - 2 * dy, y0 + 2 * dy)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.xticks([x0 - dx, x0, x0 + dx], [r'$x_0 - \Delta x$', r'$x_0$', r'$x_0 + \Delta x$'])
plt.yticks([y0 - dy, y0, y0 + dy], [r'$y_0 - \Delta y$', r'$y_0$', r'$y_0 + \Delta y$'])

# Показ графика
plt.grid(False)  # Отключаем сетку для чистоты изображения
plt.show()
