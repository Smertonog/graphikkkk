import numpy as np
import matplotlib.pyplot as plt

# Функция y(x) = e^x / |x - 2|
def y_function(x):
    return np.exp(x) / np.abs(x - 2)

# Создание массива значений x, исключая точку x=2
x = np.linspace(0.1, 4.9, 500)
x = x[x != 2]  # Исключаем точку разрыва

# Вычисление y(x)
y = y_function(x)

# Параметры местоположения подписи x
x_label_x = 4.5  # Положение по оси X
x_label_y = -0.5  # Положение по оси Y

# Построение графика
fig, ax1 = plt.subplots(figsize=(6, 6))  # Устанавливаем одинаковую высоту и ширину

# Линия функции
ax1.semilogy(x, y, color='blue', linewidth=1)

# Настройка осей
ax1.set_xlim(0, 5)  # Ось x от 0 до 5
ax1.set_ylim(1, 10**4)  # Ось y от 1 до 10^4

# Настройка левой оси (y)
ax1.set_yticks([1, 10, 100, 1000, 10000])
ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda y, _: "1" if y == 1 else f"$10^{{{int(np.log10(y))}}}$"))

# Настройка нижней оси (x)
ax1.set_xticks([0, 1, 2, 3, 4])
ax1.set_xticklabels(['0', '1', '2', '3', '4'])

# Добавление правой оси (lg y)
ax2 = ax1.twinx()
ax2.set_ylim(0, 4)  # Ось lg y от 0 до 4
ax2.set_yticks([0, 1, 2, 3, 4])

# Размещение подписи lgy на одной высоте с y
ax1.text(0.1, 10**4.1, '$y$', fontsize=12, rotation=0, va='center')  # Подпись y
ax2.text(4.7, 4.11, '$\\lg y$', fontsize=12, rotation=0, va='center')  # Подпись lg y

# Подпись нижней оси
ax1.text(4.8, 0.7,'$x$', fontsize=14)  # Перемещаем подпись x

# Убираем сетку
ax1.grid(False)

plt.tight_layout()
plt.show()
