import numpy as np
import matplotlib.pyplot as plt

# Функция y(x) = e^x
def y_function(x):
    return np.exp(x)

# Создание массива значений x
x = np.linspace(0, 5, 500)
y = y_function(x)

# Параметры местоположения подписи x и y
x_label_position_a = (3.8, -10)  # Положение подписи x для левого графика
x_label_position_b = (3.8, -10)  # Положение подписи x для правого графика
y_label_position_a = (-0.5, 90)  # Положение подписи y для левого графика
y_label_position_b = (-0.5, 50)  # Положение подписи y для правого графика
lg_y_label_position = (0, 2)  # Положение подписи lg y

# Построение графиков
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# График (a): обычный масштаб
axs[0].plot(x, y, color='red', linewidth=1)
axs[0].set_xlim(0, 5)
axs[0].set_ylim(0, 100)
axs[0].set_xticks([0, 1, 2, 3, 4])  # Без 5
axs[0].set_yticks([0, 20, 40, 60, 80])  # Без 100
axs[0].text(4.8, -5, '$x$', fontsize=14)  # Подпись x
axs[0].text(-0.2, 95, '$y$', fontsize=14)  # Подпись y
axs[0].text(0.3, 85, '(a)', fontsize=14)  # Метка графика (a)
axs[0].text(3.5, 70, '$e^x$', fontsize=14)  # Подпись функции
axs[0].grid(False)

# График (б): полулогарифмический масштаб
axs[1].semilogy(x, y, color='blue', linewidth=1)
axs[1].set_xlim(0, 5)
axs[1].set_ylim(1, 100)
axs[1].set_xticks([0, 1, 2, 3, 4])
axs[1].set_yticks([1, 10, 100])
axs[1].get_yaxis().set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y)}"))
axs[1].text(4.8,0.8, '$x$', fontsize=14)  # Подпись x
axs[1].text(-0.2, 75, '$y$', fontsize=14)  # Подпись y
axs[1].text(0.3, 50, '(б)', fontsize=14)  # Метка графика (б)
axs[1].text(2.5, 25, '$e^x$', fontsize=14)  # Подпись функции
axs[1].text(5.1,75, '$\\lg y$', fontsize=14)  # Подпись lg y как текст

axs[1].grid(False)

# Настройка правой оси для графика (б)
ax2 = axs[1].twinx()
ax2.set_ylim(0, 2)  # Ось \lg y от 0 до 2
ax2.set_yticks([0, 1, 2])



plt.tight_layout()
plt.show()
