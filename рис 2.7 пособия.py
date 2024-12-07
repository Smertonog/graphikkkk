import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = "Tahoma"

# Параметры графика
x = np.linspace(-2, 2, 500)
z = x**3 + x + 1  # Функция z(x)

# Константы для подписей
x_min = -1
x_mean = 0
x_max = 1
z_min = x_min**3 + x_min + 1
z_mean = x_mean**3 + x_mean + 1
z_max = x_max**3 + x_max + 1
delta_z = z_max - z_mean

# Построение графика
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, z, color='blue', label='')  # График функции z(x)

# Горизонтальные и вертикальные линии
ax.hlines([z_min, z_mean, z_max], x[0], x[-1], colors='black', linestyles='dashed', linewidth=0.8)
ax.vlines([x_min, x_mean, x_max], min(z-1), max(z), colors='black', linestyles='dashed', linewidth=0.8)

# Стрелка для Δz
ax.annotate('', xy=(x_mean -1, z_max ), xytext=(x_mean-1, z_mean), arrowprops=dict(arrowstyle='<->', color='black'))
ax.text(x[0] + 0.8, z_mean + delta_z / 2, r'$\Delta z$', fontsize=14, verticalalignment='center')

# Подписи ближе к осям
ax.text(x[-1] * 0.9, z[-1], r'$z(x)$', fontsize=14, verticalalignment='bottom', horizontalalignment='right')
ax.text(x[0] - 0.02 , z_min, r'$z_{\min}$', fontsize=14, verticalalignment='center', horizontalalignment='right')
ax.text(x[0] - 0.02, z_mean, r'$z_{\text{наил}}$', fontsize=14, verticalalignment='center', horizontalalignment='right')
ax.text(x[0] - 0.02, z_max, r'$z_{\max}$', fontsize=14, verticalalignment='center', horizontalalignment='right')
ax.text(x_min, min(z) - 1.5, r'$\bar{x}-\Delta x$', fontsize=14, verticalalignment='top', horizontalalignment='center')
ax.text(x_mean, min(z) - 1.5, r'$\bar{x}$', fontsize=14, verticalalignment='top', horizontalalignment='center')
ax.text(x_max, min(z) - 1.5, r'$\bar{x}+\Delta x$', fontsize=14, verticalalignment='top', horizontalalignment='center')

# Настройка осей
ax.set_xlim([-2, 2])
ax.set_ylim([min(z) - 1, max(z) + 1])
ax.axhline(0, color='black', linewidth=0)  # Ось x
ax.axvline(0, color='black', linewidth=0.8)  # Ось z
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Убираем числа на осях
ax.tick_params(labelbottom=False, labelleft=False)

# Подписи осей
ax.text(1.9, -11, 'x', fontsize=14, verticalalignment='center')  # Подпись оси x
ax.text(-2.1, max(z) , 'z', fontsize=14, horizontalalignment='center')  # Подпись оси z


plt.tight_layout()
plt.show()
