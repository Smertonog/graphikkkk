import numpy as np
import matplotlib.pyplot as plt

# Данные для построения графика
x = np.linspace(1, 10, 500)
y = x**3

# Создание фигуры с двумя подграфиками
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Линейный масштаб (левый график)
axes[0].plot(x, y, color='black')
axes[0].set_title(r'$a$', fontsize=14, loc='center')
axes[0].set_xlabel(r'$x$', fontsize=14, labelpad=1)  # Сдвиг подписи оси x
axes[0].set_ylabel(r'$y$', fontsize=14, labelpad=0)  # Сдвиг подписи оси y
axes[0].set_xlim(1, 10)
axes[0].set_ylim(0, 100)

# Логарифмический масштаб (правый график)
axes[1].plot(x, y, color='black')
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].set_title(r'$б$', fontsize=14, loc='center')
axes[1].set_xlabel(r'$x$', fontsize=14, labelpad=1)  # Сдвиг подписи оси x
axes[1].set_ylabel(r'$y$', fontsize=14, labelpad=0)  # Сдвиг подписи оси y

# Указываем простые значения на осях (1, 10, 100, 1000)
axes[1].set_xticks([1, 10])
axes[1].set_xticklabels([r'$1$', r'$10$'])
axes[1].set_yticks([1, 10, 100, 1000])
axes[1].set_yticklabels([r'$1$', r'$10$', r'$100$', r'$1000$'])

# Настройка общих параметров
for ax in axes:
    ax.tick_params(axis='both', which='major', labelsize=12)

# Показать график
plt.tight_layout()
plt.show()
