import matplotlib.pyplot as plt
import numpy as np

# Данные из таблицы
delta_p = [7.8, 15.6, 23.4, 31.3, 39.0, 46.9, 54.7, 62.6, 78.3, 86.0, 87.6, 93.9, 101.6, 109.6, 118.0]
v_avg = [35, 65, 78, 126, 142, 171, 194, 226, 245, 258, 258, 271, 277, 284, 290]

# Увеличиваем диапазон вертикальной оси до 450
plt.figure(figsize=(10, 7))
plt.plot(delta_p, v_avg, 'o', label="Экспериментальные данные", markersize=8)  # точки данных
plt.plot(delta_p[:9], np.polyval(np.polyfit(delta_p[:9], v_avg[:9], 1), delta_p[:9]), 
        '-', label="Линейная аппроксимация", linewidth=2)  # линейная часть

# Оформление графика
plt.xlabel("$\Delta p$, Па/м", fontsize=14)
plt.ylabel("$\\bar{v}$, мм/с", fontsize=14)
plt.ylim(0, 350)  # Устанавливаем лимит по вертикальной оси
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Показать график
plt.show()


