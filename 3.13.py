import numpy as np
import matplotlib.pyplot as plt

# Исходные экспериментальные данные
x_exp = np.array([1, 2, 3, 4, 5])
y_exp = np.array([2.2, 4.1, 6.0, 7.8, 9.9])  # Исходные значения
y_err = np.array([0.6, 0.5, 0.7, 0.5, 0.8])  # Погрешности по y

# Используем метод наименьших квадратов для вычисления углового коэффициента и свободного члена
A = np.vstack([x_exp, np.ones_like(x_exp)]).T
k, b = np.linalg.lstsq(A, y_exp, rcond=None)[0]

# Уменьшаем угол наклона (уменьшаем k)
k *= 0.6  # Уменьшаем наклон

# Смещаем точки так, чтобы они лежали на новой прямой с учетом погрешности
y_exp_corrected = k * x_exp + b  # Корректируем значения точек на прямой

# Погрешность углового коэффициента
Δk = 0.4  # Погрешность углового коэффициента

# Диапазон x для графика (продлеваем диапазон)
x = np.linspace(-5, 8, 500)

# Мы хотим, чтобы вспомогательные прямые пересекались в точке (3, 3.71)
# Для этого находим свободный член (b) для вспомогательных прямых:
# Мы должны решить для b: y = k * x + b, где x = 3, y = 3.71

# Для верхней прямой (с положительной погрешностью)
b_upper = 3.71 - (k + Δk) * 3  # Решаем для b в верхней прямой
upper_line = (k + Δk) * x + b_upper  # Верхняя вспомогательная прямая

# Для нижней прямой (с отрицательной погрешностью)
b_lower = 3.71 - (k - Δk) * 3  # Для нижней прямой свободный член тот же
lower_line = (k - Δk) * x + b_lower  # Нижняя вспомогательная прямая

# Пересечения наилучшей прямой с осями
x_star = -b / k  # Пересечение с осью x
y_star = b       # Пересечение с осью y

# Пересечения вспомогательных прямых с осями
x_star_upper = -b_upper / (k + Δk)  # Пересечение верхней вспомогательной прямой
x_star_lower = -b_lower / (k - Δk)  # Пересечение нижней вспомогательной прямой

# Вычисление центра тяжести
xc = np.mean(x_exp)
yc = np.mean(y_exp_corrected)

# Сдвигаем все элементы графика на 1.5 единицы вверх
shift_y = 1.5

# Сдвигаем все значения y
y_exp_corrected += shift_y
upper_line += shift_y
lower_line += shift_y
y_star += shift_y
yc += shift_y

# Построение графика
plt.figure(figsize=(12, 8))

# Экспериментальные точки с погрешностями (сдвигаем только точки на прямую)
plt.errorbar(x_exp, y_exp_corrected, yerr=y_err, fmt='o', color='black', capsize=5, markersize=8)

# Строим прямые
plt.plot(x, k * x + b + shift_y, label='Наилучшая прямая', color='black', linewidth=2)
plt.plot(x, upper_line, linestyle='--', color='gray', linewidth=2)
plt.plot(x, lower_line, linestyle='--', color='gray', linewidth=2)

# Пересечения с осями
# Обновление координат красной точки (по запросу)
plt.scatter([-1.55], [0], color='red', zorder=5, s=100)
plt.scatter([0], [y_star], color='blue', zorder=5, s=100)

# Линии для двойной погрешности
plt.plot([0, 0], [y_star - Δk, y_star + Δk], color='blue', linestyle='--', linewidth=2)

# Центр тяжести
plt.scatter([xc], [yc], color='green', zorder=5, s=100)

# Добавление пунктирных линий для каждой экспериментальной точки с указанием X1, X2, X3...
for i, (x_val, y_val, err) in enumerate(zip(x_exp, y_exp_corrected, y_err)):
    # Пунктирные линии от каждой точки перпендикулярно к оси X
    if i != 1 and i != 3:  # Убираем линии для 2 и 4 точки
        plt.plot([x_val, x_val], [y_val, 0], linestyle=':', color='orange', linewidth=2)
    if i == 0:
        plt.text(x_val, -0.7, r'$X_1$', ha='center', color='orange', fontsize=14, weight='bold')
    if i == 4:
        plt.text(x_val, -0.7, r'$X_n$', ha='center', color='orange', fontsize=14, weight='bold')

# Убираем координаты зеленой точки с графика
# Добавление зеленых пунктирных линий от центра тяжести к осям
plt.plot([xc, xc], [yc, 0], linestyle=':', color='green', linewidth=2)  # к оси X
plt.plot([0, xc], [yc, yc], linestyle=':', color='green', linewidth=2)  # к оси Y

# Подписи для Xc и Yc
plt.text(xc, -0.7, r'$X_c$', color='green', ha='center', fontsize=14, weight='bold')  # на оси X
plt.text(-0.5, yc, r'$Y_c$', color='green', va='center', fontsize=14, weight='bold')  # на оси Y

# Добавление красной сплошной линии для 2deltaX
plt.plot([-4, -0.38], [0, 0], color='red', linewidth=2)  # Сплошная красная линия для 2deltaX

# Подпись 2deltaX по центру красной линии
plt.text(-2.19, 0.2, r'$2\Delta X$', color='red', ha='center', fontsize=14, weight='bold')

# Добавление синей сплошной линии от (0, 3) до (0, 0.65)
plt.plot([0, 0], [3, 0.65], color='blue', linewidth=2)

# Подпись для синей линии
plt.text(-0.3, 1.75, r'$y^*$', color='blue', ha='center', fontsize=14, weight='bold')

# Добавление подписи для 2deltaY
plt.text(0.4, 2.5, r'$2\Delta Y$', color='blue', ha='center', fontsize=14, weight='bold')

# Подпись для синей точки
plt.text(-1.5, -0.7, r'$x^*$', color='red', ha='center', fontsize=14, weight='bold')

# Настройки графика
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.xlim(-6, 8)
plt.ylim(-6, 10)
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Скрытие меток осей (делаем их прозрачными)
plt.tick_params(axis='both', which='both', labelcolor='white', length=0)

# Показываем график
plt.show()
