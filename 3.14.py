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

# Расчет верхней и нижней прямых
b_upper = 3.71 - (k + Δk) * 3  # Свободный член для верхней прямой
b_lower = 3.71 - (k - Δk) * 3  # Свободный член для нижней прямой

upper_line = (k + Δk) * x + b_upper  # Верхняя вспомогательная прямая
lower_line = (k - Δk) * x + b_lower  # Нижняя вспомогательная прямая

# Параметры для серой убывающей линии
k_gray = -0.3  # Наклон (отрицательный)
b_gray = -0.5  # Пересечение с осью y
dx = 3         # Сдвиг влево на 3 единицы

# Продлеваем диапазон для серой линии
x_gray = np.linspace(-7, 4.6, 500)  # Диапазон расширен вверх и вниз
gray_line = k_gray * (x_gray + dx) + b_gray  # Уравнение серой линии с сдвигом по x

# Сдвигаем все элементы графика на 1.5 единицы вверх
shift_y = 1.5

# Сдвигаем значения для линий
y_exp_corrected += shift_y
upper_line += shift_y
lower_line += shift_y
gray_line += shift_y  # Сдвиг серой линии

# Задаём координаты синей точки вручную
x_star = -1.15
y_star = 0.46

# Построение графика
plt.figure(figsize=(12, 8))

# Экспериментальные точки с погрешностями
plt.errorbar(x_exp, y_exp_corrected, yerr=y_err, fmt='o', color='black', capsize=5, markersize=8)

# Основная прямая
plt.plot(x, k * x + b + shift_y, color='black', linewidth=2)

# Вспомогательные верхняя и нижняя прямые
plt.plot(x, upper_line, linestyle='--', color='gray', linewidth=2)
plt.plot(x, lower_line, linestyle='--', color='gray', linewidth=2)

# Добавляем серую убывающую линию
plt.plot(x_gray, gray_line, color='gray', linewidth=2)

# Добавляем точку пересечения x*, y* (синяя точка)
plt.scatter([x_star], [y_star], color='blue', zorder=5, s=100)

# Добавляем пунктирные синие линии от точки пересечения до осей
plt.plot([x_star, x_star], [y_star, -1], linestyle=':', color='blue', linewidth=2)  # До оси x
plt.plot([x_star, -6], [y_star, y_star], linestyle=':', color='blue', linewidth=2)  # До оси y

# Центр тяжести
xc = np.mean(x_exp)
yc = np.mean(y_exp_corrected)
plt.scatter([xc], [yc], color='green', zorder=5, s=100)

# Пунктирные линии для каждой экспериментальной точки (оранжевые)
for i, (x_val, y_val, err) in enumerate(zip(x_exp, y_exp_corrected, y_err)):
    if i not in [1, 3]:  # Исключаем 2-ю и 4-ю точки
        plt.plot([x_val, x_val], [y_val, -1], linestyle=':', color='orange', linewidth=2)
        # Добавляем подписи для первой и пятой линий
        if i == 0:  # Первая линия
            plt.text(x_val, -1.3, r'$X_1$', color='orange', fontsize=14, ha='center', va='top', weight='bold')
        if i == 4:  # Пятая линия
            plt.text(x_val, -1.3, r'$X_n$', color='orange', fontsize=14, ha='center', va='top', weight='bold')


# Зеленые пунктирные линии
plt.plot([xc, xc], [yc, -1], linestyle=':', color='green', linewidth=2)
plt.plot([-6, xc], [yc, yc], linestyle=':', color='green', linewidth=2)

# Подписи для Xc и Yc
plt.text(xc, -1.7, r'$X_c$', color='green', ha='center', fontsize=14, weight='bold')  # на оси X
plt.text(-6.5, yc, r'$Y_c$', color='green', va='center', fontsize=14, weight='bold')  # на оси Y

# Подписи для x* и y*
plt.text(x_star, -1.7, r'$x^*$', color='blue', ha='center', fontsize=14, weight='bold')  # на оси X
plt.text(-6.5, y_star, r'$y^*$', color='blue', va='center', fontsize=14, weight='bold')  # на оси Y

# Координаты красных точек и концов линий
red_point_1 = (-2.75, 0.94)
red_line_1_end = (-4, 0.94)

red_point_2 = (-0.26, 0.2)
red_line_2_end = (-4, 0.2)

# Добавление красных точек
plt.scatter(*red_point_1, color='red', zorder=5, s=20)  # Первая красная точка
plt.scatter(*red_point_2, color='red', zorder=5, s=20)  # Вторая красная точка

# Добавление красных линий
plt.plot([red_point_1[0], red_line_1_end[0]], [red_point_1[1], red_line_1_end[1]], color='red', linewidth=1, linestyle='--')  # Линия от первой точки
plt.plot([red_point_2[0], red_line_2_end[0]], [red_point_2[1], red_line_2_end[1]], color='red', linewidth=1, linestyle='--')  # Линия от второй точки

# Подпись между красными линиями
label_x = -4.2  # Позиция подписи по x
label_y = (red_line_1_end[1] + red_line_2_end[1]) / 2  # Среднее значение по y
plt.text(label_x + 0.1, label_y + 0.1, r'$2\Delta Y$', color='red', fontsize=12, va='center', ha='right', weight='bold')


# Координаты для вертикальной линии
x_vertical = -4
y_start = 0.2
y_end = 0.94

# Добавление вертикальной линии со стрелками
plt.plot([x_vertical, x_vertical], [y_start, y_end], color='red', linewidth=1, linestyle='-')

# Добавление стрелочек в начале и конце линии
arrow_props = dict(arrowstyle='-|>', color='red', linewidth=1)

# Верхняя стрелка
plt.annotate('', xy=(x_vertical, y_end), xytext=(x_vertical, y_end - 0.05), arrowprops=arrow_props)
# Нижняя стрелка
plt.annotate('', xy=(x_vertical, y_start), xytext=(x_vertical, y_start + 0.05), arrowprops=arrow_props)

# Координаты для первой вертикальной линии
x1 = -2.74
y1_start = 0.94
y1_end = -0.5

# Координаты для второй вертикальной линии
x2 = -0.26
y2_start = 0.2
y2_end = -0.5

# Добавление вертикальных красных линий
plt.plot([x1, x1], [y1_start, y1_end], color='red', linewidth=1, linestyle='--')  # Первая линия
plt.plot([x2, x2], [y2_start, y2_end], color='red', linewidth=1, linestyle='--')  # Вторая линия

# Горизонтальная линия со стрелкой между концами вертикальных линий
horizontal_line_y = -0.5
plt.plot([x1, x2], [horizontal_line_y, horizontal_line_y], color='red', linewidth=1)  # Горизонтальная линия

# Добавление стрелок на горизонтальной линии
arrow_props = dict(arrowstyle='<|-|>', color='red', linewidth=1)
plt.annotate('', xy=(x1, horizontal_line_y), xytext=(x2, horizontal_line_y), arrowprops=arrow_props)

# Подпись 2ΔX
label_x = (x1 + x2) / 2  # Средняя координата по x
label_y = horizontal_line_y - 0.1  # Позиция подписи немного ниже линии
plt.text(label_x, label_y, r'$2\Delta X$', color='red', fontsize=12, ha='center', va='top', weight='bold')

# Настройки графика
plt.axhline(-1, color='black', linewidth=1)  # Сместить ось X вверх
plt.axvline(-6, color='black', linewidth=1)  # Сместить ось Y вправо
plt.xlim(-8, 7)
plt.ylim(-4 + shift_y, 8 + shift_y)
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Скрытие меток осей (делаем их прозрачными)
plt.tick_params(axis='both', which='both', labelcolor='white', length=0)

# Показываем график
plt.show()
