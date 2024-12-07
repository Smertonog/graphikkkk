#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Импортируем необходимые библиотеки
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 14
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r"""
    \usepackage[utf8]{inputenc}    % Кодировка UTF-8
    \usepackage[russian]{babel}   % Поддержка русского языка
"""

# Читаем данные из файла (например, CSV-файл)
# Предположим, что у нас есть два столбца: X и Y
file_name = 'data.csv'  # Имя файла с данными
data = pd.read_csv(file_name, delimiter=',')  # Чтение файла в DataFrame

# Разделяем данные на X и Y
X = data['X']  # Столбец X
Y = data['Y']  # Столбец Y

# Строим график исходных данных
plt.scatter(X, Y, color='blue', label='Исходные данные')

# Аппроксимация данных линейной функцией с помощью функции линейной регрессии
# Функция linregress вернет несколько параметров, включая угол наклона и пересечение
slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)

# Создаем линейную функцию для аппроксимации
def linear_function(x):
    return slope * x + intercept

# Строим аппроксимированную линию на основе линейной функции
plt.plot(X, linear_function(X), color='red', label=f'Аппроксимация: y={slope:.2f}x+{intercept:.2f}')

# Добавляем подписи и легенду
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Аппроксимация данных линейной функцией')
plt.legend()

# Показываем график
plt.show()

# Выводим характеристики графика
print(f'Угол наклона (slope): {slope:.2f}')
print(f'Пересечение с осью Y (intercept): {intercept:.2f}')
print(f'Коэффициент детерминации (R^2): {r_value**2:.2f}')


# In[ ]:


pd.read_csv('data.csv', delimiter=',')


# Рассмотрим текст кода, позволяющий реализовать картинку для текста про робототехнику

# In[20]:


# Параметры рычагов
l1, l2, l3 = 5, 5, 5  # длины рычагов
theta1, theta2, theta3 = np.radians([30, 45, 60])  # начальные углы в радианах

# Прямая кинематика
def forward_kinematics(theta1, theta2, theta3, l1, l2, l3):
    x = l1 * np.cos(theta1) + l2 * np.cos(theta2) + l3 * np.cos(theta3)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta2) + l3 * np.sin(theta3)
    return x, y

# Рассчитываем начальные координаты
x_init, y_init = forward_kinematics(theta1, theta2, theta3, l1, l2, l3)

# Определим конечные координаты
x_final, y_final = 8, 8  # конечные точки

# Время для перемещения
T = 10
t_values = np.linspace(0, T, 100)

# Линейная интерполяция по времени
x_traj = x_init + (x_final - x_init) * t_values / T
y_traj = y_init + (y_final - y_init) * t_values / T

# Визуализация траектории
plt.plot(x_traj, y_traj, label="Траектория")
plt.scatter([x_init, x_final], [y_init, y_final], color="red", label="Начало и конец")
plt.title("Траектория движения манипулятора")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.savefig("pic1.pdf", dpi=300)
plt.show()


# In[18]:


# Параметры длины звеньев манипулятора
l1, l2, l3 = 5, 5, 5  # длины рычагов

# Прямая кинематика: расчет координат x, y рабочего органа по углам сочленений
def forward_kinematics(theta1, theta2, theta3, l1, l2, l3):
    x = l1 * np.cos(theta1) + l2 * np.cos(theta2) + l3 * np.cos(theta3)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta2) + l3 * np.sin(theta3)
    return x, y

# Параметры начальной и конечной точки
x_start, y_start = 2, 2  # Точка А (начальная)
x_end, y_end = 8, 8      # Точка Б (конечная)

# Время и количество шагов для траектории
T = 10  # Общее время движения
n_steps = 100  # Количество шагов

# Создаем массив времени
t_values = np.linspace(0, T, n_steps)

# Определение нелинейной траектории: синусоида между двумя точками
# Сначала мы интерполируем по оси X и Y, но с добавлением синусоиды к Y
x_traj = np.linspace(x_start, x_end, n_steps)
amplitude = 2  # Амплитуда синусоиды
y_traj = np.linspace(y_start, y_end, n_steps) + amplitude * np.sin(np.linspace(0, 2 * np.pi, n_steps))

# Визуализация траектории
plt.figure(figsize=(8, 6))
plt.plot(x_traj, y_traj, label="Нелинейная траектория (синусоида)")
plt.scatter([x_start, x_end], [y_start, y_end], color="red", label="Точки А и Б")
plt.title("Траектория перемещения манипулятора")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()

# Функция для расчета углов обратной кинематики
# Простая обратная кинематика для демонстрации
# В реальных системах обратная кинематика может быть сложнее
def inverse_kinematics(x, y, l1, l2, l3):
    # Пример простой задачи обратной кинематики: расчет углов для простоты
    # В реальности обратная кинематика может требовать решения системы уравнений
    theta1 = np.arctan2(y, x)  # Пример простого угла для демонстрации
    # Реальные углы могут зависеть от конкретной конфигурации манипулятора
    return theta1, theta1, theta1  # Возвращаем одинаковые углы как пример

# Пример использования обратной кинематики для вычисления углов на каждом шаге траектории
angles_trajectory = []
for x, y in zip(x_traj, y_traj):
    theta1, theta2, theta3 = inverse_kinematics(x, y, l1, l2, l3)
    angles_trajectory.append([theta1, theta2, theta3])

# Вывод углов для каждого положения рабочего органа на траектории
for i, (x, y) in enumerate(zip(x_traj, y_traj)):
    theta1, theta2, theta3 = angles_trajectory[i]
    print(f"Точка: ({x:.2f}, {y:.2f}) -> Углы: θ1 = {np.degrees(theta1):.2f}°, θ2 = {np.degrees(theta2):.2f}°, θ3 = {np.degrees(theta3):.2f}°")


# In[11]:


# Настройки
true_value = 100  # Истинное значение физической величины (например, длина предмета в см)
num_measurements = 100  # Количество измерений (например, 100 замеров длины)
np.random.seed(0)  # Для повторяемости результатов

# Случайные погрешности: измерения распределены случайно вокруг истинного значения
random_error = np.random.normal(0, 5, num_measurements)  # Случайные погрешности с разбросом 5 см
measurements_random = true_value + random_error

# Систематическая погрешность: добавляем постоянную ошибку в 3 см
systematic_error = 3
measurements_systematic = true_value + random_error + systematic_error

# Комбинированная случайная и систематическая погрешность
measurements_combined = true_value + random_error + systematic_error

# Визуализация
plt.figure(figsize=(15, 6))

# График случайных погрешностей
plt.subplot(1, 3, 1)
plt.plot(range(num_measurements), measurements_random, 'o', color='skyblue', label="Измерения")
plt.axhline(true_value, color='red', linestyle='--', linewidth=2, label='Истинное значение')
plt.title("Случайные погрешности")
plt.xlabel("Номер измерения")
plt.ylabel("Измеренное значение (см)")
plt.legend()
plt.ylim(true_value - 20, true_value + 20)  # Ограничение на ось Y для лучшего сравнения

# График систематических погрешностей
plt.subplot(1, 3, 2)
plt.plot(range(num_measurements), measurements_systematic, 'o', color='lightgreen', label="Измерения")
plt.axhline(true_value, color='red', linestyle='--', linewidth=2, label='Истинное значение')
plt.title("Систематические погрешности")
plt.xlabel("Номер измерения")
plt.legend()
plt.ylim(true_value - 20, true_value + 20)

# График комбинированных погрешностей
plt.subplot(1, 3, 3)
plt.plot(range(num_measurements), measurements_combined, 'o', color='plum', label="Измерения")
plt.axhline(true_value, color='red', linestyle='--', linewidth=2, label='Истинное значение')
plt.title("Комбинированные погрешности")
plt.xlabel("Номер измерения")
plt.legend()
plt.ylim(true_value - 20, true_value + 20)

plt.suptitle("Влияние разных типов погрешностей на результаты измерений", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# In[16]:


# Настройки
true_value = 100  # Истинное значение физической величины (например, длина предмета в см)
num_measurements = 100  # Количество измерений
np.random.seed(0)  # Для повторяемости результатов

# Случайные погрешности
random_error = np.random.normal(0, 3, num_measurements)  # Случайные погрешности с разбросом 5 см
measurements_random = true_value + random_error

# Систематическая погрешность
systematic_error = 10  # Постоянная систематическая ошибка
measurements_systematic = true_value + random_error + systematic_error

# Комбинированная погрешность (увеличенный разброс случайных погрешностей)
combined_random_error = np.random.normal(2, 5, num_measurements)  # Разброс увеличен до 10 см
measurements_combined = true_value + combined_random_error + systematic_error

# Визуализация
plt.figure(figsize=(15, 6))

# График случайных погрешностей
plt.subplot(1, 3, 1)
plt.plot(range(num_measurements), measurements_random, 'o', color='skyblue', label="Измерения")
plt.axhline(true_value, color='red', linestyle='--', linewidth=2, label='Истинное значение')
plt.title("Случайные погрешности")
plt.xlabel("Номер измерения")
plt.ylabel("Измеренное значение (см)")
plt.legend()
plt.ylim(true_value - 30, true_value + 30)

# График систематических погрешностей
plt.subplot(1, 3, 2)
plt.plot(range(num_measurements), measurements_systematic, 'o', color='lightgreen', label="Измерения")
plt.axhline(true_value, color='red', linestyle='--', linewidth=2, label='Истинное значение')
plt.title("Систематические погрешности")
plt.xlabel("Номер измерения")
plt.legend()
plt.ylim(true_value - 30, true_value + 30)

# График комбинированных погрешностей
plt.subplot(1, 3, 3)
plt.plot(range(num_measurements), measurements_combined, 'o', color='plum', label="Измерения")
plt.axhline(true_value, color='red', linestyle='--', linewidth=2, label='Истинное значение')
plt.title("Комбинированные погрешности")
plt.xlabel("Номер измерения")
plt.legend()
plt.ylim(true_value - 30, true_value + 30)

plt.suptitle("Влияние разных типов погрешностей на результаты измерений", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# In[35]:


# Настройки
true_value = 100  # Истинное значение физической величины (например, длина предмета в см)
num_measurements = 10  # Количество измерений
np.random.seed(0)  # Для повторяемости результатов

# Случайные погрешности
random_error = np.random.normal(0, 3, num_measurements)  # Случайные погрешности с разбросом 5 см
measurements_random = true_value + random_error

# Систематическая погрешность
systematic_error = 10  # Постоянная систематическая ошибка
measurements_systematic = true_value + random_error + systematic_error

# Комбинированная погрешность (увеличенный разброс случайных погрешностей)
combined_random_error = np.random.normal(2, 5, num_measurements)  # Разброс увеличен до 10 см
measurements_combined = true_value + combined_random_error + systematic_error

# Визуализация
plt.figure(figsize=(15, 6))

# График случайных погрешностей
plt.subplot(1, 3, 1)
plt.plot(range(num_measurements), measurements_random, 'o', color='skyblue', label="Измерения")
plt.axhline(true_value, color='red', linestyle='--', linewidth=1, label='Истинное значение')
plt.title("Случайные погрешности")
plt.xlabel("Номер измерения")
plt.ylabel("Измеренное значение (см)")
plt.legend()
plt.ylim(true_value - 30, true_value + 30)

# График систематических погрешностей
plt.subplot(1, 3, 2)
plt.plot(range(num_measurements), measurements_systematic, 'o', color='lightgreen', label="Измерения")
plt.axhline(true_value, color='red', linestyle='--', linewidth=1, label='Истинное значение')
plt.title("Систематические погрешности")
plt.xlabel("Номер измерения")
plt.legend()
plt.ylim(true_value - 30, true_value + 30)

# График комбинированных погрешностей
plt.subplot(1, 3, 3)
plt.plot(range(num_measurements), measurements_combined, 'o', color='plum', label="Измерения")
plt.axhline(true_value, color='red', linestyle='--', linewidth=1, label='Истинное значение')
plt.title("Комбиинрованные погрешности")
plt.xlabel("Номер измерения")
plt.legend()
plt.ylim(true_value - 30, true_value + 30)

plt.suptitle("Влияние разных типов погрешностей на результаты измерений")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("erros0.pdf")
plt.show()


# In[34]:


from scipy.stats import norm

# Настройки
mean = 0  # Среднее значение для всех распределений
std_devs = [1, 2, 3, 4]  # Разные значения стандартного отклонения
x = np.linspace(-10, 10, 1000)  # Диапазон значений для оси X

# Визуализация
plt.figure(figsize=(10, 6))

for std_dev in std_devs:
    y = norm.pdf(x, mean, std_dev)  # Вычисляем плотность вероятности для данного sigma
    plt.plot(x, y, label=fr'$\sigma = {std_dev}$')

plt.title("Распределение Гаусса для разных стандартных отклонений")
plt.xlabel("Значение")
plt.ylabel("Плотность вероятности")
plt.legend(title="Стандартное отклонение")
plt.grid(True)
plt.savefig("gauss.pdf")
plt.show()


# In[52]:


from scipy.stats import norm

# Параметры нормального распределения
mean = 0  # Среднее значение
std_dev = 1  # Стандартное отклонение

# Диапазон значений для оси X
x = np.linspace(-4, 4, 1000)
# Вычисление плотности вероятности для каждого x
y = norm.pdf(x, mean, std_dev)

# Определение границ доверительного интервала (±1 стандартное отклонение)
interval_min = mean - 2 * std_dev
interval_max = mean + 2 * std_dev

# Визуализация
plt.figure(figsize=(10, 6))
plt.plot(x, y, label=r'Распределение Гаусса ($\sigma=1$)', color='blue')
plt.fill_between(x, y, where=((x >= interval_min) & (x <= interval_max)), 
                color='skyblue', alpha=0.5, label='Доверительный интервал 95\% ($\pm 2\sigma$)')

# Добавление линий для границ доверительного интервала
plt.axvline(interval_min, color='red', linestyle='--', label=fr'$-1\sigma\ ({interval_min})$')
plt.axvline(interval_max, color='red', linestyle='--', label=fr'$+1\sigma\ ({interval_max})$')

# Настройки графика
plt.title("Распределение Гаусса с выделенным доверительным интервалом (95\%)")
plt.xlabel("Значение")
plt.ylabel("Плотность вероятности")
plt.legend(fontsize=11)
plt.grid(True)
plt.savefig("gauss95.pdf")
plt.show()


# In[ ]:





# In[40]:


from scipy.stats import norm, t

# Диапазон значений для оси X
x = np.linspace(-5, 5, 1000)

# Параметры распределений
mean = 0  # Среднее значение для нормального распределения
std_dev = 1  # Стандартное отклонение для нормального распределения

# Расчет плотности вероятности
y_gauss = norm.pdf(x, mean, std_dev)  # Нормальное распределение
y_student_10 = t.pdf(x, df=10)  # Распределение Стьюдента с 10 степенями свободы
y_student_1 = t.pdf(x, df=1)  # Распределение Стьюдента с 1 степенью свободы

# Визуализация
plt.figure(figsize=(10, 6))

# Построение кривых
plt.plot(x, y_gauss, label=r"Распределение Гаусса ($\sigma=1$)", color="blue")
plt.plot(x, y_student_10, label=r"Распределение Стьюдента ($N=10$)", color="green")
plt.plot(x, y_student_1, label=r"Распределение Стьюдента ($N=1$)", color="red", linestyle='--', linewidth=1)

# Настройки графика
plt.title("Распределение Гаусса и распределения Стьюдента с разным числом измерений")
plt.xlabel("Значение")
plt.ylabel("Плотность вероятности")
plt.legend(loc='upper right', fontsize='small')
plt.grid(True)
plt.savefig("gauss_student.pdf")
plt.show()


# In[89]:


# Создаем произвольную возрастающую функцию z(x)
x = np.linspace(0.1, 4, 100)
z = np.log(x**(2*x)+1)  # Возрастающая функция

# Параметры для средней точки и интервала погрешности
mean_x = 2  # Среднее значение на оси X
error_x = 0.2  # Погрешность на оси X (±10% от среднего значения)
mean_z = np.log(mean_x**(2*mean_x)+1)  # Соответствующее значение на оси Z
error_z = (((2*mean_x^(2*mean_x))*(1 + np.log(mean_x)))/((1 + mean_x^(2*mean_x))*np.log(10))) * error_x  # Погрешность на оси Z (производная функции z(x) = x^2)

# Визуализация функции
plt.figure(figsize=(10, 6))
plt.plot(x, z, label="Функция z(x)", color="blue")

# Отметка среднего значения
plt.plot(mean_x, mean_z, 'o', color='red', label="Среднее значение")

# Отметка интервалов погрешности пунктирными линиями
plt.axvline(mean_x - error_x, color='gray', linestyle='--')
plt.axvline(mean_x + error_x, color='gray', linestyle='--')
plt.axhline(mean_z - error_z, color='gray', linestyle='--')
plt.axhline(mean_z + error_z, color='gray', linestyle='--')
plt.axvline(mean_x, color='red', linestyle='-')
plt.axhline(mean_z, color='red', linestyle='-')

# Настройки графика
plt.xlabel("X")
plt.ylabel("Z")
plt.xticks([])
plt.yticks([])
plt.legend()
plt.show()


# In[69]:


# Настройки
true_value = 100  # Истинное значение физической величины (например, длина предмета в см)
num_measurements = 10  # Количество измерений
np.random.seed(0)  # Для повторяемости результатов

# Случайные погрешности
random_error = np.random.normal(0, 3, num_measurements)  # Случайные погрешности с разбросом 5 см
measurements_random = true_value + random_error

# Систематическая погрешность
systematic_error = 10  # Постоянная систематическая ошибка
measurements_systematic = true_value + random_error + systematic_error

# Комбинированная погрешность (увеличенный разброс случайных погрешностей)
combined_random_error = np.random.normal(2, 5, num_measurements)  # Разброс увеличен до 10 см
measurements_combined = true_value + combined_random_error + systematic_error

# Визуализация
plt.figure(figsize=(15, 6))

# График случайных погрешностей
plt.subplot(1, 3, 1)
plt.plot(range(num_measurements), measurements_random, 'o', color='skyblue', label="Измерения")
plt.axhline(true_value, color='red', linestyle='--', linewidth=1, label='Истинное значение')
plt.title("Г")
plt.xlabel("Номер измерения")
plt.ylabel("Измеренное значение (см)")
plt.legend()
plt.ylim(true_value - 30, true_value + 30)

# График систематических погрешностей
plt.subplot(1, 3, 2)
plt.plot(range(num_measurements), measurements_systematic, 'o', color='lightgreen', label="Измерения")
plt.axhline(true_value, color='red', linestyle='--', linewidth=1, label='Истинное значение')
plt.title("Д")
plt.xlabel("Номер измерения")
plt.legend()
plt.ylim(true_value - 30, true_value + 30)

# График комбинированных погрешностей
plt.subplot(1, 3, 3)
plt.plot(range(num_measurements), measurements_combined, 'o', color='plum', label="Измерения")
plt.axhline(true_value, color='red', linestyle='--', linewidth=1, label='Истинное значение')
plt.title("Е")
plt.xlabel("Номер измерения")
plt.legend()
plt.ylim(true_value - 30, true_value + 30)

plt.suptitle("Влияние разных типов погрешностей на результаты измерений", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# In[216]:


# Установка шрифта на Times New Roman с размером 16pt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
# Определяем более интересную функцию z(x)
x = np.linspace(0, 10, 100)
z = np.exp(0.3 * x) * np.sin(x)  # Функция z(x) = e^(0.3x) * sin(x)

# Параметры для средней точки и интервала погрешности
mean_x = 5  # Среднее значение на оси X
error_x = 0.7  # Погрешность на оси X (±10% от среднего значения)
mean_z = np.exp(0.3 * mean_x) * np.sin(mean_x)  # Соответствующее значение на оси Z
# Погрешность на оси Z (рассчитанная как производная функции z(x) в точке mean_x)
error_z = np.abs((0.3 * np.exp(0.3 * mean_x) * np.sin(mean_x) + np.exp(0.3 * mean_x) * np.cos(mean_x)) * error_x)

# Визуализация функции
plt.figure(figsize=(10, 6))
plt.plot(x, z, label=r"Функция $z(x) = e^{0.3x} \cdot \sin(x)$", color="blue")

# Отметка среднего значения
plt.plot(mean_x, mean_z, 'o', color='red', label="Среднее значение")
plt.text(mean_x, mean_z + 1, '$X_{ср}$', color='red', ha='center', fontsize=12)
plt.text(mean_x + 0.5, mean_z, '$Z_{ср}$', color='red', va='center', fontsize=12)

# Отметка интервалов погрешности пунктирными линиями
plt.axvline(mean_x - error_x, color='gray', linestyle='--', label="Интервал погрешности по X")
plt.axvline(mean_x + error_x, color='gray', linestyle='--')
plt.axhline(mean_z - error_z, color='purple', linestyle='--', label="Интервал погрешности по Z")
plt.axhline(mean_z + error_z, color='purple', linestyle='--')

# Настройки графика
plt.title("Функция $z(x) = e^{0.3x} \cdot \sin(x)$ с интервалом погрешности")
plt.xlabel("X")
plt.ylabel("Z")
plt.legend()
plt.grid(True)

# Убираем метки на осях
plt.xticks([])
plt.yticks([])

plt.show()


# In[217]:


# Установка шрифта на Times New Roman с размером 16pt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
# Определяем функцию z(x)
x = np.linspace(2, 10, 100)
z = x**3 - 5*x**2 + 4*x  # Функция z(x) = x^3 - 5x^2 + 4x

# Параметры для средней точки и интервала погрешности
mean_x = 7  # Среднее значение на оси X
error_x = 0.5  # Погрешность на оси X (±10% от среднего значения)
mean_z = mean_x**3 - 5*mean_x**2 + 4*mean_x  # Соответствующее значение на оси Z

# Погрешность на оси Z, рассчитанная как производная функции z(x) в точке mean_x
# Производная z'(x) = 3x^2 - 10x + 4
error_z = np.abs((3 * mean_x**2 - 10 * mean_x + 4) * error_x)

# Визуализация функции
plt.figure(figsize=(10, 6))
plt.plot(x, z, label=r"Функция $z(x) = x^3 - 5x^2 + 4x$", color="blue")

# Отметка среднего значения
plt.plot(mean_x, mean_z, 'o', color='red', label="Среднее значение")


# Отметка интервалов погрешности пунктирными линиями
plt.axvline(mean_x - error_x, color='gray', linestyle='--', label="Интервал погрешности по X")
plt.axvline(mean_x + error_x, color='gray', linestyle='--')
plt.axhline(mean_z - error_z, color='purple', linestyle='--', label="Интервал погрешности по Z")
plt.axhline(mean_z + error_z, color='purple', linestyle='--')

# Настройки графика
plt.xlabel("X", loc='right')
plt.ylabel("Z(x)", loc='top')
plt.grid(True)

# Убираем метки на осях
plt.xticks([])
plt.yticks([])

plt.show()


# In[125]:


# Функция f(x, l)
def f(x, l):
    return (2 * x) / (4 * x - l)

# Определяем диапазоны для x и l только для положительных значений
x_const = 1  # Постоянное значение x для первого случая
l_values = np.linspace(0.1, 6, 20)  # Диапазон l от 0.1 до 4x - 0.1
l_const = 5  # Постоянное значение l для второго случая
x_values = np.linspace(0.1, 6, 20)  # Диапазон x от 0.1 до l/4 - 0.1

# Вычисляем значения функции для первого случая (x - постоянное)
f_values_l = f(x_const, l_values)

# Вычисляем значения функции для второго случая (l - постоянное)
f_values_x = f(x_values, l_const)

# Создаем DataFrame для хранения таблицы значений
data = {
    "l (при x=3)": l_values,
    "f(x, l) (при x=3)": f_values_l,
    "x (при l=5)": x_values,
    "f(x, l) (при l=5)": f_values_x
}

df = pd.DataFrame(data)

# Сохраняем таблицу в файл CSV
df.to_csv("function_values2.csv", index=False, encoding="utf-8-sig")

print("Таблица значений функции успешно сохранена в файл 'function_values2.csv'")


# In[213]:


from scipy.stats import linregress
# Установка шрифта на Times New Roman с размером 16pt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
# Генерация данных с линейной зависимостью и добавлением случайного разброса
np.random.seed(42)  # Для воспроизводимости
x = np.linspace(0, 10, 10)
y_true = 2.5 * x + 3  # Истинная линейная зависимость y = 2.5x + 3

# Фиксированные погрешности
error_x = 0.5  # Погрешность по X для каждой точки
error_y = 1.5  # Погрешность по Y для каждой точки

error_x = np.random.normal(0, 0.3, size=x.size)  # Случайные погрешности по x
error_y = np.random.normal(0, 2, size=x.size)    # Случайные погрешности по y
x_observed = x + error_x
y_observed = y_true + error_y

# Линейная регрессия для наблюдаемых данных
slope, intercept, r_value, p_value, std_err = linregress(x_observed, y_observed)
y_fit = slope * x_observed + intercept  # Линия наилучшего соответствия

# Построение графика
plt.figure(figsize=(10, 6))
plt.errorbar(x_observed, y_observed, xerr=np.abs(error_x), yerr=np.abs(error_y),
            fmt='o', color='blue', ecolor='gray', elinewidth=1, capsize=3,
            label="Наблюдаемые данные с погрешностями")

# Отображение линии наилучшего соответствия
plt.plot(x_observed, y_fit, color='green', label=f"Линия наилучшего соответствия: y = {slope:.2f}x + {intercept:.2f}")
# Убираем метки на осях
plt.xticks([])
plt.yticks([])
# Настройки графика
plt.xlabel("X")
plt.ylabel("Y")
plt.show()




# In[212]:


from scipy.stats import linregress
# Установка шрифта на Times New Roman с размером 16pt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
# Генерация данных с линейной зависимостью и добавлением случайного разброса
np.random.seed(42)  # Для воспроизводимости
x = np.linspace(1, 10, 20)  # Исходные значения x
y_true = 2.5 * x + 3  # Истинная линейная зависимость y = 2.5x + 3
error_x = np.random.normal(0, 0.3, size=x.size)  # Случайные погрешности по x
error_y = np.random.normal(0, 2, size=x.size)    # Случайные погрешности по y
x_observed = x + error_x
y_observed = y_true + error_y

# Линейная регрессия для наблюдаемых данных
slope, intercept, r_value, p_value, std_err = linregress(x_observed, y_observed)

# Параметры линии наилучшего соответствия
x_fit = np.linspace(min(x_observed), max(x_observed), 100)  # Основной диапазон
y_fit = slope * x_fit + intercept  # Линия наилучшего соответствия

# Экстраполяция до оси Y
x_extrap = np.linspace(0, min(x_observed), 100)  # Диапазон для экстраполяции до оси Y
y_extrap = slope * x_extrap + intercept

# Построение графика
plt.figure(figsize=(10, 6))

# График с наблюдаемыми данными и погрешностями
plt.errorbar(x_observed, y_observed, xerr=np.abs(error_x), yerr=np.abs(error_y),
            fmt='o', color='blue', ecolor='gray', elinewidth=1, capsize=3,
            label="Наблюдаемые данные с погрешностями")

# Линия наилучшего соответствия
plt.plot(x_fit, y_fit, color='green', linestyle='-', label="Линия наилучшего соответствия")

# Экстраполяция до оси Y
plt.plot(x_extrap, y_extrap, color='green', linestyle='--', label="Экстраполяция до оси Y")
# Убираем метки на осях
plt.yticks([])
# Настройки графика
plt.xlabel("X")
plt.ylabel("Y")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

plt.show()


# In[211]:


from scipy.stats import linregress
# Установка шрифта на Times New Roman с размером 16pt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
# Генерация данных с линейной зависимостью и добавлением случайного разброса
np.random.seed(42)  # Для воспроизводимости
x = np.linspace(1, 10, 20)  # Исходные значения x
y_true = 2.5 * x + 3  # Истинная линейная зависимость y = 2.5x + 3
error_x = np.random.normal(0, 0.3, size=x.size)  # Случайные погрешности по x
error_y = np.random.normal(0, 2, size=x.size)    # Случайные погрешности по y
x_observed = x + error_x
y_observed = y_true + error_y

# Линейная регрессия для наблюдаемых данных
slope, intercept, r_value, p_value, std_err = linregress(x_observed, y_observed)

# Параметры линии наилучшего соответствия
x_fit = np.linspace(min(x_observed), max(x_observed), 100)  # Основной диапазон
y_fit = slope * x_fit + intercept  # Линия наилучшего соответствия

# Экстраполяция до оси Y
x_extrap = np.linspace(0, min(x_observed), 100)  # Диапазон для экстраполяции до оси Y
y_extrap = slope * x_extrap + intercept

# Выбор двух произвольных точек на линии наилучшего соответствия
x_point1, x_point2 = 2, 8
y_point1 = slope * x_point1 + intercept
y_point2 = slope * x_point2 + intercept

# Построение графика
plt.figure(figsize=(10, 6))

# График с наблюдаемыми данными и погрешностями
plt.errorbar(x_observed, y_observed, xerr=np.abs(error_x), yerr=np.abs(error_y),
            fmt='o', color='blue', ecolor='gray', elinewidth=1, capsize=3,
            label="Наблюдаемые данные с погрешностями")

# Линия наилучшего соответствия
plt.plot(x_fit, y_fit, color='green', linestyle='-', label="Линия наилучшего соответствия")

# Экстраполяция до оси Y
plt.plot(x_extrap, y_extrap, color='green', linestyle='--', label="Экстраполяция до оси Y")

# Отображение двух произвольных точек на линии наилучшего соответствия
plt.scatter([x_point1, x_point2], [y_point1, y_point2], color='purple', zorder=5)

# Пунктирные линии для обозначения точек
plt.vlines([x_point1, x_point2], ymin=0, ymax=[y_point1, y_point2], color='purple', linestyles='dotted')
plt.hlines([y_point1, y_point2], xmin=0, xmax=[x_point1, x_point2], color='purple', linestyles='dotted')

# Настройки графика
plt.xlabel("X")
plt.ylabel("Y")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True)

plt.show()


# In[210]:


# Установка шрифта на Times New Roman с размером 16pt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14

# Генерация данных
x = np.linspace(1, 10, 100)
y = np.exp(x)  # Экспоненциальная функция

# Построение полулогарифмического графика
plt.figure(figsize=(10, 6))
plt.semilogy(x, y, label='y = exp(x)', color='blue')  # Полулогарифмический масштаб по оси Y

# Добавление дополнительных данных
y2 = 10 * x  # Линейная зависимость
plt.plot(x, y2, label='y = 10x', color='orange')

# Настройки графика
plt.title("Полулогарифмический график")
plt.xlabel("X")
plt.ylabel("Y (логарифмический масштаб)")
plt.legend()
plt.grid(True)
plt.axhline(1, color='gray', linewidth=0.5, linestyle='--')  # Линия для ориентира
plt.ylim(0.1, 30000)  # Ограничение по оси Y для лучшего восприятия

plt.show()


# In[177]:


# Генерация данных
x = np.linspace(1, 10, 100)
y_exp = np.exp(x)  # Экспоненциальная функция
y_linear = 10 * x  # Линейная зависимость

# Создание фигуры и осей
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Установка шрифта на Times New Roman с размером 16pt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14

# Полулогарифмический график
ax1.semilogy(x, y_exp, label='y = exp(x)', color='blue')  # Полулогарифмический масштаб по оси Y
ax1.plot(x, y_linear, label='y = 10x', color='green')

# Настройки полулогарифмического графика
ax1.set_title("Полулогарифмический масштаб")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.legend()
ax1.grid(True)
ax1.axhline(1, color='gray', linewidth=0.5, linestyle='--')  # Линия для ориентира
ax1.set_ylim(0.1, 30000)  # Ограничение по оси Y для лучшего восприятия

# Обычный график
ax2.plot(x, y_exp, label='y = exp(x)', color='blue')  # Экспоненциальная функция
ax2.plot(x, y_linear, label='y = 10x', color='green')

# Настройки обычного графика
ax2.set_title("Обычный масштаб")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.legend()
ax2.grid(True)

# Показ графиков
plt.tight_layout()  # Автоматическая настройка для лучшего расположения графиков
plt.show()


# In[209]:


# Задаем диапазон значений x
x = np.linspace(-2, 2, 500)
x0, y0 = 0, 0  # центральная точка (x0, y0)
dx, dy = 0.4, 0.4  # отклонения по x и y

# Установка шрифта на Times New Roman с размером 16pt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14

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


# In[197]:


from scipy.interpolate import interp1d

# Задание экспериментальных данных в диапазоне, где нет асимптот (например, -π/4 < x < π/4)
x_data = np.linspace(-np.pi/4, np.pi/4, 10)
y_data = np.tan(2 * x_data) + 0.1 * np.random.randn(10)  # tan(2x) с небольшим шумом

# Интерполяция данных (сглаживание)
interpolator = interp1d(x_data, y_data, kind='cubic')  # кубическая интерполяция
x_interp = np.linspace(-np.pi/4, np.pi/4, 100)
y_interp = interpolator(x_interp)

# Вычисление погрешности интерполяции (расстояния от экспериментальных точек до кривой)
y_interp_data = interpolator(x_data)  # значения интерполяции в точках x_data
errors = y_data - y_interp_data  # погрешности

# Построение графика
plt.figure(figsize=(10, 6))

# Экспериментальные данные
plt.plot(x_data, y_data, 'o', label='Experimental data', color='blue')

# Интерполяционная кривая
plt.plot(x_interp, y_interp, '-', label='Interpolation curve', color='green')

# Линии погрешностей
for (xi, yi, ei) in zip(x_data, y_data, errors):
    plt.plot([xi, xi], [yi, yi - ei], 'r--', linewidth=1)

# Подписи погрешностей
for (xi, yi, ei) in zip(x_data, y_data, errors):
    plt.text(xi, yi - ei / 2, f'{ei:.2f}', color='red', ha='right')

# Оформление графика
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.title(r'Error in Graphical Interpolation for $y = \tan(2x)$', fontsize=14)
plt.legend()
plt.grid(True)

# Показ графика
plt.show()


# In[205]:


# Устанавливаем шрифт Times New Roman
plt.rcParams["font.family"] = "Times New Roman"

# Параметры центральной точки
xM, yM = 0, 1  # вершина параболы
dx, dy = 0.5, 0.3  # отклонения по x и y

# Генерируем данные для основной параболы и границ отклонений
x = np.linspace(xM - 1.5 * dx, xM + 1.5 * dx, 500)
y = -4 * (x - xM) ** 2 + yM  # основная парабола
y_upper = y + dy  # верхняя граница отклонений
y_lower = y - dy  # нижняя граница отклонений

# Построение графика
plt.figure(figsize=(8, 6))

# Основная кривая
plt.plot(x, y, 'k-', linewidth=2, label='Main curve')

# Пунктирные кривые для границ отклонений
plt.plot(x, y_upper, 'k--', linewidth=1)
plt.plot(x, y_lower, 'k--', linewidth=1)

# Горизонтальные и вертикальные пунктирные линии
plt.axhline(y=yM, color='k', linestyle='--', linewidth=0.8)
plt.axhline(y=yM + dy, color='k', linestyle='--', linewidth=0.8)
plt.axhline(y=yM - dy, color='k', linestyle='--', linewidth=0.8)
plt.axvline(x=xM, color='k', linestyle='--', linewidth=0.8)
plt.axvline(x=xM + dx, color='k', linestyle='--', linewidth=0.8)
plt.axvline(x=xM - dx, color='k', linestyle='--', linewidth=0.8)

# Точки с погрешностью
error_points_x = [xM - dx / 2, xM + dx / 2]
error_points_y = -4 * (np.array(error_points_x) - xM) ** 2 + yM  # значения функции в этих точках
errors = [0.15, 0.2]  # значения погрешностей для каждой точки

for xi, yi, ei in zip(error_points_x, error_points_y, errors):
    plt.plot([xi, xi], [yi - ei, yi + ei], 'gray', linewidth=1)  # линия погрешности
    plt.plot(xi, yi, 'D', color='gray', markersize=6)  # ромбовидная метка

# Точки центральная и крайние
plt.plot(xM, yM, 'wo', markeredgecolor='k')  # центральная точка
plt.plot(xM - dx, -4 * (xM - dx - xM) ** 2 + yM, 'ko')  # левая крайняя точка
plt.plot(xM + dx, -4 * (xM + dx - xM) ** 2 + yM, 'ko')  # правая крайняя точка

# Настройки осей и меток
plt.xlim(xM - 1.5 * dx, xM + 1.5 * dx)
plt.ylim(yM - 1.5 * dy, yM + 1.5 * dy)
plt.xlabel(r'$x$', fontsize=14)
plt.ylabel(r'$y$', fontsize=14)
plt.xticks([xM - dx, xM, xM + dx], [r'$x_M - \Delta x$', r'$x_M$', r'$x_M + \Delta x$'], fontsize=12)
plt.yticks([yM - dy, yM, yM + dy], [r'$y_M - \Delta y$', r'$y_M$', r'$y_M + \Delta y$'], fontsize=12)

# Показ графика
plt.grid(False)  # Отключаем сетку для чистоты изображения
plt.show()


# In[ ]:




