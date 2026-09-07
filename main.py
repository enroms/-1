# 1.1 Створення змінних різних типів та їх виведення
integer_var = 42
float_var = 3.1415
string_var = "Hello, Data Science!"
list_var = [1, 2, 3, "apple", True]
dict_var = {"name": "Alice", "age": 22, "city": "Kyiv"}

print("--- 1.1 Типи даних ---")
print(f"int: {integer_var}, float: {float_var}, str: {string_var}")
print(f"list: {list_var}, dict: {dict_var}\n")

# 1.2 Умовний оператор (перевірка парності)
number = 15
print("--- 1.2 Перевірка парності ---")
if number % 2 == 0:
    print(f"Число {number} є парним.")
else:
    print(f"Число {number} є непарним.\n")

# 1.3 Цикли for та while
print("--- 1.3 Цикли ---")
print("Цикл for по списку:")
for item in list_var:
    print(f"  Елемент: {item}")

print("Цикл while (лічильник до 3):")
count = 0
while count < 3:
    print(f"  Крок {count}")
    count += 1
print()

# 1.4 Функція з параметрами
print("--- 1.4 Функція ---")
def process_number(x, step=1):
    result = x + step
    print(f"Вхідне: {x}, збільшено на {step} -> Результат: {result}")
    return result

res = process_number(10, 5)

import numpy as np

print("--- 2.1 Масиви NumPy та їх розмірність ---")
arr_1d = np.array([1, 2, 3, 4, 5])
arr_2d = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)

print(f"1D масив: {arr_1d}, розмірність (shape): {arr_1d.shape}")
print(f"2D масив:\n{arr_2d}\nрозмірність (shape): {arr_2d.shape}\n")

print("--- 2.2 Базові арифметичні операції ---")
a = np.array([10, 20, 30])
b = np.array([2, 4, 5])
print(f"a = {a}, b = {b}")
print(f"Додавання (a + b): {a + b}")
print(f"Множення (a * b): {a * b}\n")

print("--- 2.3 Діапазон від 0 до 1 з кроком 0.1 ---")
# np.arange(start, stop, step)
range_arr = np.arange(0.0, 1.05, 0.1)
print(f"Масив [0; 1]: {np.round(range_arr, 2)}\n")

print("--- 2.4 Скалярний добуток двох одномірних масивів ---")
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
dot_product = np.dot(v1, v2)  # 1*4 + 2*5 + 3*6 = 32
print(f"v1: {v1}, v2: {v2}")
print(f"Скалярний добуток np.dot: {dot_product}")

import matplotlib.pyplot as plt
import numpy as np

# 3.1, 3.2, 3.3 Побудова графіків функцій
x = np.linspace(-10, 10, 400)
y1 = x ** 2
y2 = x ** 3

plt.figure(figsize=(10, 5))
plt.plot(x, y1, color='blue', linestyle='-', linewidth=2, label=r'$y = x^2$')
plt.plot(x, y2, color='red', linestyle='--', linewidth=2, label=r'$y = x^3$')

plt.title('Графіки функцій $y = x^2$ та $y = x^3$', fontsize=14)
plt.xlabel('Вісь X', fontsize=12)
plt.ylabel('Вісь Y', fontsize=12)
plt.ylim(-150, 150)  # для кращої видимості обох кривих
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='best')
plt.show()

# 3.4 Гістограма випадкових значень
random_data = np.random.randn(1000)

plt.figure(figsize=(8, 4))
plt.hist(random_data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Гістограма нормального розподілу (1000 значень)', fontsize=14)
plt.xlabel('Значення', fontsize=12)
plt.ylabel('Частота', fontsize=12)
plt.grid(axis='y', alpha=0.5)
plt.show()

import pandas as pd
from sklearn.datasets import load_iris

# 4.1 Завантаження датасету та створення DataFrame
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].map({i: name for i, name in enumerate(iris.target_names)})

# 4.2 Перші 5 записів
print("--- 4.2 Перші 5 записів DataFrame ---")
print(df.head(), "\n")

# 4.3 Описова статистика
print("--- 4.3 Статистична інформація (describe) ---")
print(df.describe(), "\n")

# 4.4 Кількість записів у кожному класі
print("--- 4.4 Кількість записів у кожному класі ---")
print(df['species'].value_counts())

import numpy as np
import matplotlib.pyplot as plt

# 5.1 Середні значення ознак через NumPy
features_only = df[iris.feature_names]
mean_values = np.mean(features_only.values, axis=0)

print("--- 5.1 Середні значення ознак (NumPy) ---")
for col, val in zip(iris.feature_names, mean_values):
    print(f"{col}: {val:.3f}")

# 5.2 Гістограми для кожного з атрибутів
plt.figure(figsize=(12, 8))
for i, col in enumerate(iris.feature_names, 1):
    plt.subplot(2, 2, i)
    plt.hist(df[col], bins=15, color='coral', edgecolor='black', alpha=0.7)
    plt.title(f'Розподіл {col}')
    plt.xlabel('Значення')
    plt.ylabel('Кількість')
plt.tight_layout()
plt.show()

# 5.3 Бокс-плоти (Boxplot) для ознак
plt.figure(figsize=(10, 6))
df.boxplot(column=iris.feature_names)
plt.title('Boxplot для числових атрибутів Iris Dataset')
plt.ylabel('Значення (см)')
plt.xticks(rotation=15)
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# 5.4 Порівняння розподілів атрибутів між класами
plt.figure(figsize=(12, 8))
for i, col in enumerate(iris.feature_names, 1):
    plt.subplot(2, 2, i)
    for sp in df['species'].unique():
        subset = df[df['species'] == sp]
        plt.hist(subset[col], bins=10, alpha=0.5, label=sp)
    plt.title(f'{col} за класами')
    plt.xlabel('Значення')
    plt.ylabel('Кількість')
    plt.legend()
plt.tight_layout()
plt.show()

# 5.5 Кореляційна матриця атрибутів
corr_matrix = features_only.corr()
print("\n--- 5.5 Кореляційна матриця ---")
print(corr_matrix)

