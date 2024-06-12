import numpy as np

#1.Знайти в датасеті таргет та видалити цю колонку з датасету (видаляти за індексом)
data = np.genfromtxt("iris.data.txt", dtype=object, encoding=None, delimiter=",")
arr = np.delete(data, 4, 1).astype('float64')

#2.Перетворити колонки, що залишились в 2D масив (або впевнитись, що це уже 2D масив)
def check_dim(data):
    return data.ndim

#3.Порахувати mean, median, standard deviation для 1-ї колонки
def stats_per_col(data, col_num):
    print('3:')
    print(f'Mean is: {np.mean(data[:, col_num])}, Median is: {np.median(data[:, col_num])}, Standard Deviation is: {np.std(data[:, col_num])}')

#4.Вставити 20 значень np.nan на випадкові позиції в масиві (при використанні звичайного 
#рандому можуть накластись позиції, тому знайти рішення, яке гарантує 20 унікальних позицій)
def replace_to_nan(data, num_elements):
    indices = np.unravel_index(np.random.choice(data.size, num_elements, replace=False), data.shape)
    arr[indices] = np.nan
    return arr

#5.Знайти позиції вставлених значень np.nan в 1-й колонці
def find_nan_col(data, column):
    indices = np.argwhere(np.isnan(data[:, column]))
    return indices

#6.Відфільтрувати массив за умовою: значення в 3-й колонці > 1.5 та значения в 1-й колонці < 5.0 (зберегти у іншу змінну)
def filter_by_condition(data):
    data = data[(data[:, 2] > 1.5) & (data[:, 0] < 5)]
    return data

#7.Замінити всі значення np.nan на 0
def nan_replace(data):
    data[np.isnan(data)] = 0
    return data

#8.Порахувати всі унікальні значення в массиві та вивести їх разом із кількістю
def unique_values(data):
    result = np.unique(data, return_counts=True)
    return result

#9.Розбити масив по вертикалі на 2 рівні частини (не використовувати абсолютні числа, мають бути два массиви по 4 колонки)
#10.Відсортувати обидва массиви по 1-й колонці: 1-й за збільшенням, 2-й за зменшенням
def division_sorting(data):
    print('9,10:')
    data = np.array_split(data, 2)
    data1, data2 = data[0], data[1]
    print(f'First array: \n {data1}, \n Second array: \n {data2}')
    data1 = data1[data1[:, 0].sort()]
    data2 = data2[data2[::-1][:, 0].sort()]
    print(f'First sorted array: \n {data1}, \n Second sorted array: \n {data2}')

#11.Зібрати обидва массиви в одне ціле
def reshape_2d(data):
    data = data.reshape(-1, data.shape[-1])
    return data

#12.Знайти найбільш часто повторюване значення в массиві
def most_freq(data):
    value, count = np.unique(data, return_counts=True)
    ind = np.argmax(count)
    return value[ind]

#13.Написати функцію, яка б множила всі значення в колонці, які менше середнього значения в цій колонці, 
# на 2, і ділила інші значення на 4. ((Повільна функція з довільною колонкою))
def cond_changes(data, col_num):
    column = data[:, col_num - 1]
    mean = np.mean(column)
    column = np.where(column < mean, column * 2, column / 4)
    data[:, col_num-1] = column
    return data

#14.Застосувати отриману функцію до 3-ї колонки ((Швидка функція котра приймає лише 1 готову колонку))
def cond_col_changes(data):
    mean = np.mean(data)
    data = np.where(data < mean, data * 2, data / 4)
    return data


print(f'2:\n{check_dim(arr)} Dimensions')
stats_per_col(arr, 0)
print(f'4:\n{replace_to_nan(arr, 20)}')
print(f'5:\n{find_nan_col(arr, 0)}')
print(f'6:\n{filter_by_condition(arr)}')
print(f'7:\n{nan_replace(arr)}')
print(f'8:\n{unique_values(arr)}')
division_sorting(arr)
print(f'11:\n{reshape_2d(arr)}')
print(f'12:\n{most_freq(arr)}')
# print(f'13:\n{cond_changes(arr, 3)}') #Повільна

col_arr = arr[:, 2] #Швидка - 3 колонка
arr[:, 2] = cond_col_changes(col_arr)
print(f'14:\n{arr}')