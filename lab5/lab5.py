import numpy as np
import math
from matplotlib import pyplot as plt
import csv


def multivariate_normal(ro, size, sigma_x, sigma_y):
    cov = [[1, ro/(sigma_x * sigma_y)], [ro/(sigma_x * sigma_y), 1]]
    return np.random.multivariate_normal(mean=[0,0], cov=cov, size=size).T


def misture_distribution(ro, sigma_x, sigma_y, size):
    x1, y1 = multivariate_normal(ro[0], size, sigma_x[0], sigma_y[0])
    x2, y2 = multivariate_normal(ro[1], size, sigma_x[1], sigma_y[1])
    x = np.array([])
    y = np.array([])
    for i in range(size):
        x = np.append(x, (0.9 * x1[i] + 0.1 * x2[i]))
        y = np.append(y, (0.9 * y1[i] + 0.1 * y2[i]))
    return x, y


def pearson_cor_coef(x, y, m_x, m_y, size):
    s = [0, 0, 0]
    for x_i, y_i in zip(x, y):
        s_x = x_i - m_x
        s_y = y_i - m_y
        s[0] += s_x * s_y
        s[1] += s_x ** 2
        s[2] += s_y ** 2
    return (s[0] / size) / math.sqrt((s[1] / size) * (s[2] / size))


def sample_quadrant_cor_coef(x, y, size):
    med_x = np.median(x)
    med_y = np.median(y)
    n = [0, 0, 0, 0]
    for x_i, y_i in zip(x, y):
        if x_i > med_x and y_i > med_y:
            n[0] += 1
        elif x_i < med_x and y_i > med_y:
            n[1] += 1
        elif x_i < med_x and y_i < med_y:
            n[2] += 1
        elif x_i > med_x and y_i < med_y:
            n[3] += 1
    return ((n[0] + n[2]) - (n[1] + n[3])) / size


def get_rang(distribution):
    rang = []
    distr_sort = sorted(distribution)
    for i in range(len(distribution)):
        rang.append(distr_sort.index(distribution[i]) + 1)
    return rang


def sample_rang_spirmen_cor_coef(x, y, size):
    middle_rang = (size + 1) / 2
    rang_x = get_rang(x)
    rang_y = get_rang(y)
    r = [0, 0, 0]
    for i in range(size):
        x_rang = rang_x[i] - middle_rang
        y_rang = rang_y[i] - middle_rang
        r[0] += x_rang * y_rang
        r[1] += x_rang ** 2
        r[2] += y_rang ** 2
    return (r[0] / size) / math.sqrt((r[1] / size) * (r[2] / size))


def equal_probability_ellipse(ro, sigma_x, sigma_y, x_, y_, color):
    x = np.linspace(-5, 5, 1000)
    y = np.linspace(-5, 5, 1000)
    x, y = np.meshgrid(x, y)
    Z = ((x-x_)**2)/(sigma_x**2) - 2 * ro * (x - x_) * (y - y_) / (sigma_x * sigma_y) + ((y - y_)**2) / (sigma_y ** 2)
    plt.contour(x,y,Z, [2], colors=[color])


def build_graph(x, y, ro, sigma_x, sigma_y, x_, y_, size, color):
    equal_probability_ellipse(ro, sigma_x, sigma_y, x_, y_, color[0])
    plt.scatter(x, y, marker='.', c=color[1])
    plt.xlabel('x')
    plt.ylabel('y')
    title = 'n=' + str(size) + ', ro=' + str(ro)
    plt.title(title)
    plt.savefig(title + '.png')
    plt.close()


def calculation_coef(x, y, x_, y_, size):
    return {'r': pearson_cor_coef(x, y, x_, y_, size),
            'r_q': sample_quadrant_cor_coef(x, y, size),
            'r_s': sample_rang_spirmen_cor_coef(x, y, size)}


def disp(val_1, val_2):
    return val_1 - val_2 ** 2


def calculations(size, x_, y_, sigma_x, sigma_y, ro_mix, sigma_x_mix, sigma_y_mix, x_mix, y_mix, count, mix, ro=0):
    values = {'e_z': {'r': 0, 'r_q': 0, 'r_s': 0},
              'e_z_2': {'r': 0, 'r_q': 0, 'r_s': 0},
              'd_z': {'r': 0, 'r_q': 0, 'r_s': 0}}
    for _ in range(count):
        if not mix:
            x, y = multivariate_normal(ro, size, sigma_x, sigma_y)
            m_x = x_
            m_y = y_
        else:
            x, y = misture_distribution(ro_mix, sigma_x_mix, sigma_y_mix, size)
            m_x = x_mix
            m_y = y_mix
        coef_z = calculation_coef(x, y, m_x, m_y, size)
        coef_z_2 = calculation_coef(x * x, y * y, m_x, m_y, size)
        coef_d_z = {'r': disp(coef_z_2['r'], coef_z['r']),
                    'r_q': disp(coef_z_2['r_q'], coef_z['r_q']),
                    'r_s': disp(coef_z_2['r_s'], coef_z['r_s'])}
        for key in coef_z:
            values['e_z'][key] += coef_z[key]
            values['e_z_2'][key] += coef_z_2[key]
            values['d_z'][key] += coef_d_z[key]

    for key in values:
        for k in values[key]:
            values[key][k] /= count
    return values


def file_print(dict, file, size, ro):
    file.writerow(('size=', size, ' ro=', ro))
    file.writerow((' ', 'r', 'r_q', 'r_s'))
    for key in dict:
        values = dict[key].values()
        file.writerow((key, *values))


if __name__ == '__main__':
    size_array = [20, 60, 100]
    ro_array = [0, 0.5, 0.9]
    colors = [['red', 'orange'], ['darkblue', 'skyblue'], ['purple', 'plum']]
    x_ = 0
    y_ = 0
    sigma_x = 1
    sigma_y = 1

    ro_mix = [0.9, -0.9]
    sigma_x_mix = [1, 10]
    sigma_y_mix = [1, 10]
    x_mix = 0
    y_mix = 0

    count = 1000

    for size, color in zip(size_array, colors):
        for ro in ro_array:
            x, y = multivariate_normal(ro, size, sigma_x, sigma_y)
            build_graph(x, y, ro, sigma_x, sigma_y, x_, y_, size, color)

    with open('results.csv', mode='w', encoding='utf-8') as file:
        file_writer = csv.writer(file)
        for size in size_array:
            file_writer.writerow('Normal')
            for ro in ro_array:
                dict = calculations(size, x_, y_, sigma_x, sigma_y, ro_mix, sigma_x_mix, sigma_y_mix, x_mix, y_mix, count, False, ro)
                file_print(dict, file_writer, size, ro)
            dict = calculations(size, x_, y_, sigma_x, sigma_y, ro_mix, sigma_x_mix, sigma_y_mix, x_mix, y_mix, count, True)
            file_writer.writerow('Mix')
            file_print(dict, file_writer, size, 'mix')
