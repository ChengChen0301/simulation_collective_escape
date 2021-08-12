import numpy as np
import matplotlib.pyplot as plt
import math
import os
import cv2
from matplotlib.patches import Circle
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FuncFormatter


def cal_distance(x, y):
    """
    compute the distance between two vectors
    """
    return np.sqrt(np.sum((x-y)**2))


def loadData(filepath):
    """
    read data from a .txt file
    """
    f = open(filepath, 'r')
    lines = f.readlines()
    data = []
    for line in lines:
        temp1 = line.strip('\n')
        temp2 = temp1.split('\t')
        temp2 = list(map(eval, temp2))
        data.append(temp2)
    f.close()
    return np.array(data)


def mkdir(path):
    """
    create directory
    """
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径


def draw_grid():
    """
    draw Figure.1
    :return:
    """
    save_name = "samples/round140.npz"
    data = np.load(save_name)
    position = data["position"]

    fig = plt.figure(figsize=(25, 16))
    grid = plt.GridSpec(4, 3)
    ax = plt.subplot(grid[0:4, 0:2])
    ax2 = plt.subplot(grid[1:3, 2])
    ax.axis('equal')
    ax2.axis('equal')

    ax.spines['top'].set_visible(False)  # 去掉边框
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for j in range(len(wall)):
        ax.plot([wall[j][0], wall[j][2]], [wall[j][1], wall[j][3]], 'b', lw=5)

    ax.plot([wall[4][0], wall[4][0] + width], [wall[4][1], wall[4][1]], 'b', lw=5)
    ax.plot([wall[3][2], wall[3][2] + width], [wall[3][3], wall[3][3]], 'b', lw=5)

    def draw_circle(vec):
        """
        画圆的函数
        :param vec: 圆心的坐标向量
        """
        if vec[2] == 1:  # if the point is in the room
            ax.add_patch(Circle(xy=(vec[0], vec[1]), radius=r, alpha=0.5, color='black'))

    for j in range(200):
        if j != 174:
            draw_circle(position[0][j])
        vec = position[0][174]
        ax.add_patch(Circle(xy=(vec[0], vec[1]), radius=r, alpha=0.5, color='red'))

    ax.set_xlim(left-Rgrid, right+Rgrid)
    ax.set_ylim(down-Rgrid, up+Rgrid)
    ax.xaxis.set_minor_locator(plt.MultipleLocator(Rgrid))  # 设置x从坐标间隔
    ax.yaxis.set_minor_locator(plt.MultipleLocator(Rgrid))  # 设置y从坐标间隔
    ax.grid(which='minor', axis='x', linewidth=1, linestyle='-', color='r')  # 由每个x主坐标出发对x主坐标画垂直于x轴的线段
    ax.grid(which='minor', axis='y', linewidth=1, linestyle='-', color='r')
    ax.plot([0, 0], [Rgrid, 4*Rgrid], linewidth=3, color='black')  # 圈住九个小格子
    ax.plot([0, 3*Rgrid], [4 * Rgrid, 4 * Rgrid], linewidth=3, color='black')
    ax.plot([3*Rgrid, 3*Rgrid], [4 * Rgrid, Rgrid], linewidth=3, color='black')
    ax.plot([3*Rgrid, 0], [Rgrid, Rgrid], linewidth=3, color='black')

    ax2.spines['top'].set_visible(False)  # 去掉边框
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.plot([left, left], [down, up], linewidth=3, color='black')  # 圈住九个小格子
    ax2.plot([left, right], [up, up], linewidth=3, color='black')
    ax2.plot([right, right], [up, down], linewidth=3, color='black')
    ax2.plot([right, left], [down, down], linewidth=3, color='black')
    ax2.plot([left, right], [(up + 2 * down) / 3, (up + 2 * down) / 3], linewidth=3, color='red')
    ax2.plot([left, right], [(2 * up + down) / 3, (2 * up + down) / 3], linewidth=3, color='red')
    ax2.plot([(right + 2 * left) / 3, (right + 2 * left) / 3], [down, up], linewidth=3, color='red')
    ax2.plot([(2 * right + left) / 3, (2 * right + left) / 3], [down, up], linewidth=3, color='red')

    for j in range(200):
        vec = position[0][j]
        if 0 < vec[0] < 3*Rgrid and Rgrid < vec[1] < 4*Rgrid:
            vec[0] = vec[0] / (3 * Rgrid) * (right - left) + left
            vec[1] = (vec[1] - Rgrid) / (3 * Rgrid) * (up - down) + down
            if j != 174:
                ax2.add_patch(Circle(xy=(vec[0], vec[1]), radius=2.5, alpha=0.5, color='black'))
            else:
                ax2.add_patch(Circle(xy=(vec[0], vec[1]), radius=2.5, alpha=0.5, color='red'))

    xy = (3*Rgrid, 4*Rgrid)
    xy2 = (left, up)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data", axesA=ax2, axesB=ax,lw=3)
    ax2.add_artist(con)
    xy = (3 * Rgrid, Rgrid)
    xy2 = (left, down)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data", axesA=ax2, axesB=ax,lw=3)
    ax2.add_artist(con)

    ax.set_xticks([])
    ax.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=None, hspace=None)
    plt.savefig("figures/room.png")


def OutNum_vs_time(i):
    """
    N_out v.s. t of a single run
    """
    save_name = "samples/round" + str(i) + ".npz"
    data = np.load(save_name)
    number = data["point"][:, 1]
    return number


def draw_averOutNum_vs_time():
    """
    draw Figure.2a
    """
    save_name = "computeResults/averOutNum_for_" + str(round) + "_rounds.npy"
    Y1 = np.load(save_name).T / n  # N_out/N
    X1 = np.linspace(0, len(Y1)/10, num=len(Y1), endpoint=False, retstep=False, dtype=None)

    fig = plt.figure(figsize=(15, 10))
    ax = plt.gca()  # 获取当前的axes
    ax.spines['right'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)

    plt.plot(X1, Y1, color='black', lw=5, label='average')
    for i in [60, 3380]:
        Y = OutNum_vs_time(i)/n
        X = np.linspace(0, len(Y) / 10, num=len(Y), endpoint=False, retstep=False, dtype=None)
        plt.plot(X, Y, color='r', linestyle='--', lw=5)
    for i in [20]:
        Y = OutNum_vs_time(i) / n
        X = np.linspace(0, len(Y) / 10, num=len(Y), endpoint=False, retstep=False, dtype=None)
        plt.plot(X, Y, color='r', linestyle='--', lw=5, label='single')

    plt.xlabel("t (s)", fontsize=40)
    plt.ylabel("$N_{out} / N$", fontsize=40)
    plt.legend(loc='lower right', frameon=False, borderaxespad=1, prop={'size': 40})
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=40)
    plt.tick_params(labelsize=40)
    ax.tick_params(direction='in', width=3, length=15)
    ax.xaxis.set_major_locator(MultipleLocator(25))
    plt.savefig("figures/averOutNum_for_" + str(round) + "_rounds.png", bbox_inches='tight', pad_inches=0.1)


def draw_time_hist():
    """
    draw Figure.2b
    """
    save_name = "computeResults/EvacuationTime_for_" + str(round) + "_rounds.npy"
    TT = np.load(save_name)
    fig = plt.figure(figsize=(15, 10))
    plt.hist(TT, 30, normed=1, facecolor='blue', alpha=0.5, rwidth=0.8)

    ax = plt.gca()  # 获取当前的axes
    ax.spines['right'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)
    plt.tick_params(labelsize=40)
    plt.xlabel("$T_{ev}$ (s)", fontsize=40)
    plt.ylabel("Probability", fontsize=40)
    ax.tick_params(axis='x', direction='in', width=3, length=15)
    ax.tick_params(axis='y', direction='in', width=3, length=15)
    plt.xlim(141, 169)
    plt.savefig("figures/EvacuationTime_for_" + str(round) + "_rounds.png", bbox_inches='tight', pad_inches=0.1)


def select_points_onBorder(i, semi_time, visualize):
    """
    draw upper panels Figure.3
    """
    save_name = "samples/round" + str(i) + ".npz"
    data = np.load(save_name)
    position = data["position"]
    array = position[semi_time*10, :, 0:-1]

    point = np.zeros(60, dtype=np.int)
    alpha = 1.0 / 60
    for j in range(60):
        theta_1 = j * alpha * math.pi
        theta_2 = (j + 1) * alpha * math.pi
        for k in range(n):
            if r2 < cal_distance(array[k], door) <= r1:
                if j == 0:
                    if array[k][0] < door[0] and array[k][1] > np.tan(theta_2 + math.pi / 2) * (array[k][0] - door[0]):
                        point[j] = k
                        break
                elif j == 59:
                    if array[k][0] < door[0] and array[k][1] < np.tan(theta_1 + math.pi / 2) * (array[k][0] - door[0]):
                        point[j] = k
                        break
                else:
                    if np.tan(theta_2 + math.pi / 2) * (array[k][0] - door[0]) < array[k][1] < np.tan(theta_1 + math.pi / 2) * (array[k][0] - door[0]):
                        point[j] = k
                        break
            point[j] = -1

    #----------------------------------------------可视化----------------------------------------------#
    def draw_figure():
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.axis('equal')
        for j in range(len(wall)):
            ax.plot([wall[j][0], wall[j][2]], [wall[j][1], wall[j][3]], 'b')

        for l in range(len(array)):
            if l in point:
                ax.add_patch(Circle(xy=(array[l][0], array[l][1]), radius=r, color='red', alpha=0.5))
            else:
                if position[semi_time*10][l][-1] == 1:
                    ax.add_patch(Circle(xy=(array[l][0], array[l][1]), radius=r, color='black', alpha=0.5))
            if 0 < l < 60:
                theta_1 = l * alpha * math.pi
                x = door[0] - r1 * np.cos(math.pi/2 - theta_1)
                y = r1 * np.sin(math.pi/2 - theta_1)
                ax.plot([door[0], x], [door[1], y], 'grey')

                if semi_time == 0:  # draw one angle for t = 0s
                    if l == 10:
                        ax.plot([door[0], x], [door[1], y], 'black', lw=5)
                        plt.annotate('$\phi$', xy=(right-r1/8, np.sqrt(3)/8*r1), xytext=(right, r1/4), xycoords='data',
                                    fontsize=40, arrowprops=dict(arrowstyle='->', connectionstyle='arc3', lw=4))
                    else:
                        ax.plot([door[0], x], [door[1], y], 'grey')

                else:
                    if l == 30 or l == 10 or l == 50:  # draw three angles
                        x = door[0] - (r1+1) * np.cos(math.pi / 2 - theta_1)
                        y = (r1+1) * np.sin(math.pi / 2 - theta_1)
                        ax.plot([door[0], x], [door[1], y], 'green', lw=5)
                        if semi_time == 116:
                            plt.annotate('$30^\circ$', xy=(right - r1 / 4, np.sqrt(3) / 4 * r1), xytext=(right, r1 / 2),
                                         xycoords='data', fontsize=40,
                                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3', lw=4))
                        else:
                            plt.annotate('$30^\circ$', xy=(right - r1 / 8, np.sqrt(3) / 8 * r1), xytext=(right, r1 / 4),
                                         xycoords='data', fontsize=40,
                                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3', lw=4))
                        plt.annotate('', xy=((right+x)/2, y/2), xytext=(x,y), arrowprops={'width': 5, 'headwidth': 20, 'headlength': 30, 'color': 'green'})

                    else:
                        ax.plot([door[0], x], [door[1], y], 'grey')

        x = np.arange(-15.0, door[0], 0.1)
        y = np.arange(-15.0, 15.0, 0.1)
        x, y = np.meshgrid(x, y)
        plt.contour(x, y, (x - door[0]) ** 2 + y ** 2, [r1 ** 2])  # x**2 + y**2 = r**2 的圆形
        plt.contour(x, y, (x - door[0]) ** 2 + y ** 2, [r2 ** 2])
        plt.title("t = " + str(semi_time) + "s", fontsize=40)
        plt.axis('off')
        plt.subplots_adjust(left=None, bottom=0.17, right=0.96, top=0.94, wspace=None, hspace=None)
        plt.savefig("figures/point_selection_" + str(semi_time) + ".png")

    if visualize:
        draw_figure()
    #----------------------------------------------可视化----------------------------------------------#

    return point


def draw_averEscapeTime_vs_angle(semi_time):
    """
    draw lower panels of Figure.3
    """
    X = np.linspace(0, 60, num=60, endpoint=False, retstep=False, dtype=None)  # the ith zone
    save_name = "computeResults/errorbar/EscapeTime_" + str(semi_time*10) + "_for_2000_rounds"

    error_bar = np.zeros((50, 60), dtype=np.float)
    for j in range(50):
        filename = save_name + '_' + str(j+1) + '.npy'
        point_time = np.load(filename)
        for i in range(60):
            array = point_time[:, i]
            array = array[array > 0]
            x_mean = np.mean(array)
            error_bar[j][i] = x_mean

    fig = plt.figure(figsize=(35, 28))
    ax = plt.gca()  # 获取当前的axes
    ax.spines['right'].set_linewidth(5)
    ax.spines['bottom'].set_linewidth(5)
    ax.spines['left'].set_linewidth(5)
    ax.spines['top'].set_linewidth(5)

    Y = np.zeros(60)
    for i in range(60):
        array = error_bar[:, i]
        Y[i] = np.mean(array)
        plt.errorbar(X[i], np.mean(error_bar[:, i]), yerr=np.std(error_bar[:, i]), ms=30, mfc='w', mew=6, fmt='-o', lw=8, ecolor='black')

    plt.plot(X, Y, color='black', linestyle='-', lw=8)

    plt.tick_params(labelsize=100)
    ax.tick_params(direction='in', width=5, length=30)
    plt.subplots_adjust(left=0.17, bottom=0.15, right=0.95, top=0.9, wspace=None, hspace=None)
    plt.xlabel("$\phi$ (deg)", fontsize=120)
    plt.ylabel("$\langle T_{es} \\rangle$ (s)", fontsize=120)

    ax.xaxis.set_major_locator(MultipleLocator(10))
    if semi_time == 0:
        ax.yaxis.set_major_locator(MultipleLocator(10))

    def to_percent(temp, position):
        return '%1.0f' % (3 * temp)

    plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))

    if semi_time == 0:
        x1 = 30
        y1 = Y[x1]
        plt.annotate('', xy=(x1, y1 + r), xytext=(x1, y1 + r + 10),
                     arrowprops={'width': 10, 'headwidth': 40, 'headlength': 50, 'color': 'green'})
    elif semi_time == 40:
        x0 = 10
        y0 = Y[x0]
        plt.annotate('', xy=(x0, y0 + r), xytext=(x0, y0 + r + 1.5),
                     arrowprops={'width': 10, 'headwidth': 40, 'headlength': 50, 'color': 'green'})
        x1 = 30
        y1 = Y[x1]
        plt.annotate('', xy=(x1, y1 + r), xytext=(x1, y1 + r + 1.5),
                     arrowprops={'width': 10, 'headwidth': 40, 'headlength': 50, 'color': 'green'})
        x2 = 50
        y2 = Y[x2]
        plt.annotate('', xy=(x2, y2 + r), xytext=(x2, y2 + r + 1.5),
                     arrowprops={'width': 10, 'headwidth': 40, 'headlength': 50, 'color': 'green'})
    elif semi_time == 56:
        x0 = 10
        y0 = Y[x0]
        plt.annotate('', xy=(x0, y0 + r), xytext=(x0, y0 + r + 1),
                     arrowprops={'width': 10, 'headwidth': 40, 'headlength': 50, 'color': 'green'})
        x1 = 30
        y1 = Y[x1]
        plt.annotate('', xy=(x1, y1 + r), xytext=(x1, y1 + r + 1),
                     arrowprops={'width': 10, 'headwidth': 40, 'headlength': 50, 'color': 'green'})
        x2 = 50
        y2 = Y[x2]
        plt.annotate('', xy=(x2, y2 + r), xytext=(x2, y2 + r + 1),
                     arrowprops={'width': 10, 'headwidth': 40, 'headlength': 50, 'color': 'green'})
    else:
        x0 = 10
        y0 = Y[x0]
        plt.annotate('', xy=(x0, y0 + r), xytext=(x0, y0 + r + 0.5),
                     arrowprops={'width': 10, 'headwidth': 40, 'headlength': 50, 'color': 'green'})
        x1 = 30
        y1 = Y[x1]
        plt.annotate('', xy=(x1, y1 + r), xytext=(x1, y1 + r + 0.5),
                     arrowprops={'width': 10, 'headwidth': 40, 'headlength': 50, 'color': 'green'})
        x2 = 50
        y2 = Y[x2]
        plt.annotate('', xy=(x2, y2 + 2*r), xytext=(x2, y2 + 2*r + 0.5),
                     arrowprops={'width': 10, 'headwidth': 40, 'headlength': 50, 'color': 'green'})

    plt.subplots_adjust(left=0.17, bottom=0.15, right=0.95, top=0.9, wspace=None, hspace=None)
    plt.savefig("figures/averEscapeTime_" + str(semi_time) + "_for_2000_rounds.png")


def curve_vs_line(i, semi_time):
    """
    drow Figure.5a
    """
    save_name = "samples/round" + str(i) + ".npz"
    data = np.load(save_name)
    time = data["time"]
    position = data["position"]

    point = select_points_onBorder(i, semi_time, False)

    point_time = np.zeros(len(point))
    for j in range(len(point)):
        point_time[j] = time[int(point[j])]

    k = 5
    lt = semi_time*10
    rt = int(point_time[k] / 100)
    id = int(point[k])
    if id != -1:  # there is a point in this zone
        x1 = position[lt:rt + 1, id, 0]
        y1 = position[lt:rt + 1, id, 1]

    k = 30
    lt = semi_time*10
    rt = int(point_time[k] / 100)
    id = int(point[k])
    if id != -1:  # there is a point in this zone
        x2 = position[lt:rt + 1, id, 0]
        y2 = position[lt:rt + 1, id, 1]

    # -----------------------------------------可视化-------------------------------------------#
    fig = plt.figure(figsize=(28, 28))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('equal')
    ax.set_xlim(left - Rgrid, right + Rgrid)
    ax.set_ylim(down - Rgrid, up + Rgrid)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.xticks([])
    plt.yticks([])
    for j in range(len(wall)):
        ax.plot([wall[j][0], wall[j][2]], [wall[j][1], wall[j][3]], 'b', lw=5)

    def draw_circle(vec):
        if vec[2] == 6:
            circle = mpatches.Circle(xy=(vec[0], vec[1]), radius=r, color='silver', alpha=1)
        elif vec[2] == 7:
            circle = mpatches.Circle(xy=(vec[0], vec[1]), radius=r, color='blue', alpha=1)
        elif vec[2] == 5:
            circle = mpatches.Circle(xy=(vec[0], vec[1]), radius=r, color='red', alpha=1)
        else:
            circle = mpatches.Circle(xy=(vec[0], vec[1]), radius=r, color='green', alpha=1)

        circle.set_zorder(1)
        ax.add_patch(circle)

    def draw_triangle(array1, array2):
        for i in range(len(array1)):
            line = mlines.Line2D([array2[int(array1[i][0])][0], array2[int(array1[i][1])][0]],
                                 [array2[int(array1[i][0])][1], array2[int(array1[i][1])][1]], color='silver', lw=5)
            line.set_zorder(0)
            ax.add_line(line)
            line = mlines.Line2D([array2[int(array1[i][1])][0], array2[int(array1[i][2])][0]],
                                 [array2[int(array1[i][1])][1], array2[int(array1[i][2])][1]], color='silver', lw=5)
            line.set_zorder(0)
            ax.add_line(line)
            line = mlines.Line2D([array2[int(array1[i][2])][0], array2[int(array1[i][0])][0]],
                                 [array2[int(array1[i][2])][1], array2[int(array1[i][0])][1]], color='silver', lw=5)
            line.set_zorder(0)
            ax.add_line(line)

    def update(i):
        neighbor_path = file_path + '/output' + str(i) + '_neighbor.txt'
        neighbor = loadData(neighbor_path)
        triangle_path = file_path + '/output' + str(i) + '_elements.txt'
        triangle = loadData(triangle_path) - 1
        draw_triangle(triangle, neighbor)
        np.apply_along_axis(draw_circle, 1, neighbor)
        return ax

    file_path = 'samples/round140/result'
    update(semi_time*1000)
    plt.scatter(x2, y2, marker='o', color='r', label='1', s=100, zorder=2)  # 两条轨迹
    plt.scatter(x1, y1, marker='o', color='g', label='1', s=100, zorder=2)
    ax.plot([x1[-1], x1[0]], [y1[-1], y1[0]], 'black', lw=5)  # 两条直线
    ax.plot([x2[-1], x2[0]], [y2[-1], y2[0]], 'black', lw=5)  # 两条直线

    ax.spines['top'].set_visible(False)  # 去掉边框
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('figures/trajectory.png', bbox_inches='tight', pad_inches=0.1)
    # -----------------------------------------可视化-------------------------------------------


def draw_dev_vs_angle():
    """
    draw Figure.5b
    """
    X = np.linspace(0, 60, num=60, endpoint=False, retstep=False, dtype=None)  # the ith zone

    save_name = "computeResults/averDevbot_0_for_2000_rounds.npy"
    Y1 = np.load(save_name).T
    save_name = "computeResults/averDevbot_400_for_2000_rounds.npy"
    Y2 = np.load(save_name).T
    save_name = "computeResults/averDevbot_560_for_2000_rounds.npy"
    Y3 = np.load(save_name).T
    save_name = "computeResults/averDevbot_1160_for_2000_rounds.npy"
    Y4 = np.load(save_name).T

    fig = plt.figure(figsize=(15, 10))
    ax = plt.gca()  # 获取当前的axes
    ax.spines['right'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)
    plt.plot(X, Y1, color='r', ls='--', lw=5, label="t=0s")
    plt.plot(X, Y2, color='g', ls='--', lw=5, label="t=40s")
    plt.plot(X, Y3, color='b', ls='--', lw=5, label="t=56s")
    plt.plot(X, Y4, color='y', ls='--', lw=5, label="t=116s")

    plt.tick_params(labelsize=40)
    ax.tick_params(axis='x', direction='in', width=3, length=15)
    ax.tick_params(axis='y', direction='in', width=3, length=15)
    plt.xlabel("$\phi$ (deg)", fontsize=40)
    plt.ylabel("$\langle L \\rangle /L_0$", fontsize=40)

    plt.legend(frameon=False, borderaxespad=2)
    plt.legend(frameon=False, bbox_to_anchor=(0.02, 1.02, 0.95, 0.2), loc="lower left",
                    mode="expand", borderaxespad=-1, ncol=4, prop={'size': 20})
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=40)

    ax.xaxis.set_major_locator(MultipleLocator(10))

    def to_percent(temp, position):
        return '%1.0f' % (3 * temp)

    ax.xaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.savefig("figures/averDevbot_for_2000_rounds.png", bbox_inches='tight', pad_inches=0.1)


def draw_aver_phiT():
    """
    draw Figure.4a
    :return:
    """
    save_name = "computeResults/phiTs_for_100_rounds.npy"
    error_bar = np.load(save_name)
    Y = np.mean(error_bar, axis=0)
    X = np.linspace(0, len(Y)*10 / 10, num=len(Y), endpoint=False, retstep=False)

    fig = plt.figure(figsize=(15, 10))
    ax = plt.gca()  # 获取当前的axes
    ax.spines['right'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)

    plt.errorbar(X, Y, yerr=np.std(error_bar, axis=0), ms=30, mfc='w', mew=6, fmt='-', lw=5, ecolor='gray')
    plt.plot(X, Y, color='red', linestyle='-', lw=8)
    plt.vlines(x=35, ymin=0, ymax=Y[35], colors='k', linestyles='--', lw=5, data=None,)
    ax.set_ylim(0.2, 1.1)

    plt.xlabel("$t(s)$", fontsize=40)
    plt.ylabel("$\langle \Phi_6 \\rangle$", fontsize=40)
    plt.tick_params(labelsize=40)
    ax.tick_params(direction='in', width=3, length=15)
    ax.xaxis.set_major_locator(MultipleLocator(20))

    plt.savefig("figures/averphiT_for_100_rounds.png", bbox_inches='tight', pad_inches=0.1)


def draw_snapshots(i):
    """
    draw Figure.4b-d
    :param i:
    :return:
    """
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111)
    ax.axis('equal')

    neighbor_path = 'samples/round140/triangulation'

    def draw_circle(vec):
        if vec[0] < right:
            if vec[2] == 6:
                circle = mpatches.Circle(xy=(vec[0], vec[1]), radius=r, color='silver', alpha=1)
            elif vec[2] == 7:
                circle = mpatches.Circle(xy=(vec[0], vec[1]), radius=r, color='blue', alpha=1)
            elif vec[2] == 5:
                circle = mpatches.Circle(xy=(vec[0], vec[1]), radius=r, color='red', alpha=1)
            else:
                circle = mpatches.Circle(xy=(vec[0], vec[1]), radius=r, color='green', alpha=1)
            ax.add_patch(circle)

    def update(i):
        ax.clear()
        ax.axis('equal')
        for j in range(len(wall)):
            ax.plot([wall[j][0], wall[j][2]], [wall[j][1], wall[j][3]], 'b', lw=5)

        filepath = neighbor_path + '/output' + str(i*100) + '_neighbor.txt'
        neighbor = loadData(filepath)
        np.apply_along_axis(draw_circle, 1, neighbor)

        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        if i == 461:
            plt.annotate('', xy=(12.5, 3.6), xytext=(14.5, 5),
                         arrowprops={'width': 5, 'headwidth': 20, 'headlength': 20, 'color': 'green'})
        elif i == 473:
            plt.annotate('', xy=(9.2, 0.9), xytext=(11.2, 2.4),
                         arrowprops={'width': 5, 'headwidth': 20, 'headlength': 20, 'color': 'green'})
        elif i == 489:
            plt.annotate('', xy=(6, -2), xytext=(8, -0.5),
                         arrowprops={'width': 5, 'headwidth': 20, 'headlength': 20, 'color': 'green'})

        ax.set_title("$t = $" + str(i / 10.0) + "$s$", fontsize=40, y=-0.1)
        plt.subplots_adjust(left=0, bottom=0, top=1, right=1, wspace=None, hspace=None)
        plt.margins(0, 0)

    fig_holder = 'figures/snapshots'
    mkdir(fig_holder)
    update(i)
    plt.savefig(fig_holder + '/fig' + str(i) + '.png', bbox_inches='tight', pad_inches=0.1)


def get_averEscapeTime(save_name):
    point_time = np.load(save_name)

    averET = np.zeros(60)
    for i in range(60):
        array = point_time[:, i]
        array = array[array > 0]
        if len(array) > 10 or len(array) == 0:
            averET[i] = np.mean(array)
        else:
            array = []
            averET[i] = np.mean(array)
    return averET


def compare_noise():
    """
    draw Figure.6a
    """
    X = np.linspace(0, 60, num=60, endpoint=False, retstep=False, dtype=None)  # the ith zone

    save_name = "computeResults/EscapeTime_560_for_6000_rounds.npy"
    Y1 = get_averEscapeTime(save_name)

    save_name = "computeResults/noise1_0/EscapeTime_560_for_2000_rounds.npy"
    Y2 = get_averEscapeTime(save_name)

    save_name = "computeResults/noise10_0/EscapeTime_560_for_2000_rounds.npy"
    Y3 = get_averEscapeTime(save_name)

    fig = plt.figure(figsize=(16, 10))
    ax = plt.gca()  # 获取当前的axes
    ax.spines['right'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)

    plt.plot(X, Y1, color='r', linestyle='--', lw=5, label="$c=0$")
    plt.plot(X, Y2, color='b', linestyle='--', lw=5, label="$c=1$")
    plt.plot(X, Y3, color='g', linestyle='--', lw=5, label="$c=10$")

    plt.tick_params(labelsize=40)
    ax.tick_params(direction='in', width=3, length=15)
    plt.xlabel("$\phi$ (deg)", fontsize=40)
    plt.ylabel("$\langle T_{es} \\rangle$ (s)", fontsize=40)

    ax.xaxis.set_major_locator(MultipleLocator(10))

    def to_percent(temp, position):
        return '%1.0f' % (3 * temp)

    plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))

    plt.legend(loc='lower right', bbox_to_anchor=(0.8, -0.05), frameon=False, borderaxespad=2, prop={'size': 20})

    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    lframe = leg.get_frame()
    plt.setp(ltext, fontsize=40)
    plt.setp(lframe, lw=5, ec='black')
    plt.savefig("figures/compare_noise.png", bbox_inches='tight', pad_inches=0.1)


def compare_potential():
    """
    draw Figure.6b
    """
    X = np.linspace(0, 60, num=60, endpoint=False, retstep=False, dtype=None)  # the ith zone

    save_name = "computeResults/EscapeTime_560_for_6000_rounds.npy"
    Y1 = get_averEscapeTime(save_name)

    save_name = "computeResults/ratio0_3/EscapeTime_560_for_2000_rounds.npy"
    Y2 = get_averEscapeTime(save_name)

    save_name = "computeResults/ratio0_5/EscapeTime_560_for_2000_rounds.npy"
    Y3 = get_averEscapeTime(save_name)

    save_name = "computeResults/ratio0_7/EscapeTime_560_for_2000_rounds.npy"
    Y4 = get_averEscapeTime(save_name)

    fig = plt.figure(figsize=(16, 10))
    ax = plt.gca()  # 获取当前的axes
    ax.spines['right'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)

    plt.plot(X, Y1, color='r', linestyle='--', lw=5, label="$c'=0$")
    plt.plot(X, Y2, color='b', linestyle='--', lw=5, label="$c'=0.3$")
    plt.plot(X, Y3, color='g', linestyle='--', lw=5, label="$c'=0.5$")
    plt.plot(X, Y4, color='y', linestyle='--', lw=5, label="$c'=0.7$")

    plt.tick_params(labelsize=40)
    ax.tick_params(direction='in', width=3, length=15)
    plt.xlabel("$\phi$ (deg)", fontsize=40)
    plt.ylabel("$\langle T_{es} \\rangle$ (s)", fontsize=40)

    ax.xaxis.set_major_locator(MultipleLocator(10))

    def to_percent(temp, position):
        return '%1.0f' % (3 * temp)

    plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))

    plt.legend(loc='upper left', frameon=False, borderaxespad=2, prop={'size':20})

    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    lframe = leg.get_frame()
    plt.setp(ltext, fontsize=40)
    plt.setp(lframe, lw=5, ec='black')
    plt.savefig("figures/compare_potential.png", bbox_inches='tight', pad_inches=0.1)


def draw_video(i):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.axis('equal')

    neighbor_path = 'samples/round140/triangulation'

    def draw_circle(vec):
        if vec[0] < right:
            if vec[2] == 6:
                circle = mpatches.Circle(xy=(vec[0], vec[1]), radius=r, color='silver', alpha=1)
            elif vec[2] == 7:
                circle = mpatches.Circle(xy=(vec[0], vec[1]), radius=r, color='blue', alpha=1)
            elif vec[2] == 5:
                circle = mpatches.Circle(xy=(vec[0], vec[1]), radius=r, color='red', alpha=1)
            else:
                circle = mpatches.Circle(xy=(vec[0], vec[1]), radius=r, color='green', alpha=1)
            ax.add_patch(circle)

    def update(i):
        ax.clear()
        ax.axis('equal')
        for j in range(len(wall)):
            ax.plot([wall[j][0], wall[j][2]], [wall[j][1], wall[j][3]], 'b')

        filepath = neighbor_path + '/output' + str(i*100) + '_neighbor.txt'
        neighbor = loadData(filepath)
        np.apply_along_axis(draw_circle, 1, neighbor)

        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        ax.set_title("$c = 0, t = $" + str(i/10.0) + " $s$", fontsize=16)

    fig_holder = 'video'
    mkdir(fig_holder)
    update(i)
    plt.savefig(fig_holder + '/fig' + str(i) + '.png')


def imgs2video(length):
    for i in range(length):
        draw_video(i)

    fps = 10
    size = (1000, 1000)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    imgs_dir = 'video'
    save_name = imgs_dir + '.avi'
    video = cv2.VideoWriter(save_name, fourcc, fps, size)

    i = 0
    while i < length:
        filepath = imgs_dir + '/fig' + str(i) + '.png'
        img = cv2.imread(filepath)
        video.write(img)
        i = i+5  # 0.5 seconds
    video.release()


if __name__ == "__main__":
    n = 1000
    left = -15.0
    right = 15.0
    up = 15.0
    down = -15.0
    width = 1.5
    r = 0.25
    delta = 0.001
    r1 = 13.5
    r2 = 12.5
    # r1 = 7.5  # change the values of r1 and r2 when necessary (t=116s)
    # r2 = 6.5
    Rgrid = 1.2
    round = 6000
    door = np.array([right, 0])  # centre of the door
    wall = np.array([[right, up, left, up],  # 上
                    [left, up, left, down],  # 左
                    [left, down, right, down],  # 下
                    [right, down, door[0], -width],
                    [door[0], width, right, up]])  # 右
    # draw_grid()
    # draw_averOutNum_vs_time()
    # draw_time_hist()
    # select_points_onBorder(1800, 56, True)  # 0, 40, 56, 116
    # draw_averEscapeTime_vs_angle(116)  # 0, 40, 56, 116
    # curve_vs_line(140, 40)  # 0, 40, 56, 116
    # draw_dev_vs_angle()
    # draw_aver_phiT()
    # draw_snapshots(461)  # 461, 473, 489
    # compare_noise()
    # compare_potential()
    # imgs2video(100)