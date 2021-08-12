import numpy as np
import math
import os
import sys


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


def convert_data(dir, i):
    """
    merge the data of a single run into a .npz file
    """
    position_path = dir + '/round' + str(i) + '/position.txt'
    time_path = dir + '/round' + str(i) + '/evacuation_time.txt'
    outside_path = dir + '/round' + str(i) + '/outside_point.txt'
    time = loadData(time_path)
    point = loadData(outside_path)
    output = loadData(position_path)

    if len(point) == 0 or len(time) == 0:
        sys.exit(1)

    T = int(time[-1] / 100)

    position = np.zeros((T, n, 3), dtype=np.float)
    for j in range(T):
        position[j] = output[j*n:(j+1)*n]

    save_name = dir + "/round" + str(i) + ".npz"
    np.savez(save_name, position=position, time=time, point=point)


def OutNum_vs_time(dir, i):
    """
    N_out v.s. t of a single run
    """
    save_name = dir + "/round" + str(i) + ".npz"
    data = np.load(save_name)
    number = data["point"][:, 1]
    return number


def cal_averOutNum_vs_time(dir):
    """
    平均逃离人数与时间的关系
    :param dir:
    :return:
    """
    numbers = [[]]
    for i in range(round):
        number = OutNum_vs_time(dir, i + 1).tolist()  # 对应一条曲线
        if i == 0:
            numbers[0] = number
        else:
            numbers.append(number)
        print(i)

    max_len = max((len(l) for l in numbers))  # round条曲线的最大长度
    new_numbers = list(map(lambda l: l + [n] * (max_len - len(l)), numbers))
    array = np.array(new_numbers)

    aver_num = array.sum(axis=0) / round
    save_name = dir + "/averOutNum_for_" + str(round) + "_rounds.npy"
    np.save(save_name, aver_num)


def cal_EvacuationTime(dir):
    EV = np.zeros(round)
    for i in range(round):
        save_name = dir + "/round" + str(i + 1) + ".npz"
        data = np.load(save_name)
        time = data["time"]
        EV[i] = time[-1] / 1000
        print(i)
    save_name = dir + "/EvacuationTime_for_" + str(round) + "_rounds.npy"
    np.save(save_name, EV)


def select_points_onBorder(dir, i, semi_time):
    save_name = dir + "/round" + str(i) + ".npz"
    data = np.load(save_name)
    position = data["position"]
    array = position[semi_time, :, 0:-1]

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
                    if np.tan(theta_2 + math.pi / 2) * (array[k][0] - door[0]) < array[k][1] < np.tan(
                            theta_1 + math.pi / 2) * (array[k][0] - door[0]):
                        point[j] = k
                        break
            point[j] = -1
    return point


def EscapeTime_vs_angle(dir, i, semi_time):
    """
    escape time of selected points in a single run
    """
    save_name = dir + "/round" + str(i) + ".npz"
    data = np.load(save_name)
    time = data["time"]

    point = select_points_onBorder(dir, i, semi_time)
    point_time = np.zeros(len(point))

    for j in range(len(point)):
        if point[j] < 0:
            point_time[j] = 0
        else:
            point_time[j] = time[point[j]] / 1000

    return point_time


def cal_EscapeTime_vs_angle(dir, semi_time):
    """
    escape time for multiple runs
    """
    point_time = np.zeros((round, 60), dtype=np.float)

    for i in range(round):
        point_time[i] = EscapeTime_vs_angle(dir, i + 1, semi_time)
        print(i)

    save_name = dir + "/EscapeTime_" + str(semi_time) + "_for_" + str(round) + "_rounds.npy"
    np.save(save_name, point_time)


def Devation_vs_angle(dir, i, semi_time):
    """
    deviation between trajectory and reference line in a single run
    """
    save_name = dir + "/round" + str(i) + ".npz"
    data = np.load(save_name)
    time = data["time"]
    position = data["position"]

    point = select_points_onBorder(dir, i, semi_time)
    point_time = time[point]

    dev = np.zeros(60)
    for k in range(60):
        lt = semi_time
        rt = int(point_time[k] / 100)
        id = point[k]
        if id != -1:  # there is a point in this zone
            x = position[lt:rt + 1, id, 0]
            y = position[lt:rt + 1, id, 1]
            #----------------compute L/L0--------------------#
            x1 = x[0:-1]
            x2 = list(x)
            x2.remove(x2[0])
            x2 = np.array(x2)
            y1 = y[0:-1]
            y2 = list(y)
            y2.remove(y2[0])
            y2 = np.array(y2)
            dev[k] = np.sum(np.sqrt((x2-x1)**2+(y2-y1)**2))/cal_distance(position[lt, id, 0:2], door)
        else:
            dev[k] = 0

    return dev


def cal_averDeviation_vs_angle(dir, semi_time):
    """
    average deviation of multiple runs
    """
    count = np.zeros(60)
    sum_dev = np.zeros(60)
    for i in range(round):
        dev = Devation_vs_angle(dir, i + 1, semi_time)
        logi = np.logical_and(dev > 0, True)
        count[logi] = count[logi] + 1
        sum_dev = sum_dev + dev
        print(i)
    aver_dev = sum_dev / count

    save_name = dir + "/averDevbot_" + str(semi_time) + "_for_" + str(round) + "_rounds.npy"
    np.save(save_name, aver_dev)


def fix_boundary_neighbor(dir, k):

    t = 0
    while True:

        neighborList = {} # a dictionary
        elementpath = dir + "/round" + str(k) + "/result/output" + str(t * 100) + "_elements.txt"
        neighborpath = dir + "/round" + str(k) + "/result/output" + str(t * 100) + "_neighbor.txt"

        if os.path.exists(elementpath):
            triangles = loadData(elementpath) - 1
            neighbors = loadData(neighborpath)

            for i in range(len(neighbors)):
                neighborList[str(i)] = []

            for j in range(len(triangles)):
                triangle = triangles[j]
                neighborList[str(triangle[0])].extend([triangle[1], triangle[2]])
                neighborList[str(triangle[1])].extend([triangle[0], triangle[2]])
                neighborList[str(triangle[2])].extend([triangle[0], triangle[1]])

            for j in range(len(neighbors)): # find the boundary points
                neighborList[str(j)] = list(set(neighborList[str(j)]))
                if neighbors[j][2] != len(neighborList[str(j)]):
                    neighbors[j][2] = len(neighborList[str(j)])

            dt = pd.DataFrame(neighbors)
            dt.to_csv(neighborpath, sep='\t', index=0, header=None)
        else:
            break

        t = t + 1


def cal_orient_order(dir, k):
    """
    compute phiT of a single run
    """
    phiT = []
    t = 0

    while True:
        count = 0
        neighborList = {}  # a dictionary
        boundaryPoint = []
        elementpath = dir + "/round" + str(k) + "/triangulation/output" + str(t * 100) + "_elements.txt"
        neighborpath = dir + "/round" + str(k) + "/triangulation/output" + str(t * 100) + "_neighbor.txt"

        if os.path.exists(elementpath):
            triangles = loadData(elementpath) - 1
            neighbors = loadData(neighborpath)
            abphis = np.zeros(len(neighbors))

            for i in range(len(neighbors)):
                neighborList[str(i)] = []

            for j in range(len(triangles)):
                triangle = triangles[j]
                neighborList[str(triangle[0])].extend([triangle[1], triangle[2]])
                neighborList[str(triangle[1])].extend([triangle[0], triangle[2]])
                neighborList[str(triangle[2])].extend([triangle[0], triangle[1]])

            for j in range(len(neighbors)):  # find the boundary points
                neighborList[str(j)] = list(set(neighborList[str(j)]))  # 去掉重复元素
                if neighbors[j][2] != len(neighborList[str(j)]):
                    boundaryPoint.append(j)

            for j in range(len(neighbors)):  # 计算非边界点的序参量
                if neighbors[j][2] == len(neighborList[str(j)]):
                    vec = neighbors[j][0:2]
                    if cal_distance(vec, door) > 10*0.6:
                        flag = 1
                        for s in range(len(boundaryPoint)):
                            vec1 = neighbors[boundaryPoint[s]][0:2]
                            if cal_distance(vec, vec1) < 5*0.6:
                                flag = 0
                                break
                        if flag == 1:  # compute its orientation order
                            phis = 0j
                            for l in range(int(neighbors[j][2])):
                                theta = np.arctan2(neighbors[neighborList[str(j)][l]][1]-neighbors[j][1], neighbors[neighborList[str(j)][l]][0]-neighbors[j][0])
                                phis = phis + np.exp(6*theta*1j)
                            abphis[count] = abs(phis) / neighbors[j][2]
                            count = count + 1
            if count != 0:
                phiT.append(np.sum(abphis)/count)
        else:
            break

        t = t + 10  # every 1 second

    save_name = dir + "/round" + str(k) + "/phiT.npy"
    np.save(save_name, phiT)


def cal_OrientOrder_vs_time(dir):
    round = 100
    T = 95
    phiTs = np.zeros((round, T), dtype=np.float)
    for i in range(round):
        save_name = dir + "/round" + str(i+1) + "/phiT.npy"
        phiT = np.load(save_name)
        phiTs[i] = phiT[0:T]
        print(i+1)

    save_name = dir + "/phiTs_for_" + str(round) + "_rounds.npy"
    np.save(save_name, phiTs)


if __name__ == "__main__":
    n = 1000
    r = 0.3
    right = 15.0
    r1 = 13.5
    r2 = 12.5
    # r1 = 7.5  # change the values of r1 and r2 when necessary (t=116s)
    # r2 = 6.5
    round = 6000
    door = np.array([right, 0])  # centre of the door

    dir = "data"  # put the simulation data under this directory (.npz file, triangulation results)
    for i in range(round):
        convert_data(dir, i+1)
    # cal_averOutNum_vs_time(dir)
    # cal_EvacuationTime(dir)
    # cal_EscapeTime_vs_angle(dir, 0)  # 0, 40, 56, 116
    # cal_averDeviation_vs_angle(dir, 0)  # 0, 40, 56, 116
    # cal_OrientOrder_vs_time(dir)
