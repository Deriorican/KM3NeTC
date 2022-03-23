from SpaceStructures import *

def BuildDetector(filename):
    unitList = []
    file = open(filename, 'r')
    lines = file.readlines()
    first = True
    minX = np.inf
    minY = np.inf
    minZ = np.inf
    maxX = -np.inf
    maxy = -np.inf
    maxZ = -np.inf
    for line in lines:
        splitted = line.split()
        if len(splitted) == 0:
            pass
        elif len(splitted) == 4:
            if first:
                first = False
            else:
                sensorsRaw = np.array(sensorsRaw)
                X = np.mean(sensorsRaw[:,1])
                minX = X if minX > X else minX
                maxX = X if minX < X else maxX
                Y = np.mean(sensorsRaw[:,2])
                minY = Y if minY > Y else minY
                maxY = Y if minX < Y else maxY
                Z = np.mean(sensorsRaw[:,3])
                minZ = Z if minZ > Z else minZ
                maxZ = Z if minZ < Z else maxZ
                unitList.append(Unit(np.array([X, Y, Z])))
                for sens in sensorsRaw:
                    dx = sens[4]
                    dy = sens[5]
                    dz = sens[6]
                    sensors.append(Detector(np.array([dx, dy, dz])))
                unitList[-1].addDetectors(sensors)
            sensorsRaw = []
            sensors = []

        elif len(splitted) == 9:
            sensorsRaw.append(np.array([float(i) for i in splitted]))
    return unitList, [minX, maxX, minY, maxY, minZ, maxZ]

