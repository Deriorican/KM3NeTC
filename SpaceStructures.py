import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.optimize import minimize
import random as rd


class SourcePoint:
    def __init__(self, intensity, coordinates):
        self.coordinates = coordinates
        self.I = intensity


class Detector:
    def __init__(self, orientation):
        self.orientation = orientation
        self.parent = None
        self.coordinates = None
        self.value = 0

    def incValue(self, inc):
        self.value += inc

    def resetValue(self):
        self.value = 0

    def setParent(self, parent):
        self.parent = parent
        self.coordinates = self.parent.coordinates


class Unit:
    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.detectors = []

    def addDetectors(self, detectors):
        for detector in detectors:
            detector.setParent(self)
            self.detectors.append(detector)


class Space:
    def __init__(self, limits, n, noise, sigma):
        self.n = n
        self.limits = limits
        self.axes = []
        self.dim = len(self.limits) // 2
        for i in range(self.dim):
            self.axes.append(np.linspace(self.limits[i * 2], self.limits[i * 2 + 1], self.n))
        self.grid = np.meshgrid(*self.axes)
        self.units = []
        self.estimatedSourcePoints = []
        self.overlay = np.zeros(np.shape(self.grid[0]))
        self.noise_intensity = noise
        self.sigma = sigma

    def resetSpace(self):
        for unit in self.units:
            for det in unit.detectors:
                det.resetValue()
        self.overlay = np.zeros(np.shape(self.grid[0]))
        self.estimatedSourcePoints = []

    def addUnits(self, units):
        for unit in units:
            self.units.append(unit)

    def addSourcePoints(self, sourcePoints):
        for sourcePoint in sourcePoints:
            self.estimatedSourcePoints.append(sourcePoint)

    def FindLightSourcesByOverlay(self):
        for point in range(self.n ** self.dim):
            index_point = []
            rest = point
            for i in range(self.dim - 1, -1, -1):
                index_point.append(rest // (self.n ** i))
                rest = rest % (self.n ** i)
            index_point = tuple(index_point)
            current_point = []
            for i in range(self.dim):
                current_point.append(self.axes[i][index_point[i]])
            current_point = np.array(current_point)
            point_val = - self.OverlayVal(current_point)
            self.overlay[index_point] = point_val
        self.overlay = self.overlay.T
        print("overlayFilled")
        points, val = findLocalMaximas(self.overlay, self.dim, self.axes, self.n)
        l = []
        for i in range(len(points)):
            l.append(SourcePoint(val[i]*10, points[i]))
        self.addSourcePoints(l)

    def plotOverlay(self, special_points=[]):
        if self.dim == 2:
            plt.contour(self.axes[0], self.axes[1], self.overlay, levels=50)
            # plt.pcolormesh(self.axes[0], self.axes[1], self.overlay)
            for i in range(len(special_points)):
                plt.scatter(special_points[i].coordinates[0], special_points[i].coordinates[1], c="r")
            for i in range(len(self.estimatedSourcePoints)):
                plt.scatter(self.estimatedSourcePoints[i].coordinates[0], self.estimatedSourcePoints[i].coordinates[1],
                            marker="*", c="b")
            for unit in self.units:
                plt.scatter(unit.coordinates[0], unit.coordinates[1], marker="p", c="g")
            plt.show()

    def IntensityEqs(self, X):
        l_nbr = min(int(X[0]), (len(X) - 1) // (self.dim + 1))
        lights = []
        res = []
        for i in range(l_nbr):
            coordinates = []
            for j in range(self.dim):
                coordinates.append(X[1 + i * (self.dim + 1) + j])
            lights.append((X[1 + i * (self.dim + 1) + self.dim], np.array(coordinates)))
        for unit in self.units:
            for det in unit.detectors:
                res.append(det.value)
                for i in range(l_nbr):
                    relative_position = lights[i][1] - det.coordinates
                    distance = np.linalg.norm(relative_position)
                    cosAlpha = np.dot(relative_position, det.orientation) / (
                                np.linalg.norm(relative_position) * np.linalg.norm(det.orientation))
                    if cosAlpha > 0:
                        res[-1] -= cosAlpha * lights[i][0] / (2 ** (self.dim - 1) * np.pi * distance ** (self.dim - 1))
                if abs(res[-1]) < 3 * self.sigma:
                    res[-1] = 0
        return np.array(res)


    def OverlayVal(self, point):
        pointVal = 0
        dx = np.array([1e-2] * self.dim)
        for unit in self.units:
            for det in unit.detectors:
                rel_pos = point - det.coordinates
                if np.linalg.norm(rel_pos) == 0:
                    rel_pos = dx
                val = det.value
                coef = np.dot(rel_pos, det.orientation) / (
                        np.linalg.norm(rel_pos) * np.linalg.norm(det.orientation))
                val *= coef
                if val > 0:
                    pointVal -= val
        return pointVal

    def grad(self, x):
        dx = 1e-8
        grad_vec = np.zeros(3)
        for i in range(3):
            x1 = x + np.array([dx if j == i else 0 for j in range(3)])
            x2 = x - np.array([dx if j == i else 0 for j in range(3)])
            grad_vec[i] = (self.OverlayVal(x1) - self.OverlayVal(x2)) / (2 * dx)
        return grad_vec


    def FindLightSourcesByUnfullOverlay(self):
        threshold = 1e-8
        corrector_factor = -34.56891070437435
        zones = 2
        initialGuesses = []
        bounds = []
        for i in range(zones ** self.dim):
            tempGuess = []
            tempBound = []
            for j in range(self.dim):
                tempGuess.append(self.limits[j * 2] + ((self.limits[j * 2 + 1] - self.limits[j * 2]) / zones) * (
                    (i // (zones ** j)) % zones + 1/2))
                tempBound.append((self.limits[2 * j], self.limits[2 * j + 1]))
            bounds.append(tempBound)
            initialGuesses.append(np.array(tempGuess))
        points = []
        vals = []
        sol = []
        for i in range(len(initialGuesses)):
            print("start at " + str(initialGuesses[i]) + " in " + str(bounds[i]))
            ans = minimize(self.OverlayVal, initialGuesses[i], bounds=bounds[i], tol=threshold, jac=self.grad)
            if ans.success:
                if not foundInList(ans.x, points, 0.05):
                    points.append(ans.x)
                    vals.append(ans.fun * corrector_factor)
                    print("source found at : " + str(points[-1]) + " with value : " + str(vals[-1]))
                else :
                    print("already found")
        for i in range(len(points)):
            sol.append(SourcePoint(vals[i], points[i]))
        self.addSourcePoints(sol)


    def FindLightSourcesByEqs(self, min_number, max_number):
        realSol = None
        threshold = 1e-10
        for i in range(min_number, max_number + 1):
            x0 = [i]
            x0.extend([2] * i * (self.dim + 1))
            sol = root(self.IntensityEqs, np.array(x0), method='lm')
            test_sol = self.IntensityEqs(sol.x)
            error_array = np.ones(np.shape(test_sol)) * threshold
            if (test_sol < error_array).all():
                realSol = sol
                break
        if realSol is not None:
            new_ls = []
            for i in range(int(realSol.x[0])):
                coordinates = []
                for j in range(self.dim):
                    coordinates.append(realSol.x[1 + i * (self.dim + 1) + j])
                new_ls.append(SourcePoint(realSol.x[1 + i * (self.dim + 1) + self.dim], np.array(coordinates)))
            self.addSourcePoints(new_ls)
            return True
        else:
            return False

    def FindLightSources(self, min_number, max_number):
        if not self.FindLightSourcesByEqs(min_number, max_number):
            self.FindLightSourcesByUnfullOverlay()

    def plotSources(self, special_points=[]):
        fig = plt.figure()
        if self.dim == 2:
            ax = fig.add_subplot()
            for i in range(len(special_points)):
                ax.scatter(special_points[i].coordinates[0], special_points[i].coordinates[1], c="r",
                           s=special_points[i].I * 10, alpha=0.3, label="Sources simulées")
            for i in range(len(self.estimatedSourcePoints)):
                ax.scatter(self.estimatedSourcePoints[i].coordinates[0], self.estimatedSourcePoints[i].coordinates[1],
                           marker="*", c="b", s=self.estimatedSourcePoints[i].I * 10, label = "Sources estimées")
            for unit in self.units:
                ax.scatter(unit.coordinates[0], unit.coordinates[1], marker="p", c="g", label = "Unité de détection")
        if self.dim == 3:
            ax = fig.add_subplot(projection='3d')
            for i in range(len(special_points)):
                ax.scatter(special_points[i].coordinates[0], special_points[i].coordinates[1],
                           special_points[i].coordinates[2], c="r", s=special_points[i].I * 10, alpha=0.3, label="Sources simulées")
            for i in range(len(self.estimatedSourcePoints)):
                ax.scatter(self.estimatedSourcePoints[i].coordinates[0], self.estimatedSourcePoints[i].coordinates[1],
                           self.estimatedSourcePoints[i].coordinates[2], marker="*", c="b",
                           s=self.estimatedSourcePoints[i].I * 10, label = "Sources estimées")
            for unit in self.units:
                ax.scatter(unit.coordinates[0], unit.coordinates[1], unit.coordinates[2], marker="p", c="g")

        plt.show()


def propagateLight(space, sourcePoints):
    for unit in space.units:
        for det in unit.detectors:
            for sourcePoint in sourcePoints:
                rel_pos = sourcePoint.coordinates - det.coordinates
                val = sourcePoint.I / (2 ** (space.dim - 1) * np.pi * np.linalg.norm(rel_pos) ** (space.dim - 1))
                coef = np.dot(rel_pos, det.orientation) / (np.linalg.norm(rel_pos) * np.linalg.norm(det.orientation))
                val *= coef
                if val > 0:
                    det.incValue(val)
            det.incValue(rd.gauss(space.noise_intensity, space.sigma))


def foundInList(a, l, tol):
    tolArray = np.ones(np.shape(a)) * tol
    for b in l:
        if (a - b <= tolArray).all():
            return True
    return False


def findLocalMaximas(values, dim, axes, n):
    points = []
    val = []
    for i in range(0, n ** dim):
        currentPoint = []
        rest = i
        for j in range(dim - 1, -1, -1):
            currentPoint.append(rest // (n ** j))
            rest = rest % (n ** j)
        currentVal = values[tuple(currentPoint)]
        testMax = True
        for k in range(dim):
            for l in [-1, 1]:
                alternativePoint = np.array(currentPoint)
                alternativePoint[k] += l
                if alternativePoint[k] < n and values[tuple(alternativePoint)] >= currentVal:
                    testMax = False
        if testMax:
            coords = []
            for j in range(dim):
                coords.append(axes[j][currentPoint[dim - 1 - j]])
            coords = np.array(coords)
            if not foundInList(coords, points, 0.5):
                points.append(coords)
                val.append(currentVal)
    return points, val
