from Initialization import *

if __name__ == '__main__':

    dim = 3
    units, limits = BuildDetector("Data/SensorsPos.detx")
    n = 50
    noise = 0.001
    sigma = 0.0001

    space = Space(limits, n, noise, sigma)
    space.addUnits(units)

    randomPos = np.random.rand(3)
    for i in range(dim):
        randomPos[i] = limits[i*2] + randomPos[i] * (limits[i*2 + 1] - limits[i*2])
    randomSource = SourcePoint(10, randomPos)

    propagateLight(space, [randomSource])

    space.FindLightSources(0, 10)
    print("real source : " + str(randomSource.coordinates) + " with intensity : " + str(randomSource.I))
    print("estimated source(s) : " + str([i.coordinates for i in space.estimatedSourcePoints]) +
          " with intensities : " + str([i.I for i in space.estimatedSourcePoints]))
    space.plotSources([randomSource])