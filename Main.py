import logging
import matplotlib.pyplot as plt
from multiprocessing import Process
import time
import SimAnn

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    Test1 = SimAnn.SimulatedAnnealing(iters = 200)
    Test2 = SimAnn.SimulatedAnnealing(iters= 300)
    Test3 = SimAnn.SimulatedAnnealing(iters= 400)
    Test2 = SimAnn.SimulatedAnnealing(iters= 450)

    def runInParallel(*fns):
        proc = []
        for fn in fns:
            p = Process(target=fn)
            p.start()
            proc.append(p)
        for p in proc:
            p.join()

    runInParallel(Test1, Test2, Test3)