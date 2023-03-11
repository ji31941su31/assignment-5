import numpy as np
from scipy.special import ndtri
import matplotlib.pyplot as plt

class SignalDetection:
    def __init__(self, hits, misses, false_alarms, correct_rejections):
        self.__hits = hits
        self.__misses = misses
        self.__false_alarms = false_alarms
        self.__correct_rejections = correct_rejections
    
    def hit_rate(self):
        return (self.__hits / (self.__hits + self.__misses))

    def false_alarm_rate(self):
        return (self.__false_alarms / (self.__false_alarms + self.__correct_rejections))

    def d_prime(self):
        return (ndtri(self.hit_rate()) - ndtri(self.false_alarm_rate()))

    def criterion(self):
        return -0.5 * (ndtri(self.hit_rate()) + ndtri(self.false_alarm_rate()))
    
    def __add__(self, other):
        return SignalDetection(self.__hits + other.__hits, self.__misses + other.__misses, self.__false_alarms + other.__false_alarms, self.__correct_rejections + other.__correct_rejections)
    
    def __mul__(self, scalar):
        return SignalDetection(self.__hits * scalar, self.__misses * scalar, self.__false_alarms * scalar, self.__correct_rejections * scalar)
    
    def plot_roc(self):
        point = [self.false_alarm_rate(), self.hit_rate()]
        begin = [0,0]
        target = [1,1]
        plt.plot(self.false_alarm_rate(), self.hit_rate(), marker = 'o')
        line_x1 = [begin[0],point[0]]
        line_x2 = [point[0],target[0]]
        line_y1 = [begin[1],point[1]]
        line_y2 = [point[1],target[1]]
        plt.plot(line_x1,line_y1)
        plt.plot(line_x2,line_y2)
        plt.plot([0,1],'--')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.show()
    
    def plot_sdt(self):
       d = SignalDetection.dprime(self)
       criterion = self.criterion
       x = np.linspace(-6,6,100)
       plt.plot(x, norm.pdf(x, 0, 1),label = 'noise')
       plt.plot(x, norm.pdf(x, -d, 1), label = 'signal')
       plt.axvline(x = criterion, label = 'criterion')
       plt.title('Noise vs. Signal distribution')
       plt.legend()
       plt.show()

    def simulate(dprime, criteria_list, signal_count, noise_count)
        sdt_list = list()
        for i in range(len(criteria_list)):
            hit_rate = norm.cdf(0.5*dprime - criteria_list[i])
            falsealarm_rate = norm.cdf(-0.5*dprime - criteria_list[i])
            hits, false_alarms = np.random.binomial(n = [signal_count, noise_count], p = [hit_rate. falsealarm_rate])
            misses = signal_count - hits
            correct_rejection = noise_count - false_alarms
            sdt_object = SignalDetection(hits, misses, false_alarms, correct_rejection)
            sdt_list.append(sdt_object)
        return sdt_list
    
    def plot_roc(sdt_list):
        
        