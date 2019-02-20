# BP_neural_network

题目要求：
采用的数据集著名的“MNIST数据集”完成一个神经网络的训练和测试，不允许使用tensorflow等框架。并用两种不同的bp模型做性能对比 （比如一个层数和神经元较少的简单模型和一个层数和神经元较多的复杂模型）。

def load(path, kind='train')#数据加载函数，kind值标明了读取文件的类型
class neuralNetwork:
    def __init__(self,numNeuronLayers,numNeurons_perLayer,learningrate)神经网络层数，每层神经元个数，学习率
    update(self,inputnodes,targets)训练函数，参数为输入节点，目标值，包括前馈和后向训练
    test(self,test_inputnodes,test_labels)测试函数，判断模型输出值是否与预期值一致