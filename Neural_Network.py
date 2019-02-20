import os
import struct
import numpy
#数据加载函数，kind值标明了读取文件的类型
def load(path, kind='train'):
    labels_path = os.path.join(path,'%s-labels.idx1-ubyte'% kind)
    images_path = os.path.join(path,'%s-images.idx3-ubyte'% kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = numpy.fromfile(lbpath,dtype=numpy.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = numpy.fromfile(imgpath,dtype=numpy.uint8).reshape(len(labels), 784)
    #读取到的labels为0-9的数字，需转化为十位的numpy数组
    ls=[]
    for i in range(labels.size):
        temp=numpy.zeros(10)
        temp[labels[i]]=1
        ls.append(temp.T)
    labels=numpy.array(ls)
    #由于源数据有些数据过大，会导致激活函数计算溢出，所以对数据集集体缩小，
    #由于图片数据每一位的值均为0-255之间，但统一除以255后发现当神经元个数达到一定数目或层数增加时还是会计算溢出，于是决定统一除以2550
    return (images/2550), labels
class neuralNetwork:
    def __init__(self,numNeuronLayers,numNeurons_perLayer,learningrate):
        self.numNeurons_perLayer=numNeurons_perLayer
        self.numNeuronLayers=numNeuronLayers
        self.learningrate = learningrate
        self.weight=[]
        for i in range(numNeuronLayers):
        	self.weight.append(numpy.random.normal(0.0, pow(self.numNeurons_perLayer[i+1],-0.5), 
                (self.numNeurons_perLayer[i+1],self.numNeurons_perLayer[i]) )  )
        self.activation_function = lambda x: 1.0/(1.0+numpy.exp(-x))
    def update(self,inputnodes,targets):     
        inputs = numpy.array(inputnodes,ndmin=2).T
        targets = numpy.array(targets,ndmin=2).T
        #前向传播
        #定义输出值列表（outputs[0]为输入值）
        self.outputs=[]
        self.outputs.append(inputs)
        #用激活函数对神经网络的每一层计算输出值，并保存到outputs列表中
        for i in range(self.numNeuronLayers):
        	temp_inputs=numpy.dot(self.weight[i],inputs)
        	temp_outputs=self.activation_function(temp_inputs)
        	inputs=temp_outputs
        	self.outputs.append(temp_outputs)
        #计算每层的训练误差
        self.output_errors=[]
        for i in range(self.numNeuronLayers):
        	#输出层的误差=目标值-输出值
        	if i == 0:
        		self.output_errors.append(targets - self.outputs[-1])
        	#隐藏层的误差=当前隐藏层与下一层之间的权值矩阵与下一层的误差矩阵的乘积
        	else:
        		self.output_errors.append(numpy.dot((self.weight[self.numNeuronLayers-i]).T, 
        											self.output_errors[i-1]))
        #反向传播
        for i in range(self.numNeuronLayers):
        	#权值更新规则为之前权值+学习率*误差*第二层输出*（1-第二层输出）*第一层输出
        	#f(x)*（1-f(x)）即为激活函数f(x)的导函数
        	self.weight[self.numNeuronLayers-i-1] += self.learningrate * numpy.dot((self.output_errors[i] 
                * self.outputs[-1-i] * (1.0 - self.outputs[-1-i])), numpy.transpose(self.outputs[-1-i-1]))
    def test(self,test_inputnodes,test_labels):
        inputs = numpy.array(test_inputnodes,ndmin=2).T
        #走一遍前向传播得到输出
        for i in range(self.numNeuronLayers):
        	temp_inputs=numpy.dot(self.weight[i],inputs)
        	temp_outputs=self.activation_function(temp_inputs)
        	inputs=temp_outputs
        #返回模型输出结果是否与测试用例标签一致
        return list(inputs).index(max(list(inputs)))==list(test_labels).index(1)
    def output(self,test_inputnodes,f):
        inputs = numpy.array(test_inputnodes,ndmin=2).T
        #走一遍前向传播得到输出
        for i in range(self.numNeuronLayers):
            temp_inputs=numpy.dot(self.weight[i],inputs)
            temp_outputs=self.activation_function(temp_inputs)
            inputs=temp_outputs
        
            f.write(str(list(inputs).index(max(list(inputs))))+' ')

if __name__ == '__main__': 
    learning_rate = 0.1
    images_data,labels=load("C:\\Users\\Anonymous\\Documents\\机器学习\\作业四赵虎201600301325", kind='train')
    test_images_data,test_labels=load("C:\\Users\\Anonymous\\Documents\\机器学习\\作业四赵虎201600301325", kind='t10k')
    ls=[784,30,10]
    n=neuralNetwork(2,ls,0.3)
    for i in range(5):
    	# if i/6==0:
    	# 	n.learningrate=n.learningrate/2len(images_data)
    	for i in range(len(images_data)):
    		n.update(images_data[i],labels[i])
    count=0
    f=open("classificationforNN.txt",'w')
    for i in range(len(images_data)):     
        # n.output(test_images_data[i],f)
        if n.test(test_images_data[i],test_labels[i]):
            count+=1
    print(count/10000)




