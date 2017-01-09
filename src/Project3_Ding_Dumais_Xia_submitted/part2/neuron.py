import numpy as np

class Neuron:

	def __init__(self, bias_b):
		self.bias_b = bias_b
		self.Weight=[]
		self.net=0.0
		self.out=0.5
		self.Error=0.0
		self.Update=[]
		self.Input=[]

	def get_net(self):
		net=0.0
		for i in range (0,len(self.Weight)):
			net=self.Weight[i]*self.Input[i] + net
		self.net=net+self.bias_b
		return self.net

	def get_out(self,Input):
		self.Input = Input		
		self.net=self.get_net();
		self.out=1.0/(1.0+np.exp(-self.net))
		return self.out

	def get_Error(self,target):
		Error=((target-self.out)**2)*0.5
		self.Error=Error
		return Error

	def get_information(self):
		print 'bias_b is : ' + str(self.bias_b)
		print 'Weight is : ' + str(self.Weight)
		print 'Input is : ' + str(self.Input)
		print 'Net is : ' + str(self.net)
		print 'Out is : ' + str(self.out)
		print 'Error is :' + str(self.Error)
		print 'Update is :' + str(self.Update)


	# result is eqaul to Part1* Part2* Part3	
	def derivative_Et_over_Weight(self,target):
		Update = (self.derivative_Et_over_Neuron_out, target)*(self.derivative_Neuron_out_over_Neuron_net)*(self.derivative_Neuron_net_over_Weight(self,idx))
		self.Update=Update
		return Update

	# result is equal to Part1* Part2
	def derivative_Et_over_Neuron_net(self, target):
		return (self.derivative_Et_over_Neuron_out(target))*(self.derivative_Neuron_out_over_Neuron_net())


	# 	return part1
	# Part 1
	def derivative_Et_over_Neuron_out(self,target):
		print 'target is : '+str(target)
		print 'out is : '+str(self.out)
		return (-1)*(target-self.out)


	# Part 2
	# just matches to this neuron out to this Nueron net
	def derivative_Neuron_out_over_Neuron_net(self):
		return self.out*(1-self.out)

	# Part 3
	# matches to the weight we want to change
	def derivative_Neuron_net_over_Weight(self,idx):
		return self.Input[idx]