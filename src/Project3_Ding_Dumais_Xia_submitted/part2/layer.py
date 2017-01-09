from neuron import Neuron

class Layer:

	def __init__ (self,neuron_num,layer_num,bias_b):
		self.neuron_num=neuron_num
		self.layer_num=layer_num
		self.neuron_list=[]
		self.set_neurons(bias_b)
		self.Neuron_out_list=[0.0]*neuron_num
		self.bias_b=bias_b

	def get_information(self):
		print 'This is Layer number : '+str(self.layer_num)
		print 'Number of Neurons inside : '+str(self.neuron_num)
		print 'Nurons are : '+ str(self.neuron_list)
		print 'The output of each neuron in this layer are' + str(self.Neuron_out_list)
		print 'The bias_b for each neuron is :'+str(self.bias_b)

	#Instantiate each neuron inside this layer by assigning a bias value and Put the neuron into this layer 	
	def set_neurons(self,bias_b):
		for i in range (0, self.neuron_num):
			self.neuron_list.append(Neuron(bias_b))
		return

	# The return value is a vector Neuron_out value for each Neuron
	def forward(self, Input):

		outputs = []
		# connetion_num=len(Weight)/self.neuron_num
		# for i in range (0, self.neuron_num):
		# 	temp=self.neuron_list[i].get_out(Input, Weight[0+connetion_num*i:connetion_num+connetion_num*i])		
		# 	self.Neuron_out_list[i]=temp
		# return self.Neuron_out_list
		for neuron in self.neuron_list:
			outputs.append(neuron.get_out(Input))
		return outputs




