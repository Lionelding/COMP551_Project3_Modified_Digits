import numpy as np
from neuron import Neuron
from layer import Layer

class Network:
	def __init__(self, thres, L_rate, hidden_num, hidden_neuron_num, outer_neuron_num, hidden_weight, outer_weight ):

		self.hidden_num=hidden_num
		self.hidden_neuron_num=hidden_neuron_num
		self.outer_neuron_num=outer_neuron_num
		self.layer1=Layer(hidden_neuron_num, 1, 1)
		self.layer2=Layer(outer_neuron_num, 2, 1)
		self.Input=[]
		self.L_rate=L_rate
		self.thres=thres
		self.hidden_weight=hidden_weight
		self.outer_weight=outer_weight
		self.layer1_out=1.0
		self.layer2_out=0.0
		self.delta_Et_over_Neuron_net2=[0.0]*outer_neuron_num
		self.delta_E_over_Neuron_net1=[0.0]*hidden_neuron_num

		# Assign weights to hidden layer
		for i in range (0,hidden_neuron_num):
			connection_num=len(hidden_weight)/hidden_neuron_num
			self.layer1.neuron_list[i].Weight=hidden_weight[0+i*connection_num:connection_num+i*connection_num]


		# Assign weights to outer layer
		for i in range (0,outer_neuron_num):
			connection_num=len(outer_weight)/outer_neuron_num
			self.layer2.neuron_list[i].Weight=outer_weight[0+i*connection_num:connection_num+i*connection_num]

	def get_information(self):
		print '_'*100
		print '0. Neuron Network information :'
		print 'Number of hidden layer is : ' + str(self.hidden_num)
		print ''
		print '1. Hidden Layer information : '
		print self.layer1.get_information()
		print ''
		print '2. Hidden Layer Neuron information'
		for i in range (0, self.hidden_neuron_num):
			self.layer1.neuron_list[i].get_information()
			print ''

		print '3. Outer Layer information : '
		print self.layer2.get_information()
		print ''
		print '4. Outer Layer Neuron information'
		for i in range (0, self.outer_neuron_num):
			self.layer2.neuron_list[i].get_information()
			print ''
		print '_'*100

	def Network_forward(self, Input):
		self.layer1_out=self.layer1.forward(Input)
		self.layer2_out=self.layer2.forward(self.layer1_out)
		return self.layer2_out

	def get_Network_error(self, target):
		Network_error=0.0
		for i in range (0, len(self.layer2_out)):
			Network_error=self.layer2.neuron_list[i].get_Error(target[i])+Network_error

		return Network_error


	def Network_backward(self, Input, target):
		self.Network_forward(Input)

        # 1. Error over Ouput layer Neuros
		delta_Et_over_Neuron_net2=[0]*len(self.layer2.neuron_list)
 		for o in range(0, len(self.layer2.neuron_list)):
			delta_Et_over_Neuron_net2[o]=self.layer2.neuron_list[o].derivative_Et_over_Neuron_net(target[o])

        # 2. Error over Hidden layer Neuros
		derivative_Et_over_hidden_Neuron_out = [0] * len(self.layer1.neuron_list)
		for h in range(0, len(self.layer1.neuron_list)):

            # derivative of the E over output of each hidden layer neuron
			E_over_hidden_Neuron_output = 0
			for o in range(0, len(self.layer2.neuron_list)):
				E_over_hidden_Neuron_output += delta_Et_over_Neuron_net2[o] * self.layer2.neuron_list[o].Weight[h]

			derivative_Et_over_hidden_Neuron_out[h] = E_over_hidden_Neuron_output * self.layer1.neuron_list[h].derivative_Neuron_out_over_Neuron_net()

        # 3. Weights Update in output layer
		for o in range(0, len(self.layer2.neuron_list)):
			for w_ho in range(0, len(self.layer2.neuron_list[o].Weight)):
				Error_over_weight2 = delta_Et_over_Neuron_net2[o] * self.layer2.neuron_list[o].derivative_Neuron_net_over_Weight(w_ho)
				self.layer2.neuron_list[o].Weight[w_ho] = self.layer2.neuron_list[o].Weight[w_ho]-self.L_rate * Error_over_weight2

        # 4. Weights Update in hidden layer
		for h in range(0, len(self.layer1.neuron_list)):
			for w_ih in range(0, len(self.layer1.neuron_list[h].Weight)):
				Error_over_weight1 = derivative_Et_over_hidden_Neuron_out[h] * self.layer1.neuron_list[h].derivative_Neuron_net_over_Weight(w_ih)
				self.layer1.neuron_list[h].Weight[w_ih] = self.layer1.neuron_list[h].Weight[w_ih]-self.L_rate * Error_over_weight1

	def test(self, test_in, test_target):
		print test_target
		print "self.Network_forward(test_in)"+str(self.Network_forward(test_in)[0])


		if ((test_target)==round(self.Network_forward(test_in)[0], 1)):
			print 'Correct Output is '+str(round(self.Network_forward(test_in)[0], 2))
			return 0
		else:
			print 'Incorrect Output is '+str(round(self.Network_forward(test_in)[0], 2))
			return 1 





	


