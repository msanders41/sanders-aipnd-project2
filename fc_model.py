import torch
from torch import nn
import torch.nn.functional as F

class Network(nn.Module):
	"""
	
	Creates fully connected NN with specified nodes per layer
	----------
	input_size: (int) number of nodes in input layer
	output_size: (int) number of nodes in output layer
	hidden_layers: (int) ordered list of nodes in successive hidden layers
	drop_p: (float) probability of dropout of each node (Default=0.5)

	"""
	def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
		super().__init__()
		
		self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

		layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
		for h1, h2 in layer_sizes:
			self.hidden_layers.extend([nn.Linear(h1, h2)])

		self.output_layer = nn.Linear(hidden_layers[-1], output_size)
		self.dropout = nn.Dropout(p=drop_p)

	def forward(self, x):
        for layer in self.hidden_layers:
        	x = F.relu(layer(x))
        	x = self.dropout(x)

        x = self.output_layer(x)

        return F.log_softmax(x, dim=1)


def validation(model, testloader, criterion):
	test_loss = 0
	accuracy = 0

	# Turn on evaluation mode (no dropout) and turn off gradient computation
	model.eval()
	
	for images, labels in testloader:

		# Flatten each image tensor
		images = images.view(images.shape[0], -1)

		# Forward pass to compute losses
		log_ps = model.forward(images)
		loss = criterion(log_ps, labels)
		test_loss += loss.item()

		# Calculate accuracy: convert to probabilities, determine top classes, how many correct
		ps = torch.exp(log_ps)
		top_ps, top_class = ps.topk(1, dim=1)
		equals = top_class == labels.view(*top_class.shape)
		accuracy += equals.type_as(torch.FloatTensor()).mean()

	return test_loss, accuracy

def train(model, trainloader, testloader, criterion, optimizer, epochs=5, print_every=40):	
	steps = 0
	running_loss = 0
	for e in range(epochs):		
		model.train()
		for images, labels in trainloader:
			steps += 1

			# Flatten the image tensor
			images = images.view(images.shape[0], -1)

			# Forward pass through network, compute losses, backward pass, update optimizer
			optimizer.zero_grad()
			log_ps = model.forward(images)
			loss = criterion(log_ps, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()

			# Validation loop every specified number of training steps
			if steps %% print_every == 0:
				model.eval()
				with torch.no_grad():
					test_loss, accuracy = validation(model, testloader, criterion)

				# Print epoch, training loss, test loss, accuracy
				print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

			# Reset the running_loss and turn back to train mode
			running_loss = 0
			model.train()

		
