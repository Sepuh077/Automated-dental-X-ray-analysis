import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader


class Model:
    def __init__(self, model, train_input, train_label, test_input, test_label):
        self.model = model
        self.train_input = [train_input] if isinstance(train_input, torch.Tensor) else train_input
        self.train_label = train_label
        self.test_input = [test_input] if isinstance(test_input, torch.Tensor) else test_input
        self.test_label = test_label

        self.train_loss_stack = []
        self.train_acc_stack = []
        self.test_loss_stack = []
        self.test_acc_stack = []

        self.optimizer = None
        self.loss_f = None
        self.metrics = None
        self.is_compiled = False

    def draw_plots(self):
        plt.clf()
        figure, axis = plt.subplots(2, 1)
        axis[0].plot(self.train_loss_stack, label='Սովորեցվող')
        axis[0].plot(self.test_loss_stack, label='Թեստավորվող')
        axis[0].set_title('Կորուստ')
        axis[0].grid(axis='y')

        axis[1].plot(self.train_acc_stack, label='Սովորեցվող')
        axis[1].plot(self.test_acc_stack, label='Թեստավորվող')
        axis[1].set_title('Ճշգրտություն')
        axis[1].grid(axis='y')

        figure.set_size_inches(10, 8, forward=True)

        plt.legend()
        plt.show()

    def compile(self, optimizer=torch.optim.Adam, lr=0.001, loss=torch.nn.CrossEntropyLoss(), metrics=['loss', 'acc']):
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.loss_f = loss
        self.metrics = metrics

        self.is_compiled = True

    def get_accuracy(self, out, pred):
        return torch.sum(torch.argmax(out, axis=-1) == torch.argmax(pred, axis=-1)).item() / out.size(0) * 100
    
    def get_prediction(self):
        self.model.eval()

        pred = self.model( self.test_input if len(self.test_input) > 1 else self.test_input[0] )

        return pred

    def get_test_loss(self):
        pred = self.get_prediction()

        return self.loss_f(self.test_label, pred).item()

    def get_test_info(self):
        pred = self.get_prediction()

        return self.loss_f(self.test_label, pred).item(), self.get_accuracy(self.test_label, pred)

    def train(self, epochs, batch_size):
        for epoch in range(1, epochs+1):
            self.model.train()
            train_loader = DataLoader(
                range(self.train_label.size(0)), 
                batch_size=batch_size,
                shuffle=True,
                drop_last=True
            )

            count = len(train_loader)

            running_loss = 0
            running_acc = 0

            for i, batch_indxs in enumerate(train_loader):
                input_data = [ x[batch_indxs] for x in self.train_input ]
                out = self.train_label[batch_indxs]

                pred = self.model(input_data if len(input_data) > 1 else input_data[0])

                loss = self.loss_f(out, pred)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                running_loss += loss.item()

                accuracy = self.get_accuracy(out, pred)

                running_acc += accuracy

                procent = int((i + 1) / count * 100)

                print(f'EPOCH [{epoch}/{epochs}], {"=" * int(procent / 5)}> {procent}%, Loss = {loss.item()}, Accuracy = {accuracy}', end='\r')

            running_loss /= count
            running_acc /= count

            test_loss, test_acc = self.get_test_info()

            print(f'\nLoss = {running_loss:.5f}, Accuracy = {running_acc:.3f}%\nTest loss = {test_loss:.5f}, Test accuracy = {test_acc:.3f}%')

            self.train_acc_stack.append(running_acc)
            self.train_loss_stack.append(running_loss)
            self.test_acc_stack.append(test_acc)
            self.test_loss_stack.append(test_loss)
