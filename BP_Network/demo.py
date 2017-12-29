import mnist_loader
import Network



def main():
    training_data, validation_dat, test_data = mnist_loader.load_data_wrapper()
    print("training data")
    print(type(training_data))
    print(len(training_data))
    print(training_data[0][0].shape)
    print(training_data[0][1].shape)

    net = Network.Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0,test_data=test_data)


if __name__== "__main__":
    main()