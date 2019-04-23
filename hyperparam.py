is_training = True
maxdisp = 192
learning_rate = 0.0001
is_server = True

def ps___(name, tensor):
    print(name, str(tensor.shape))