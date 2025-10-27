trainset = 'bigvul'
partset = 'reveal'
testset = 'juliet'
under = 1.0
selection = "random"
seed = 123456
indices_path = "./storage/indices/bigvul_line_juliet_line_123456_under0.8_over_accumulated_margin/coreset_index.txt"

def set_trainset(name):
    global trainset
    trainset = name

def set_partset(name):
    global partset
    partset = name

def set_testset(name):
    global testset
    testset = name

def set_under(under_rate):
    global under
    under = under_rate

def set_selection(selection_method):
    global selection
    selection = selection_method

def set_seed(seed_value):
    global seed
    seed = seed_value

def set_indices_path(path):
    global indices_path
    indices_path = path

def get_trainset():
    return trainset

def get_partset():
    return partset

def get_testset():
    return testset

def get_under():
    return under

def get_selection():
    return selection

def get_indices_path():
    return indices_path

def get_seed():
    return seed