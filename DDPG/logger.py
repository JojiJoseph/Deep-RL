class Logger:
    @staticmethod
    def print(key, value):
        print("| %20s | %20s |"%(key,value))
    def print_boundary():
        print("-"*47)