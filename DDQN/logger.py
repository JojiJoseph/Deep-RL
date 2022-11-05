class Logger:
    @staticmethod
    def print(key, value):
        print("| %20s | %20.20s |" % (key, value))

    @staticmethod
    def print_title(title):
        print("|  %-42s |" % title)

    @staticmethod
    def print_boundary():
        print("-" * 47)

    @staticmethod
    def print_double_boundary():
        print("|" + "=" * 45 + "|")
