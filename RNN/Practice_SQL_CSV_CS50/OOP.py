class TracableList(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.history = []  # Keeps track of modifications

    def append(self, object):
        print(f"Appending {object}")
        self.history.append(f"Appended {object}")  # Log the operation
        super().append(object)  # Call the parent class method

    def pop(self, index=-1):
        value = super().pop(index)  # Call the parent class method
        print(f"Popped {value}")
        self.history.append(f"Popped {value}")  # Log the operation
        return value

    def get_history(self):
        return self.history


# Example Usage
li = TracableList()
li.append('apple')
li.append('banana')
li.pop()
print(li)
print("History:", li.get_history())
