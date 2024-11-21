
# # List Data Strcture
# # Feature	List
# # Definition	A mutable, ordered sequence of items.
# # Syntax	Defined with square brackets: [ ]
# # Mutability	Mutable: Items can be added, removed, or changed.
# # Performance	Slower due to dynamic nature and mutability.
# # Use Cases	Ideal for collections requiring frequent changes.
# # Methods	Provides methods like append(), extend(),
# # Hashability	Not hashable; cannot be used as dictionary keys.
# # Iterability	Fully iterable like a tuple.	F
# import inspect
# a = [1, 2, 3, 4, 5]
# a.append(5)
# a.extend([2, 4, 51, 2, 3])
# print(a)

# # Important syntax to Remember but try to avoid using them for readablity.
# # Same as Append
# a[len(a):] = [777]
# # Same as extend
# a[len(a):] = list("Fathallah")
# print(a)

# # Insert
# a.insert(0, "Feto")
# print(a)

# # List Remove
# a.remove(2)
# print(a)
# # pop  remove and return the item in specific index
# result = a.pop(1)
# print("After Poping")
# print(a)
# print(result)
# #
# # a.clear()
# # print(a)
# # Get me the index of 2 start search from index 7 to the end of the array.
# print(a.index(2, 7, -1))


# def my_function(x, y=5, z=None):
#     pass


# sig = inspect.signature(my_function)
# sig = inspect.signature(list.index)
# print(sig)
# for param in sig.parameters.values():
#     print(param.name, param.default)

# # ------------------------------------------
# print(a.count(2))
# numbers = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]

# # Sorting without key and reverse
# print(sorted(numbers))  # Output: [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9]

# # Sorting with reverse
# # Output: [9, 6, 5, 5, 5, 4, 3, 3, 2, 1, 1]
# print(sorted(numbers, reverse=True))

# # Sorting with key (sorting by absolute value, just for example)
# # Output: [5, 5, 5, 4, 6, 3, 3, 2, 1, 1, 9]
# print(sorted(numbers, key=lambda x: abs(x - 5)))


# List of strings
words = ["banana", "apple", "kiwi", "strawberry", "grape"]

# Sorting the list by the length of each word
sorted_words = sorted(words, key=len)

print("Sorted by length:", sorted_words)


# List of tuples
data = [(1, "apple"), (3, "banana"), (2, "cherry"), (4, "date")]

# Sorting the list by the second element (index 1) of each tuple
sorted_data = sorted(data, key=lambda x: x[1])

print("Sorted by second element:", sorted_data)


# List of dictionaries
students = [
    {"name": "Alice", "age": 25},
    {"name": "Bob", "age": 20},
    {"name": "Charlie", "age": 22}
]

# Sorting the list of dictionaries by the 'age' key
sorted_students = sorted(students, key=lambda x: x["age"])

print("Sorted by age:", sorted_students)

# List of strings
words = ["banana", "Apple", "orange", "Mango", "grape"]

# Sorting the list in a case-insensitive manner
sorted_words = sorted(words, key=str.lower)

print("Case-insensitive sorted words:", sorted_words)

# words.pop(0)
