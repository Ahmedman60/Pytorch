class ValidatedSet(set):
    def __init__(self, *args, validators=None, **kwargs):
        # Initialize validators
        self.validators = validators if isinstance(validators, list) else list(
            validators) if validators is not None else []

        # If arguments are passed, validate them
        if args:
            # Unpack the elements (assuming only one iterable argument)
            # args is tuple of values  you care about first value which is the list or iterable argument.
            if len(args) != 1 or not hasattr(args[0], '__iter__'):
                raise ValueError("Expected exactly one iterable argument")
            elements = args[0]
            self.validate_many(elements)

        # Call parent constructor
        super().__init__(*args, **kwargs)

    def validate_one(self, element):
        """Validate a single element using all validators."""
        for validator in self.validators:
            if not validator(element):
                raise ValueError(f"Validation failed for element: {element}")

    def validate_many(self, elements):
        """Validate multiple elements."""
        for element in elements:
            self.validate_one(element)

    def add(self, element):
        """Validate and add a single element to the set."""
        self.validate_one(element)
        super().add(element)  # Use the parent `set`'s add method


def is_positive(x):
    return x > 0


def is_even(x):
    return x % 2 == 0


validators = [is_positive, is_even]
my_set = ValidatedSet([2, 4, 6], validators=validators)

print("Initial set:", my_set)


my_set.add(8)
print("After adding 8:", my_set)  # Output: {2, 4, 6, 8}

try:
    my_set.add(-2)  # Fails validation
except ValueError as e:
    print(e)  # Output: Validation failed for element: -2

try:
    my_set.add(5)  # Fails validation
except ValueError as e:
    print(e)  # Output: Validation failed for element: 5

print("Final Set: ", my_set)
