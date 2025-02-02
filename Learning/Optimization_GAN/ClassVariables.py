# create example about class variables and changing it from objects

class Car:
    owner = 0

    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year
        self.speed = 0
        Car.owner += 1  # this will increase each time new car is created

    def accelerate(self, amount):
        self.speed += amount
        return self.speed

    def brake(self, amount):
        self.speed -= amount
        return self.speed

    def get_info(self):
        return f"{self.year} {self.make} {self.model} with speed {self.speed}"


# Create an instance of Car
car1 = Car("Toyota", "Corolla", 2015)

# Access and change class variables
print(car1.speed)
print(car1.accelerate(10))
print(car1.accelerate(20))

print(car1.get_info())


# Change class variable from object
car1.owner = 2
print(car1.owner)
# the class variable should be changed from the class itself  not from instance variable
print(Car.owner)

car2 = Car("Toyota", "camera", 2018)

print("----------------All see the updates----------------")
print(Car.owner)
print(car2.owner)
print(car1.owner)
