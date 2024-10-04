# learning.py
# Learning classes and methods in python

# Define a Car class, (capitalised)
class Car:
    # We define a 'self' method function of the class, containing attributes
    def __init__(self, colour: str, position: int) -> None:
        # This has 2 arguments we need to specify when we create a Car instance
        self.colour = colour
        self.position = position


# Create an instance of the Car class
bmw = Car("white", 5)

# Print the colour of the bmw Car
print(f"The BMW has a {bmw.colour} colour")




