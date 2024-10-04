# learning_classes.py

# Define a Car class, (capitalised)
class Car:
    # We define a 'self' method function of the class, containing attributes
    def __init__(self, colour: str, position: float) -> None:
        # This has 2 arguments we need to specify when we create a Car instance
        self.colour = colour
        self.position = position

    # We define a 'drive' method
    def drive(self, distance) -> None:
        self.position += distance
        print(f"{self.colour} car now at position {self.position}")

# Package script into 'main'
def main_function():
    # Create an instance of the Car class
    bmw = Car("white", 5)

    # Print the colour of the bmw Car
    print(f"The BMW has a {bmw.colour} colour")

    # Drive the bmw a few metres down the road
    bmw.drive(5.4)

# Call main_function 
if __name__ == '__main__':
    main_function()