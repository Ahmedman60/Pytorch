# Initialize an empty bitmask
bitmask = 0

# Add numbers 3, 5, and 7 to the bitmask
bitmask |= (1 << 3)  # Add number 3 (set the 3rd bit)
bitmask |= (1 << 5)  # Add number 5 (set the 5th bit)
bitmask |= (1 << 7)  # Add number 7 (set the 7th bit)

# Print the bitmask in binary form
print("Bitmask after adding 3, 5, and 7:", bin(bitmask))

# Remove number 5 from the bitmask
bitmask &= ~(1 << 5)

# Print the bitmask after removal of 5
print("Bitmask after removing 5:", bin(bitmask))

# Check if number 3 exists
if bitmask & (1 << 3):
    print("Number 3 exists in the bitmask")
else:
    print("Number 3 does not exist in the bitmask")

# Check if number 5 exists
if bitmask & (1 << 5):
    print("Number 5 exists in the bitmask")
else:
    print("Number 5 does not exist in the bitmask")

# Check if number 7 exists
if bitmask & (1 << 7):
    print("Number 7 exists in the bitmask")
else:
    print("Number 7 does not exist in the bitmask")
