# python programming practice

# write a program thar reads product names and prices storing them in a dictionary
#products = {}
#n = int(input("Enter number of products: "))
#for _ in range(n):
 #   name = input("Enter product name: ")
  #  price = float(input("Enter product price: "))
   # products[name] = price

#print("Product names:", list(products.keys()))
#print("Prices:", list(products.values()))
#if "sugar" in products:
 #   print(f"Price of sugar: {products['sugar']}")
#else:
 #   print("Product not found")

















# write a program to read triangle length and width then claculate the hypothenuse
import math
def read_dimentions():
    length = float(input("Enter the mength of the triangle:"))
    width = float(input("Enter the width of the triangle:"))
    return length, width
def calculate_hypothenuse(length,width):
    return math.sqrt(length**2 + width**2)
length, width = read_dimentions()
hypothenuse = calculate_hypothenuse(length, width)
print("Hypothenuse:", hypothenuse)