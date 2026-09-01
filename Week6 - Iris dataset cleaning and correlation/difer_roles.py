# =========================================
# 1. Ignoring unwanted values
# =========================================

student = ("Alex", 25, "Auckland")

name, _, city = student

print("Name:", name)
print("City:", city)


# =========================================
# 2. Using _ in loops when variable is not needed
# =========================================

for _ in range(3):
    print("Hello")


# =========================================
# 3. Separator for large numbers (readability)
# =========================================

population = 8_500_000
budget = 1_250_000_000

print("Population:", population)
print("Budget:", budget)


# =========================================
# 4. Last result in Python interpreter / Jupyter
# =========================================

10 + 5
print(_)   # Stores previous output (15)


# =========================================
# 5. Private/internal variable naming convention
# =========================================

class Student:
    def __init__(self):
        self._score = 90   # internal use convention

s = Student()
print(s._score)


# =========================================
# 6. Name mangling with double underscore
# =========================================

class Teacher:
    def __init__(self):
        self.__salary = 5000

t = Teacher()

# print(t.__salary)   # Error

print(t._Teacher__salary)   # Accessing mangled name


# =========================================
# 7. Special (magic/dunder) methods
# =========================================

class Book:
    def __init__(self, title):
        self.title = title

    def __str__(self):
        return f"Book: {self.title}"

b = Book("Python Basics")
print(b)


# =========================================
# 8. Underscore in variable names
# =========================================

student_name = "Mohammad"
course_name = "Data Analytics"

print(student_name)
print(course_name)